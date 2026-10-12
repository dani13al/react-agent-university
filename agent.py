import importlib

import json

import os

import pkgutil

import re

from datetime import datetime



import tiktoken

from colorama import Fore, Style, init

from dotenv import load_dotenv

from openai import OpenAI, BadRequestError



from tools.base_tool import BaseTool

from utils.message import Message





class ReActAgent:

    def __init__(self):

        load_dotenv(".env.local", override=True)



        self.tools = {}

        self.messages = []

        self.max_iterations = 10

        self.current_iteration = 0

        self.old_chats_summary = ""

        self.messages_to_summarize = 3

        self.llm_max_tokens = 500

        self.max_messages_tokens = 1000

        self.model = os.getenv("MODEL_NAME")

        self.client = self.get_llm_client()

        self.system_prompt = self.load_prompt("prompts/system_prompt.txt")

        self.summary_prompt = self.load_prompt("prompts/summary_prompt.txt")

        try:

            self.tokenizer = tiktoken.encoding_for_model(self.model)

        except KeyError:

            self.tokenizer = tiktoken.get_encoding("cl100k_base")



        # Register tools dynamically

        self.register_tools()



    def get_llm_client(self):

        return OpenAI(

            api_key=os.getenv("GROQ_API_KEY"),

            base_url="https://api.groq.com/openai/v1",

            timeout=60.0,

        )



    def register_tools(self):

        """Dynamically registers all available tools."""

        tool_modules = [name for _, name, _ in pkgutil.iter_modules(["tools"])]



        for module_name in tool_modules:

            try:

                module = importlib.import_module(f"tools.{module_name}")

                for attr_name in dir(module):

                    attr = getattr(module, attr_name)

                    if (

                        isinstance(attr, type)

                        and issubclass(attr, BaseTool)

                        and attr is not BaseTool

                    ):

                        tool_instance = attr()

                        self.tools[tool_instance.name.lower()] = tool_instance

            except Exception as e:

                print(

                    f"\n{Fore.RED}[ERROR] Failed to register tool {module_name}: {e}{Style.RESET_ALL}\n"

                )



    def get_tools(self):

        """Returns a formatted string listing available tools."""

        return "\n".join(

            [f"{tool.name}: {tool.description}" for tool in self.tools.values()]

        )



    def num_tokens_from_messages(self, messages):

        """Return the number of tokens used by a list of messages"""

        value = "".join(

            [msg["content"] for msg in messages]

        )  # Extract content from chat messages

        encoded = self.tokenizer.encode(value)  # Tokenize the extracted text



        return len(encoded)



    def num_tokens_from_text(self, text):

        """Return the number of tokens used by the given text."""

        encoded = self.tokenizer.encode(text)



        return len(encoded)



    def load_prompt(self, path):

        """Returns a prompt from a file."""

        with open(path, "r") as file:

            return file.read() if file else ""



    def add_message(self, role, content):

        """Add a message to the messages list."""

        self.messages.append(Message(role=role, content=content))



    def think(self):

        """Think and decide based on the response from OpenAI."""

        self.current_iteration += 1



        if self.current_iteration > self.max_iterations:

            print(

                f"\n{Fore.YELLOW}Reached maximum iterations. Stopping.{Style.RESET_ALL}"

            )

            self.add_message(

                "assistant",

                "I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations.",

            )

            return



        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt = self.system_prompt.format(tools=self.get_tools(), date=current_date)



        response = self.get_llm_response(prompt)

        self.add_message("assistant", response)



        # Print each thought immediately

        self.format_output(response)



        # Continue processing actions

        self.determine_action(response)



    def determine_action(self, response):

        """Decide on the next action based on the response, without using regex."""



        if "Final Answer:" in response:

            return



        # Find "Action:" in the response

        action_start = response.find("Action:")



        if action_start == -1:

            print(

                f"{Fore.YELLOW}No action or final answer found in the response.{Style.RESET_ALL}"

            )

            return



        # Extract the action line

        action_line = response[action_start:].split("\n")[0].strip()



        # Remove the "Action:" prefix

        action_parts = action_line.replace("Action:", "").strip().split(":", 1)



        if len(action_parts) < 2:

            print(

                f"{Fore.RED}Error: Action format is incorrect: {action_line}{Style.RESET_ALL}"

            )

            return



        tool_name = action_parts[0].strip().lower()

        query = action_parts[1].strip()



        # Handle calculator JSON validation

        if tool_name == "calculator":

            try:

                json_data = json.loads(query)



                if "operation" not in json_data:

                    print(

                        f"{Fore.RED}Error: Missing 'operation' in calculator JSON: {query}{Style.RESET_ALL}"

                    )

                    return



                query = json.dumps(json_data)



            except json.JSONDecodeError:

                print(

                    f"{Fore.RED}Error: Invalid JSON input for calculator: {query}{Style.RESET_ALL}"

                )

                return



        # Execute the extracted action

        self.execute_action(tool_name, query)



    def execute_action(self, tool_name, query):
        """Execute a tool once per unique query and keep observations concise."""
        key = (tool_name, " ".join(str(query).casefold().split()))
        cache = getattr(self, "_tool_observations", None)
        if cache is None:
            cache = self._tool_observations = {}

        # One forced finalization attempt is allowed. Never recurse endlessly.
        if getattr(self, "_force_final", False):
            text = "\n".join(cache.values())[:2400]
            self.add_message(
                "assistant",
                "Final Answer: The agent could not reliably finish its summary. "
                "Here are the available tool results (check the source URLs):\n" + text,
            )
            return

        if key in cache:
            self.add_message(
                "system",
                "Observation: Duplicate tool call blocked. The earlier result was: "
                + cache[key]
                + "\nThe current question already has evidence. "
                "Give the Final Answer now; do not call tools again.",
            )
            self._force_final = True
            self.think()
            return

        tool = self.tools.get(tool_name)
        if tool is None:
            self.add_message("system", f"Observation: Tool '{tool_name}' is unavailable.")
            self.think()
            return

        try:
            result = tool.run(query)
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"

        # Tavily often returns long page excerpts. Keep titles, URLs and a
        # short snippet so the LLM can still answer and attribute its sources.
        if tool_name == "web_search" and isinstance(result, list):
            compact = []
            for item in result[:3]:
                if not isinstance(item, dict):
                    continue
                compact.append({
                    "title": str(item.get("title", ""))[:200],
                    "content": " ".join(str(item.get("content", "")).split())[:800],
                    "url": str(item.get("url", ""))[:350],
                    **({"error": str(item["error"])[:250]} if "error" in item else {}),
                })
            result_text = json.dumps(compact, ensure_ascii=False)
        else:
            result_text = str(result)[:2300]

        observation = f"Observation: {tool_name} tool output: {result_text}"
        cache[key] = observation
        self.add_message("system", observation)
        print(f"{Fore.CYAN}\n[SYSTEM]:{Style.RESET_ALL} {observation}\n")
        self.think()

    def format_output(self, response):

        """Format output for better readability."""

        response = re.sub(

            r"Final Answer:", f"{Fore.RED}\n[FINAL ANSWER]:{Style.RESET_ALL}", response

        )

        response = re.sub(

            r"Action:", f"{Fore.YELLOW}\n[ACTION]:{Style.RESET_ALL}", response

        )

        response = re.sub(

            r"PAUSE", f"{Fore.MAGENTA}\n[PAUSE]:{Style.RESET_ALL}", response

        )



        print(f"{Fore.GREEN}\n[ASSISTANT]:{Style.RESET_ALL} {response}\n")



    def get_llm_response(self, prompt):
        """Ask Groq for one ReAct step, recovering valid rejected native tool calls."""
        history = [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]
        self.memory_management(history)
        # Memory management may have deleted messages.
        history = [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]

        available = sorted(self.tools.keys())
        instructions = (
            "You control a ReAct agent. Decide ONE next step per response. "
            "The Python application executes tools; do NOT call native functions. "
            "Reply ONLY in the required JSON schema.\n"
            f"Today: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Registered tools:\n{self.get_tools()}\n"
            "If the user asks to use a registered tool, choose kind=action. "
            "For kind=action, use a registered tool_name and place the "
            "tool input in tool_input; leave answer empty. "
            "For calculator, tool_input MUST be a JSON-encoded string like "
            "{\"operation\":\"add\",\"params\":{\"a\":12,\"b\":8}}. "
            "For wikipedia, web_search, and weather, tool_input is plain text. "
            "After a tool observation, use it to decide whether another step "
            "is needed. To finish, set kind=final, tool_name=none, "
            "tool_input empty, and put the complete answer in answer. "
            "Never invent tool results. Do not include formatting tags, "
            "'Action:', 'PAUSE', or 'Thought:' in the JSON field values."
        )
        instructions += (
            "\nFINALIZATION IS IMPORTANT: After a successful tool Observation "
            "that contains the requested answer, you MUST choose kind=final "
            "immediately. For a single air-distance question, one successful "
            "web_search is enough. Report the approximate distance and the "
            "source URL from the Observation. Never repeat the same tool call "
            "with the same input. For multi-part questions, only perform "
            "actions needed for information not already observed. "
            "Ignore unrelated long page excerpts. "
        )
        if getattr(self, "_force_final", False):
            instructions += (
                "\nSTOP SEARCHING. You already have tool results. "
                "Your next answer MUST be kind=final with a concise answer "
                "using only the observations, including any available URLs. "
            )
        if self.old_chats_summary:
            instructions += "\nPrevious conversation summary:\n" + self.old_chats_summary

        schema = {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["action", "final"]},
                "tool_name": {"type": "string", "enum": ["none"] + available},
                "tool_input": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["kind", "tool_name", "tool_input", "answer"],
            "additionalProperties": False,
        }

        if getattr(self, "_force_final", False):
            schema["properties"]["kind"]["enum"] = ["final"]
            schema["properties"]["tool_name"]["enum"] = ["none"]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": instructions}] + history,
                max_tokens=max(1024, self.llm_max_tokens),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "react_next_step",
                        "strict": True,
                        "schema": schema,
                    },
                },
                extra_body={"reasoning_effort": "low"},
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            if not raw:
                return "Final Answer: Model returned an empty response."
            step = json.loads(raw)

        except BadRequestError as error:
            # Groq sometimes rejects an attempted native tool call with HTTP 400,
            # even though tools were not explicitly supplied to the API.
            # Accept only known tools / the ReAct schema and validate inputs.
            body = error.body if isinstance(error.body, dict) else {}
            details = body.get("error", body)
            if (
                not isinstance(details, dict)
                or details.get("code") != "tool_use_failed"
            ):
                raise

            try:
                attempted = details.get("failed_generation")
                if isinstance(attempted, str):
                    attempted = json.loads(attempted)
                if not isinstance(attempted, dict):
                    raise ValueError("Unexpected failed_generation format")

                native_name = attempted.get("name")
                native_args = attempted.get("arguments")
                if isinstance(native_args, str):
                    native_args = json.loads(native_args)
                if not isinstance(native_args, dict):
                    raise ValueError("Invalid native arguments")

                if native_name == "react_next_step":
                    step = native_args
                elif native_name in self.tools:
                    if native_name == "calculator":
                        tool_input = json.dumps(native_args, ensure_ascii=False)
                    elif native_name == "weather":
                        tool_input = (
                            native_args.get("query")
                            or native_args.get("city")
                            or native_args.get("location")
                        )
                    elif native_name == "wikipedia":
                        tool_input = (
                            native_args.get("query")
                            or native_args.get("topic")
                            or native_args.get("title")
                        )
                    else:
                        tool_input = native_args.get("query")

                    if not isinstance(tool_input, str) or not tool_input.strip():
                        raise ValueError("Invalid native tool input")
                    step = {
                        "kind": "action",
                        "tool_name": native_name,
                        "tool_input": tool_input,
                        "answer": "",
                    }
                else:
                    raise ValueError("Unknown native tool call")
            except (ValueError, TypeError, KeyError):
                raise error

        if not isinstance(step, dict) or step.get("kind") not in ("action", "final"):
            return "Final Answer: Model returned an invalid ReAct step."
        if step["kind"] == "final":
            answer = step.get("answer", "")
            if not isinstance(answer, str):
                return "Final Answer: Model returned an invalid answer."
            return "Final Answer: " + (answer.strip() or "No answer was provided.")

        name = step.get("tool_name")
        query = step.get("tool_input")
        if name not in self.tools:
            return "Final Answer: The requested tool is unavailable."
        if not isinstance(query, str) or not query.strip():
            return "Final Answer: The model provided an invalid tool input."
        query = query.strip()
        if "\n" in query or "\r" in query:
            return "Final Answer: The model provided an invalid tool input."

        if name == "calculator":
            try:
                data = json.loads(query)
                if (
                    not isinstance(data, dict)
                    or data.get("operation")
                    not in ("add", "subtract", "multiply", "divide", "power", "modulus")
                    or not isinstance(data.get("params"), dict)
                    or not all(k in data["params"] for k in ("a", "b"))
                ):
                    raise ValueError("Invalid calculator fields")
                query = json.dumps(data, ensure_ascii=False)
            except (ValueError, TypeError):
                return "Final Answer: The model provided invalid calculator JSON."

        return f"Action: {name}: {query}\nPAUSE"

    def summarize_old_chats(self, chats):

        """Summarizes old chat history and returns a concise summary response."""

        prompt = self.summary_prompt.format(chats=chats)

        messages = [{"role": "system", "content": prompt}]



        raw_response = self.client.chat.completions.create(

            model=self.model, messages=messages, max_tokens=self.llm_max_tokens

        )

        response = raw_response.choices[0].message.content



        return response.strip() if response else "No response from LLM"



    def get_indices(self, chat_history):

        """Extracts a specified number of consecutive user queries from the given chat history."""

        user_indices = [

            i for i, msg in enumerate(chat_history) if msg["role"] == "user"

        ]



        start_index = user_indices[0]

        end_index = user_indices[self.messages_to_summarize]



        return start_index, end_index



    def memory_management(self, chat_history):

        """Manages memory by summarizing and deleting old chat history"""

        try:

            user_messages = [msg for msg in chat_history if msg["role"] == "user"]

            if (

                len(user_messages) > self.messages_to_summarize

                and self.num_tokens_from_messages(chat_history)

                > self.max_messages_tokens

            ):

                indices = self.get_indices(chat_history)

                if indices:

                    start_index, end_index = indices

                    chats = chat_history[start_index:end_index]

                    new_summary = self.summarize_old_chats(chats)

                    print(

                        f"##### Tokens used by the old messages: {self.num_tokens_from_messages(chats)}"

                    )

                    # print("##### New Summary : ", new_summary)

                    if new_summary != "No response from LLM":

                        print(

                            f"##### Tokens used by the new summary: {self.num_tokens_from_text(new_summary)}"

                        )

                        self.old_chats_summary = (

                            f"{self.old_chats_summary} {new_summary}".strip()

                        )

                        print("##### Old messages summary : ", self.old_chats_summary)

                        del self.messages[start_index:end_index]

        except Exception as e:

            print(f"An error occurred during memory management: {e}")



    def execute(self, query):

        """Execute a user query and return the full Agent response."""

        self.current_iteration = 0
        self._tool_observations = {}
        self._force_final = False

        self.add_message("user", query)

        self.think()



        result_messages = []

        for message in self.messages[::-1]:

            if message.role == "user":

                break

            elif message.role != "user":

                result_messages.append(message)



        return result_messages[::-1]





if __name__ == "__main__":

    init(autoreset=True)

    react_agent = ReActAgent()



    while True:

        query = input(f"{Fore.CYAN}USER:{Style.RESET_ALL} ").strip()

        if query.lower() in ["exit", "quit"]:

            print(f"{Fore.YELLOW}Exiting the ReAct agent. Goodbye!{Style.RESET_ALL}")

            break



        result = react_agent.execute(query)

        # print('#### Result : ', result)

        print("\n" + "=" * 60 + "\n")
