"""ReAct Agent for Google Gemini (OpenAI-compatible API) with native tool calling and bounded multi-step execution.

Drop-in replacement for agent.py in ReAct-Agent-from-Scratch.
Requires the existing tools/ and utils/ modules and dependencies in requirements.txt.
Does not store or print API keys. Supports .env.local locally and environment
variables supplied through Streamlit Cloud secrets.
"""
from __future__ import annotations

import importlib
import json
import os
import pkgutil
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from tools.base_tool import BaseTool
from utils.message import Message


class ReActAgent:
    def __init__(self):
        load_dotenv(".env.local", override=True)
        self.model = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Put it in .env.local for local use, or Streamlit Secrets when deploying.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=25.0,
            max_retries=1,
        )
        self.tools: dict[str, BaseTool] = {}
        self.messages: list[Message] = []
        self.max_iterations = 14
        self.max_tool_calls = 11
        self.max_run_seconds = 100
        self.register_tools()

    def register_tools(self):
        """Find classes in the existing tools folder, skipping optional missing keys."""
        for _, name, _ in pkgutil.iter_modules(["tools"]):
            try:
                module = importlib.import_module(f"tools.{name}")
                for item in vars(module).values():
                    if (
                        isinstance(item, type)
                        and issubclass(item, BaseTool)
                        and item is not BaseTool
                        and item.__module__ == module.__name__
                    ):
                        tool = item()
                        self.tools[tool.name.lower()] = tool
            except Exception as exc:
                print(f"[Tool unavailable] {name}: {type(exc).__name__}")

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def _tool_definitions(self) -> list[dict[str, Any]]:
        definitions = []
        for name, tool in sorted(self.tools.items()):
            if name == "calculator":
                parameters = {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide", "power", "modulus"],
                        },
                        "params": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                    "required": ["operation", "params"],
                }
            else:
                parameters = {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query, topic, or city name"}},
                    "required": ["query"],
                }
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": getattr(tool, "description", f"Call {name}"),
                    "parameters": parameters,
                },
            })
        return definitions

    @staticmethod
    def _required_calls(question: str) -> dict[str, int]:
        """Minimum tool coverage for explicit multi-part requests (not a fixed city script)."""
        q = question.lower()
        if "distance" not in q or "weather" not in q:
            return {}
        requirements: dict[str, int] = {}
        legs = len(re.findall(r"\bfrom\s+.+?\s+to\s+", question, re.I))
        if "web_search" in q:
            requirements["web_search"] = 2 if legs >= 2 or "two distances" in q else 1
        if "weather" in q:
            requirements["weather"] = 3 if "three cities" in q or all(x in q for x in ("tokyo", "london", "new york")) else 1
        if "calculator" in q and ("add" in q or "sum" in q or "total" in q):
            requirements["calculator"] = 1
        return requirements

    @staticmethod
    def _compact_result(name: str, result: Any) -> str:
        if name == "web_search" and isinstance(result, list):
            trimmed = []
            for item in result[:2]:
                if not isinstance(item, dict):
                    continue
                trimmed.append({
                    "title": str(item.get("title", ""))[:130],
                    "content": " ".join(str(item.get("content", "")).split())[:450],
                    "url": str(item.get("url", ""))[:250],
                    **({"error": str(item["error"])[:150]} if "error" in item else {}),
                })
            return json.dumps(trimmed, ensure_ascii=False)
        return str(result)[:1100]

    @staticmethod
    def _parse_arguments(name: str, arguments: Any) -> tuple[str, Any]:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        if name == "calculator":
            operation = arguments.get("operation")
            params = arguments.get("params")
            if (
                operation not in ("add", "subtract", "multiply", "divide", "power", "modulus")
                or not isinstance(params, dict)
                or not all(k in params and isinstance(params[k], (int, float)) for k in ("a", "b"))
            ):
                raise ValueError("Invalid calculator arguments")
            return json.dumps({"operation": operation, "params": {"a": params["a"], "b": params["b"]}}), arguments
        query = arguments.get("query") or arguments.get("city") or arguments.get("topic")
        if not isinstance(query, str) or not query.strip() or "\n" in query or "\r" in query:
            raise ValueError("Invalid tool query")
        return query.strip(), arguments

    def _run_tool(self, name: str, arguments: Any, cache: dict, records: list) -> str:
        if name not in self.tools:
            return f"Error: tool '{name}' is not enabled on this server."
        try:
            query, parsed_args = self._parse_arguments(name, arguments)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return f"Error: invalid {name} arguments ({exc})."
        key = (name, query.casefold() if name != "calculator" else json.dumps(parsed_args, sort_keys=True))
        if key in cache:
            return "Duplicate tool call: result already obtained. " + cache[key]
        self.add_message("assistant", f"Action: {name}: {query}\nPAUSE")
        try:
            result = self.tools[name].run(query)
            output = self._compact_result(name, result)
        except Exception as exc:
            output = f"Tool failed: {type(exc).__name__}: {str(exc)[:180]}"
        cache[key] = output
        failed = (
            output.startswith("Error:")
            or output.startswith("Tool failed:")
            or '"error":' in output.lower()
            or "unable to fetch weather" in output.lower()
        )
        records.append({"tool": name, "query": query, "result": output, "success": not failed})
        self.add_message("system", f"Observation: {name} tool output: {output}")
        print(f"[TOOL] {name}({query[:100]}) => {output[:170]}")
        return output

    @staticmethod
    def _pending(requirements: dict, records: list) -> dict:
        counts: dict[str, int] = {}
        for rec in records:
            if rec["success"]:
                counts[rec["tool"]] = counts.get(rec["tool"], 0) + 1
        return {name: needed - counts.get(name, 0) for name, needed in requirements.items() if needed > counts.get(name, 0)}

    def _partial_answer(self, records: list, reason: str) -> str:
        if not records:
            return f"I couldn't complete the request ({reason}). No tool results were obtained."
        lines = [f"I couldn't complete every requested step ({reason}). Verified tool results so far:"]
        for rec in records[-8:]:
            result = rec["result"]
            if rec["tool"] == "web_search":
                try:
                    items = json.loads(result)
                    for item in items[:2]:
                        lines.append(f"- Search: {rec['query']} — {item.get('title', '')}. {item.get('content', '')[:180]} Source: {item.get('url', '')}")
                except (ValueError, TypeError):
                    lines.append(f"- Search {rec['query']}: {result[:220]}")
            else:
                lines.append(f"- {rec['tool']} ({rec['query']}): {result[:320]}")
        lines.append("Missing parts are not calculated or assumed. Please retry the incomplete part.")
        return "\n".join(lines)

    def execute(self, query: str) -> list[Message]:
        """Plan through model tool calls; execute and observe one tool at a time."""
        start_index = len(self.messages)
        self.add_message("user", query)
        tools_schema = self._tool_definitions()
        requirements = self._required_calls(query)
        system = (
            "You are a ReAct assistant with executable external tools. "
            "Use tool calls for external facts and calculations; never invent tool outputs. "
            "For multi-part questions, carry out all requested parts before answering. "
            "A distance from A to B and B to C requires TWO distinct searches, "
            "weather in three cities requires THREE distinct weather calls, "
            "and adding two distances requires calculator. "
            "Cite exact source URLs returned by web_search. Prefer concise tool queries. "
            "After each tool response decide what remains. Do not repeat the same tool call. "
            "When all requested work is done, answer in clear prose. "
            "If a tool fails, report exactly which data could not be obtained; do not fabricate it."
        )
        llm_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        cache: dict = {}
        records: list = []
        deadline = time.monotonic() + self.max_run_seconds
        calls = 0
        early_final_rejected = 0
        repeat_calls = 0
        final_answer = None
        stop_reason = "tool or model limit reached"

        for _ in range(self.max_iterations):
            if time.monotonic() >= deadline:
                stop_reason = "request time limit reached"
                break
            if calls >= self.max_tool_calls:
                stop_reason = "tool-call limit reached"
                break
            pending = self._pending(requirements, records)
            llm_messages.append({
                "role": "system",
                "content": (
                    "Progress: " + ("remaining successful calls: " + json.dumps(pending) if pending else "requested tool coverage met or no specific count required")
                    + ". Already executed: " + "; ".join(f"{r['tool']}({r['query'][:90]})" for r in records[-8:])
                    + ". Do not request the same tool with the same query twice."
                ),
            })
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": llm_messages,
                    "max_tokens": 2048,
                    "temperature": 0.1,
                }
                if tools_schema:
                    kwargs.update({"tools": tools_schema, "tool_choice": "auto"})
                completion = self.client.chat.completions.create(**kwargs)
                msg = completion.choices[0].message
            except BadRequestError as exc:
                stop_reason = f"Gemini returned HTTP {getattr(exc, 'status_code', '?')}. Check the selected model and tool schema."
                break
            except Exception as exc:
                stop_reason = f"model request failed: {type(exc).__name__}"
                break

            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                assistant_payload = {"role": "assistant", "content": msg.content,
                                     "tool_calls": [c.model_dump(exclude_none=True) for c in tool_calls]}
                # Some Gemini models return auxiliary metadata needed when continuing a tool call.
                extra_content = getattr(msg, "extra_content", None)
                if extra_content:
                    assistant_payload["extra_content"] = extra_content
                llm_messages.append(assistant_payload)
                for call in tool_calls:
                    if calls >= self.max_tool_calls:
                        break
                    tool_name = call.function.name
                    arguments = call.function.arguments
                    count_before = len(records)
                    result = self._run_tool(tool_name, arguments, cache, records)
                    if len(records) == count_before:
                        repeat_calls += 1
                    else:
                        repeat_calls = 0
                    llm_messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                    calls += 1
                if repeat_calls >= 3:
                    stop_reason = "model repeated a tool call too many times"
                    break
                continue

            answer = (getattr(msg, "content", None) or "").strip()
            if not answer:
                llm_messages.append({"role": "system", "content": "Your response was empty. Call a tool or give a final answer."})
                continue
            pending = self._pending(requirements, records)
            if pending and early_final_rejected < 2:
                early_final_rejected += 1
                llm_messages.append({"role": "assistant", "content": answer})
                llm_messages.append({
                    "role": "system",
                    "content": "The request is not finished. Still missing: " + json.dumps(pending)
                    + ". Use the appropriate tools. Do not give a final answer yet.",
                })
                continue
            if pending:
                stop_reason = "some required subtasks were not completed"
                break
            final_answer = answer
            break

        if not final_answer:
            final_answer = self._partial_answer(records, stop_reason)
        self.add_message("assistant", "Final Answer: " + final_answer)
        return self.messages[start_index:]


if __name__ == "__main__":
    agent = ReActAgent()
    print("ReAct agent ready. Type exit to leave.")
    while True:
        question = input("USER: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        for m in agent.execute(question):
            print(f"[{m.role}] {m.content[:1300]}")
