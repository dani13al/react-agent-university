import html

import streamlit as st

# Always import the active agent (Gemini v6), not the old backup.
from agent import ReActAgent


class WebApp:
    def __init__(self):
        self.agent = ReActAgent()
        self.initialize_ui()

    def initialize_ui(self):
        st.markdown(
            """
            <style>
                [data-testid="stChatMessage"] {
                    background-color: transparent !important;
                }
                .chat-message-container {
                    display: flex;
                    align-items: flex-start;
                    gap: 12px;
                    margin-bottom: 10px;
                    width: 100%;
                    max-width: 760px;
                }
                .chat-message {
                    padding: 12px 16px;
                    border-radius: 12px;
                    max-width: 85%;
                    overflow-wrap: anywhere;
                    line-height: 1.65;
                }
                .user-message { background-color: #D6EAF8; color: #15202b; }
                .assistant-message { background-color: #D5F5E3; color: #15202b; }
                .message-icon { font-size: 28px; min-width: 32px; }
                .message-text { white-space: normal; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h1 style='text-align: center;'>ReAct Agent Chat</h1>",
            unsafe_allow_html=True,
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "agent_logs" not in st.session_state:
            st.session_state.agent_logs = []

        with st.sidebar:
            st.title("Agent Tool Activity")
            for log in st.session_state.agent_logs:
                st.markdown(log)

        self.display_chat_history()
        self.handle_user_input()

    def display_chat_history(self):
        for message in st.session_state.chat_history:
            self.display_message(message["role"], message["content"])

    def display_message(self, role, content):
        is_user = role == "user"
        icon = "👩‍💼" if is_user else "🤖"
        message_class = "user-message" if is_user else "assistant-message"

        # Escape external text before inserting it into HTML.
        safe_content = html.escape(str(content)).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="chat-message-container">
                <span class="message-icon">{icon}</span>
                <div class="chat-message {message_class}">
                    <span class="message-text">{safe_content}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def format_sidebar_message(content):
        replacements = {
            "Thought:": "\n:green[**Thought**]:",
            "Action:": "\n:blue[**Action**]:",
            "PAUSE": "\n:orange[**PAUSE**]",
            "Observation:": "\n:violet[**Observation**]:",
            "Final Answer:": "\n:red[**Final Answer**]:",
        }
        for original, formatted in replacements.items():
            content = content.replace(original, formatted)
        return content

    def process_messages(self, messages):
        sidebar_messages = []
        final_answer = None
        normal_assistant_messages = []

        for message in messages:
            role = message.get("role", "")
            content = str(message.get("content") or "")

            # Agent.execute() may return the user's input again.
            # Never show it as an assistant message.
            if role == "user" or not content.strip():
                continue

            if "Final Answer:" in content and role == "assistant":
                final_answer = content.split("Final Answer:", 1)[1].strip()

            if any(
                marker in content
                for marker in ("Thought:", "Action:", "PAUSE", "Observation:", "Final Answer:")
            ):
                sidebar_messages.append(self.format_sidebar_message(content))
            elif role == "assistant":
                normal_assistant_messages.append(content)

        # Show exactly one final reply in the main conversation.
        if final_answer:
            reply = final_answer
        elif normal_assistant_messages:
            reply = normal_assistant_messages[-1]
        else:
            reply = "The agent did not return a final answer. Check Agent Tool Activity."

        return sidebar_messages, [{"role": "assistant", "content": reply}]

    def handle_user_input(self):
        user_input = st.chat_input("How can I help?")
        if not user_input:
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        self.display_message("user", user_input)

        try:
            with st.spinner("Agent is working with tools..."):
                result_messages = self.agent.execute(user_input)

            response_data = [
                {"role": msg.role, "content": msg.content} for msg in result_messages
            ]
            sidebar_messages, main_messages = self.process_messages(response_data)

            st.session_state.agent_logs.extend(sidebar_messages)
            with st.sidebar:
                for log in sidebar_messages:
                    st.markdown(log)

            for message in main_messages:
                st.session_state.chat_history.append(message)
                self.display_message(message["role"], message["content"])
        except Exception as error:
            st.error(f"An error occurred: {type(error).__name__}: {error}")


if __name__ == "__main__":
    WebApp()
