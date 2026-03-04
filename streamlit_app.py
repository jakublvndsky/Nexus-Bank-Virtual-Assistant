import streamlit as st

from app.agent import build_agent
from app.vector_db import initialize_vector_db
from app.chat_utils import run_agent_turn


def initialize_resources():
    """
    Initialize vector database and agent once per Streamlit session.
    """
    if "vector_db_initialized" not in st.session_state:
        initialize_vector_db()
        st.session_state["vector_db_initialized"] = True

    if "agent" not in st.session_state:
        st.session_state["agent"] = build_agent()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def main():
    initialize_resources()

    st.title("Nexus Assistant")
    st.subheader("Virtual Assistant Nexus Bank S.A.")

    # Render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Type your question about Nexus Bank's services...")

    if user_input:
        # Add user message to history and render it immediately
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run agent turn (stream into assistant bubble, then save full answer to history)
        agent = st.session_state["agent"]
        with st.chat_message("assistant"):
            full_answer = st.write_stream(run_agent_turn(agent, user_input))
        st.session_state["messages"].append({"role": "assistant", "content": full_answer})


if __name__ == "__main__":
    main()
