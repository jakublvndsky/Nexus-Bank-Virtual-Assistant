import streamlit as st

from app import build_agent, MODELS, initialize_vector_db
from app.chat_utils import run_agent_turn

MODEL_LABELS = {"openai": "OpenAI GPT-5 Mini", "ollama": "Ollama Llama 3.2 3B"}


def initialize_resources(selected_model_key: str):
    """
    Initialize vector database and agent once per Streamlit session.
    Rebuilds agent when selected model changes.
    """
    if "vector_db_initialized" not in st.session_state:
        initialize_vector_db()
        st.session_state["vector_db_initialized"] = True

    if (
        "agent" not in st.session_state
        or st.session_state.get("model_key_for_agent") != selected_model_key
    ):
        st.session_state["agent"] = build_agent(model=MODELS[selected_model_key])
        st.session_state["model_key_for_agent"] = selected_model_key

    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def main():
    with st.sidebar:
        st.subheader("Model")
        selected_model_key = st.selectbox(
            "Wybierz model",
            options=list(MODELS.keys()),
            format_func=lambda k: MODEL_LABELS[k],
            key="model_choice",
        )
        if selected_model_key == "ollama":
            st.caption("Wymaga uruchomionej Ollama z modelem llama3.2:3b.")

    initialize_resources(selected_model_key)

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
        st.session_state["messages"].append(
            {"role": "assistant", "content": full_answer}
        )


if __name__ == "__main__":
    main()
