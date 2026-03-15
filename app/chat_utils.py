from langchain.messages import HumanMessage

from app.langfuse_config import get_langfuse_callbacks_and_metadata


def run_agent_turn(agent, text: str, thread_id: str):
    """
    Streams successive chunks of the assistant response (string generator).
    """
    prompt = HumanMessage(text)
    callbacks, langfuse_metadata = get_langfuse_callbacks_and_metadata(session_id=thread_id)
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "metadata": langfuse_metadata,
        "run_name": "nexus-chat",
    }
    for event in agent.stream(
        {"messages": prompt},
        stream_mode="messages",
        config=config,
    ):
        chunk = event[0] if isinstance(event, (list, tuple)) else event
        t = getattr(chunk, "type", type(chunk).__name__)
        if not t:
            continue
        t = str(t).lower()
        if t == "human":
            continue
        if "ai" not in t and "assistant" not in t and "chat" not in t:
            continue
        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content.strip():
            yield content
