from langchain.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}


def run_agent_turn(agent, text: str):
    """
    Streamuje kolejne fragmenty treści odpowiedzi asystenta (generator stringów).
    """
    prompt = HumanMessage(text)
    for event in agent.stream(
        {"messages": prompt}, stream_mode="messages", config=config
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
