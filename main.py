import uuid

from app.agent import build_agent
from app.chat_utils import run_agent_turn
from app.langfuse_config import flush
from app.vector_db import initialize_vector_db

initialize_vector_db()
thread_id = str(uuid.uuid4())


def chat_agent():
    agent = build_agent()
    print("=== Nexus Assistant ===")
    print("Hello I am Nexus assistant how can I help you today?")
    while True:
        try:
            text = input("")
        except KeyboardInterrupt:
            break

        text = text.strip()
        if not text:
            continue
        if text.lower() in {"exit", "q", "quit"}:
            break

        for answer in run_agent_turn(agent, text, thread_id):
            print(answer, end="", flush=True)
        print()


if __name__ == "__main__":
    try:
        chat_agent()
    finally:
        flush()
