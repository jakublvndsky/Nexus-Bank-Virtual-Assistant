"""
Langfuse observability (optional).
Import and use only after environment variables are loaded (e.g. after load_dotenv()).
See: https://langfuse.com/docs/integrations/langchain
"""
import os


def _is_langfuse_configured() -> bool:
    return bool(os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse_callbacks_and_metadata(session_id: str | None = None):
    """
    Returns (callbacks_list, metadata_dict) for LangChain/LangGraph config.
    Call only after load_dotenv() has run. Returns ([], {}) when Langfuse is not configured.
    """
    if not _is_langfuse_configured():
        return [], {}

    from langfuse.langchain import CallbackHandler

    callbacks = [CallbackHandler()]
    metadata = {"langfuse_tags": ["nexus-assistant"]}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    return callbacks, metadata


def flush():
    """
    Flush Langfuse client so traces are sent before process exit.
    Call in scripts (e.g. CLI) before exit to avoid losing traces.
    """
    if not _is_langfuse_configured():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass
