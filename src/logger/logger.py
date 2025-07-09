# File: src/logger/logger.py

import logging
import os

# --------------------------------------------------------------------------- #
# Resolve default log-level from environment (LOG_LEVEL in .env, shell, etc.) #
# --------------------------------------------------------------------------- #
def _get_default_level() -> int:
    """
    Convert LOG_LEVEL from the environment to a numeric logging constant.

    • Accepts either the usual symbols (“DEBUG”, “INFO”, “WARNING”, …)  
    • …or an explicit integer like “10”.  
    • Falls back to DEBUG if the variable is unset or unrecognised.
    """
    value = os.getenv("LOG_LEVEL", "DEBUG")          # fallback = DEBUG
    try:                                             # numeric?
        return int(value)
    except ValueError:                               # symbolic?
        return getattr(logging, value.upper(), logging.DEBUG)

default_level: int = _get_default_level()

# 1) Configure the root logger exactly once:
logging.basicConfig(
    level=default_level,
    format="%(asctime)s %(levelname)s %(message)s"
)

# 1.1) Silence the noise from HTTP/OpenAI/watchfiles at DEBUG:
for noisy in (
    "httpcore",          # the low-level HTTP client
    "watchfiles.main",   # file-watcher chatter
):
    logging.getLogger(noisy).setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Record-filter: keep only the final chunk of LangGraph streamed-run events   #
# --------------------------------------------------------------------------- #
class LastChunkFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()

        # --- Suppress Worker/Queue/Sweeped stats from the in-mem queue: ---
        if record.name == "langgraph_runtime_inmem.queue" and (
            msg.startswith("Worker stats")
            or msg.startswith("Queue stats")
            or msg.startswith("Sweeped runs")
        ):
            return False

        # --- Drop intermediate “Streamed run event” chunks: ---
        if "Streamed run event" in msg and record.name.startswith(
            "langgraph_runtime_inmem.ops"
        ):
            return '"finish_reason"' in msg  # keep only the final chunk

        # Everything else gets logged normally:
        return True

# Attach the filter to every existing handler:
for handler in logging.root.handlers:
    handler.addFilter(LastChunkFilter())

# --------------------------------------------------------------------------- #
# Convenience wrapper: mirrors logging.getLogger but applies our default lvl #
# --------------------------------------------------------------------------- #
def getLogger(name: str, level: int = default_level) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# Optional helper: pretty-print a list of tool names at DEBUG
def dump_tools(label: str, tools: list):
    names = [
        t.name if hasattr(t, "name") else getattr(t, "__name__", str(t))
        for t in tools
    ]
    getLogger(__name__).debug("%s bound tools: %s", label, names)
