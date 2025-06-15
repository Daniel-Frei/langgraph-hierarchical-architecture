# File: src/logger.py

import logging

default_level = logging.DEBUG

# 1) Configure the root logger exactly once:
logging.basicConfig(
    level=default_level,
    format="%(asctime)s %(levelname)s %(message)s"
)


# 1.1) Silence the noise from HTTP/OpenAI/watchfiles at DEBUG:
for noisy in (
    "httpcore",                   # the low-level HTTP client
    "watchfiles.main",           # file-watcher chatter
):
    logging.getLogger(noisy).setLevel(logging.INFO)

#
# 2) Install a Filter that examines each record. If it’s a
#    “Streamed run event” from langgraph_runtime_inmem.ops, then
#    only allow it to pass if it contains `"finish_reason":` (i.e. the
#    final chunk).  Otherwise, suppress.
#
class LastChunkFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()

        # --- 2a) suppress Worker/Queue/Sweeped stats from the in-mem queue: ---
        if record.name == "langgraph_runtime_inmem.queue":
            # drop exactly those three lines
            if msg.startswith("Worker stats") \
            or msg.startswith("Queue stats") \
            or msg.startswith("Sweeped runs"):
                return False

        # Only special‐case the langgraph "Streamed run event" lines:
        if "Streamed run event" in msg and record.name.startswith("langgraph_runtime_inmem.ops"):
            # Final chunk always includes `"finish_reason":` in its JSON payload.
            # (Early/intermediate chunks do not.)
            if '"finish_reason"' in msg:
                return True   # allow the final chunk through
            return False      # drop all intermediate chunks

        # All other messages should be logged normally:
        return True

# 3) Attach that filter to the root logger (so it sees every record).
for handler in logging.root.handlers:
    handler.addFilter(LastChunkFilter())

# 4) Keep getLogger as before:
def getLogger(name: str, level: int = default_level) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
