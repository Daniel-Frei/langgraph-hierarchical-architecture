
HOST:=localhost
PORT:=2024
CONFIG:=langgraph.json


graph-start:
	uv run --env-file .env langgraph dev \
		--host $(HOST) \
		--port $(PORT) \
		--config $(CONFIG) \
		--allow-blocking

graph-install:
	uv sync