# car-color-graph

**Version:** 0.1.0
**Description:** LangGraph supervisor/subgraph demo – with shared and private states and tools

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)

---

## Introduction

This is a minimal demo of a LangGraph-based workflow that uses a **supervisor** graph coordinating two **subgraph** “agents”:

1. **color_agent** – collects a single-word car colour from the user.
2. **speed_agent** – collects a speed adjective for the car.

The **supervisor** orchestrates delegation to each specialist in turn and then assembles the final sentence:
> *“The car is {color} and {speed}.”*

---

## Features

- **Stateful multi-agent orchestration** using LangGraph’s `StateGraph`.
- **Tools** for interacting with the user, getting and editing the state (`ask_user`, `set_state`, `get_state`).
- **State** - using shared and private states
- **Supervisor/subgraph pattern** via `langgraph-supervisor`.
- **In-memory execution** powered by `langgraph-cli[inmem]` for easy local experimentation.

---

## Prerequisites

- **Python 3.11+**
- An **OpenAI API key** (for `langchain-openai`)
- **uv** and **make**

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://your.repo.url/car-color-graph.git
   cd car-color-graph
   ```

2. **Create & activate a virtual environment**
   ```bash
   uv venv
   ```

3. **Install dependencies**
   ```bash
   make graph-install
   ```

## Configuration
Create a .env file in the project root with your OpenAI API credentials:

   ```bash
   OPENAI_API_KEY=sk-...
   ```


## Usage
To run the demo:

**Run**
   ```bash
   make graph-start
   ```

## 📚 Tool API

> The project ships three reusable, schema-aware LangGraph tools.  
> All three are *factories*: they return a concrete tool function that you register with `bind_tools` or a `ToolNode`.

| Factory            | Purpose                                               | Typical call                                                            |
| ------------------ | ----------------------------------------------------- | ----------------------------------------------------------------------- |
| `make_ask_user()`  | Pause the agent and ask the human a question.         | `ask_user = make_ask_user("messagesSpeed")`                             |
| `make_get_state()` | Read a value from the current graph state.            | `get_state = make_get_state(state_schema=SharedState)`                  |
| `make_set_state()` | Write / overwrite a value in the current graph state. | `set_state = make_set_state("messagesSpeed", state_schema=SharedState)` |

---

### `make_ask_user` — interactive prompt

```python
ask_speed = make_ask_user(
    msg_key="messagesSpeed",   # optional – defaults to "messages"
    name="ask_user_speed",     # optional – defaults to "ask_user"
)
```

| Argument  | Type           | Default      | Description                                                                                     |
| --------- | -------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| `msg_key` | `str \| None`  | `"messages"` | Thread list in the state that will receive the assistant prompt & the resulting `ToolMessage`. |
| `name`    | `str \| None`  | `"ask_user"` | Public tool name exposed to the LLM.                                                            |

**Runtime signature**

```python
(prompt: str) -> str     # raw user reply
```

* If *prompt* is **not** a string the tool **does not crash**; it returns a `ToolMessage` with an `"ERROR: prompt argument must be a string."` payload so the LLM can retry.

---

### `make_get_state` — read-only access

```python
get_state = make_get_state(
    state_schema=SharedState,  # REQUIRED
    name="get_state_speed",    # optional
)
```

| Argument       | Type                          | Default       | Description                                                                        |
| -------------- | ----------------------------- | ------------- | ---------------------------------------------------------------------------------- |
| `state_schema` | `pydantic.BaseModel subclass` | **required**  | Schema the state must comply with. Omitting it raises a *build-time* `ValueError`. |
| `name`         | `str \| None`                 | `"get_state"` | Public tool name.                                                                  |

**Runtime signature**

```python
(key: str) -> str          # stored value or ""
```

* Non-string *key* → returns `"ERROR: ‘key’ argument must be a string."` (tool never crashes).

---

### `make_set_state` — destructive write

```python
set_state = make_set_state(
    msg_key="messagesSpeed",   # optional – defaults to "messages"
    state_schema=SharedState,  # REQUIRED
    name="set_state_speed",    # optional
)
```

| Argument       | Type                          | Default       | Description                                                       |
| -------------- | ----------------------------- | ------------- | ----------------------------------------------------------------- |
| `msg_key`      | `str \| None`                 | `"messages"`  | Thread that receives confirmation / error `ToolMessage`s.         |
| `name`         | `str \| None`                 | `"set_state"` | Public tool name.                                                 |
| `state_schema` | `pydantic.BaseModel subclass` | **required**  | Used to validate the `(key, value)` pair. Build fails without it. |

**Runtime signature**

```python
(key: str, value: str) -> None     # returns a Command update
```

Runtime safeguards:

| Condition                   | Tool response (no crash)                                            |
| --------------------------- | ------------------------------------------------------------------- |
| `key` is not `str`          | `ToolMessage → "ERROR: ‘key’ must be a string."`                    |
| `value` is not `str`        | `ToolMessage → "ERROR: ‘value’ must be a string."`                  |
| `key` not in `state_schema` | `ToolMessage → "ERROR: '<key>' is not a valid field in the state."` |
| Schema validation fails     | `ToolMessage → "ERROR: <pydantic message>"`                         |

On success the state is updated and the thread receives `"color updated."`, `"speed updated."`, etc.

---

### Quick example

```python
from state.main_state import SharedState
from tools import make_ask_user, make_get_state, make_set_state

ask_color   = make_ask_user("messagesColor")                  # uses default name "ask_user"
get_state   = make_get_state(state_schema=SharedState)        # uses default name "get_state"
set_state   = make_set_state("messagesColor",
                             state_schema=SharedState)        # uses default name "set_state"

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(
    [ask_color, get_state, set_state]
)
```

Add these tools to a `ToolNode` or `bind_tools`, connect them in your LangGraph, and you’re ready to roll.