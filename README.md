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
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Development](#development)
- [License](#license)

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

