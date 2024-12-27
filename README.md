# chatbot

## Prerequisites

* make
* poetry

## Setup

```sh
make setup
```

or

```sh
poetry install
```

## Run

```sh
make
```

or

```sh
poetry run mesop main.py
```

## Configuration

1. Provide GROQ API key in .env file as shown in .env-example.
2. If you want to include a context to be sent in each message, write it in context.txt.
3. If history is too long and LLM reaches token limit remove checkpoints.jsonl -or remove some lines-