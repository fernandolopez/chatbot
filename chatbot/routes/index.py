import json
import os
import mesop as me
import mesop.labs as mel
from mesop.labs.chat import State as ChatState
from typing import Generator
from dataclasses import asdict
from chatbot.services.groq import GROQService
from chatbot.services.rag import RAGService
from logging import getLogger

from chatbot.services.vector_store import VectorStoreService

logger = getLogger(__name__)

llm = GROQService()

if os.path.exists("context.txt"):
    with open("context.txt", "r") as f:
        context = f.read()
else:
    context = None


def transform_echo(
    msg: str, history: list[mel.ChatMessage]
) -> Generator[str, None, None]:
    yield msg


def transform_llm(
    msg: str, history: list[mel.ChatMessage]
) -> Generator[str, None, None]:
    history = [asdict(h) for h in history]
    response = llm.invoke(msg, history=history, context=context)
    with open("checkpoints.jsonl", "a") as f:
        f.write(
            json.dumps(
                [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": response},
                ]
            )
            + "\n"
        )
    yield response


def make(rag: RAGService):
    def transform_rag(
        msg: str, history: list[mel.ChatMessage]
    ) -> Generator[str, None, None]:
        response = rag.invoke({"question": msg})
        yield response["answer"]

    return transform_rag


@me.page(path="/")
def index():
    chat_state = me.state(ChatState)
    me.set_page_title("Simple chat")
    if not chat_state.output and os.path.exists("checkpoints.jsonl"):
        logger.info("Loading chat history from checkpoints.jsonl")
        with open("checkpoints.jsonl", "r") as f:
            for line in f:
                for message in json.loads(line):
                    chat_state.output.append(mel.ChatMessage(**message))
    mel.chat(transform=transform_llm, title="Simple Bot", bot_user="simple")


@me.page(path="/rag")
def rag():
    me.set_page_title("RAG Chat")
    vector_store = VectorStoreService()
    rag = RAGService(llm_api_service=llm, vector_store=vector_store)
    mel.chat(transform=make(rag), title="RAG Bot", bot_user="rag")
