import os
from typing import Sequence
from itertools import chain

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from .base import LLMServiceBase


class GROQService(LLMServiceBase):
    def __init__(self):
        self.model = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def invoke(
        self,
        message: str,
        history: Sequence[dict[str, str]],
        context: str | None = None,
    ) -> str:
        if context:
            prelude = [
                {
                    "role": "user",
                    "content": f"""Quiero generar contenido para la cursada de Sistemas Operativos 2025.
                     Usá este documento donde los docentes debatimos sobre los cambios como referencia:
                     
                     '''
                     {context}
                     '''
                     """,
                },
                {
                    "role": "assistant",
                    "content": "¡Perfecto! Te ayudare a generar contenido para la cursada de Sistemas Operativos 2025.",
                },
            ]
        else:
            prelude = []

        user_message = [
            {
                "role": "user",
                "content": message,
            }
        ]

        input_msgs = list(chain(prelude, history, user_message))
        chat_completion = self.model.invoke(input=input_msgs)
        return chat_completion.content

    def get_embeddings(self):
        return OllamaEmbeddings(model="llama3")