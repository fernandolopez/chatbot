from logging import getLogger
from typing import Any, TypedDict
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
# from langchain import hub
from langchain.prompts.chat import ChatPromptTemplate
from .vector_store import VectorStoreService
from .base import LLMServiceBase

logger = getLogger(__name__)

RAG_PROMPT = """Sos un asistente para tareas de pregunta-respuesta.
Usá los siguientes fragmentos de contexto recuperados para responder la pregunta.
Si no sabés la respuesta, simplemente decí que no sabés.
Question: {question} 
Context: {context} 
Answer:"""

# Define state for application
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


class RAGService:
    def __init__(self, llm_api_service: LLMServiceBase, vector_store: VectorStoreService):
        self.llm_api_service = llm_api_service
        self.vector_store = vector_store
        self.embeddings = self.llm_api_service.get_embeddings()
        # self.prompt = hub.pull("rlm/rag-prompt")
        self.prompt = ChatPromptTemplate([("human", RAG_PROMPT)])
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        self.graph = graph_builder.compile()


    def add_document(self, path: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        loader = TextLoader(path)
        docs = loader.load()
        splits = splitter.split_documents(docs)
        self.vector_store.add_documents(splits)

    def _retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def _generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm_api_service.model.invoke(messages)
        return {"answer": response.content}

    def invoke(self, message: dict[str, Any]) -> dict[str, Any] | Any:
        return self.graph.invoke(input=message)
