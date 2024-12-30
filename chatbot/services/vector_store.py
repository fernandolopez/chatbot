from langchain_community.vectorstores import SQLiteVec
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.schema import Document


class VectorStoreService:
    def __init__(self):
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        connection = SQLiteVec.create_connection(db_file="vector_store.db")
        self.vector_store = SQLiteVec(
            table="state_union", connection=connection, embedding=embedding_function
        )

    def add_texts(self, texts: list[str]) -> list[str]:
        self.vector_store.add_texts(texts)

    def add_documents(self, docs: list[Document]) -> list[str]:
        self.vector_store.add_documents(docs)

    def similarity_search(self, text: str, k: int = 5) -> list[Document]:
        return self.vector_store.similarity_search(text, k=k)
