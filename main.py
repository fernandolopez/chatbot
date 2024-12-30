import argparse
from logging import StreamHandler, getLogger
from pathlib import Path
from sys import argv

from chatbot.routes import *
from chatbot.services.groq import GROQService
from chatbot.services.vector_store import VectorStoreService
from chatbot.services.rag import RAGService

logger = getLogger(__name__)
logger.setLevel("INFO")
logger.addHandler(StreamHandler())
logger.handlers[0].setLevel("INFO")


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-vector-store", "-b", action="store_true", help="Build vector store"
    )
    return parser.parse_args(args)


def main(args):
    if args.build_vector_store:
        context_dir = Path(".") / "context"
        llm = GROQService()
        vector_store = VectorStoreService()
        rag = RAGService(llm_api_service=llm, vector_store=vector_store)
        logger.info(f"Building vector store from files in {context_dir}")
        for document in context_dir.rglob("*.txt"):
            basename = document.name
            logger.info(f"Adding document to vector store: {basename}")
            rag.add_document(document)
            with open("context/in_vector_store", "a") as f:
                f.write(basename + "\n")


if __name__ == "__main__":
    args = parse_args(argv[1:])
    main(args)
