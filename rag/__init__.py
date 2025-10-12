"""RAG components for workshop-huggingface."""

from .ingest import build_gdg_uam_dataset, load_cafeteria_qa, load_pdf_chunks
from .pipeline import GDGUAMRAG, Answer
from .retriever import CorpusEntry, RetrieverHit, TfidfRetriever

__all__ = [
    "Answer",
    "CorpusEntry",
    "GDGUAMRAG",
    "RetrieverHit",
    "TfidfRetriever",
    "build_gdg_uam_dataset",
    "load_cafeteria_qa",
    "load_pdf_chunks",
]
