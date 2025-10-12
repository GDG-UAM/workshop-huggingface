from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from datasets import DatasetDict

from .ingest import build_gdg_uam_dataset
from .retriever import CorpusEntry, RetrieverHit, TfidfRetriever

__all__ = [
    "Answer",
    "GDGUAMRAG",
]


@dataclass(frozen=True)
class Answer:
    """Structured answer produced by the GDG UAM pipeline."""

    question: str
    response: str
    faq_match: RetrieverHit | None
    evidence: Sequence[RetrieverHit]

    def format(self) -> str:
        """Human friendly rendering."""

        lines: list[str] = [self.response]
        if self.faq_match:
            answer = self.faq_match.entry.metadata.get("answer", "")
            lines.append("")
            lines.append("Coincidencia en la FAQ:")
            lines.append(f"- Pregunta: {self.faq_match.entry.metadata.get('question')}")
            lines.append(f"- Respuesta: {answer}")
            lines.append(f"- Score: {self.faq_match.score:.3f}")
        if self.evidence:
            lines.append("")
            lines.append("Fragmentos relevantes del PDF:")
            for hit in self.evidence:
                preview = hit.entry.metadata.get("text", "")
                if isinstance(preview, str) and len(preview) > 240:
                    preview = preview[:240] + "..."
                lines.append(f"- {hit.entry.metadata.get('source')} | score={hit.score:.3f}")
                if preview:
                    lines.append(f"  {preview}")
        return "\n".join(lines)


class GDGUAMRAG:
    """End-to-end helper to query the GDG UAM cafeteria dataset."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        qa_limit: int = 10,
        chunk_size: int = 450,
        chunk_overlap: int = 100,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._qa_limit = qa_limit
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._dataset = self._load_dataset()
        self._qa_retriever = self._build_qa_retriever()
        self._doc_retriever = self._build_doc_retriever()

    @property
    def dataset(self) -> DatasetDict:
        return self._dataset

    def answer(self, question: str, *, top_k_docs: int = 3) -> Answer:
        """Retrieve FAQ and document evidence to answer `question`."""

        question = question.strip()
        if not question:
            raise ValueError("La pregunta no puede estar vacía.")

        faq_hits = self._qa_retriever.search(question, top_k=1)
        faq_match = faq_hits[0] if faq_hits else None

        doc_hits = self._doc_retriever.search(question, top_k=top_k_docs)

        response = self._build_response(question, faq_match, doc_hits)
        return Answer(question=question, response=response, faq_match=faq_match, evidence=doc_hits)

    def _load_dataset(self) -> DatasetDict:
        return build_gdg_uam_dataset(
            self._data_dir / "cafeteria.json",
            self._data_dir / "Precios_Cafeterías_2025_signed.pdf",
            qa_limit=self._qa_limit,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def _build_qa_retriever(self) -> TfidfRetriever:
        qa_split = self._dataset["qa"]
        entries: list[CorpusEntry] = []
        for item in qa_split:
            question = item.get("question", "")
            answer = item.get("answer", "")
            entry = CorpusEntry(
                id=item.get("id", ""),
                content=f"{question} {answer}",
                metadata={
                    "question": question,
                    "answer": answer,
                    "source": item.get("source", ""),
                },
            )
            entries.append(entry)

        return TfidfRetriever(entries, min_df=1, max_df=1.0, ngram_range=(1, 2))

    def _build_doc_retriever(self) -> TfidfRetriever:
        doc_split = self._dataset["documents"]
        entries: list[CorpusEntry] = []
        for item in doc_split:
            text = item.get("text", "")
            entry = CorpusEntry(
                id=item.get("id", ""),
                content=text,
                metadata={
                    "text": text,
                    "source": item.get("source", ""),
                    "page": item.get("page", ""),
                },
            )
            entries.append(entry)

        return TfidfRetriever(entries, min_df=1, max_df=1.0, ngram_range=(1, 2))

    def _build_response(
        self,
        question: str,
        faq_match: RetrieverHit | None,
        doc_hits: Sequence[RetrieverHit],
    ) -> str:
        if faq_match and faq_match.score >= 0.25:
            return str(faq_match.entry.metadata.get("answer"))

        if not doc_hits:
            return (
                "No encontré información relevante en la FAQ ni en el PDF. "
                "Prueba reformulando la pregunta."
            )

        best = doc_hits[0]
        context = best.entry.metadata.get("text", "")
        source = best.entry.metadata.get("source", "desconocido")
        return (
            "No encontré una respuesta exacta en la FAQ, pero este fragmento del PDF puede ayudarte:\n"
            f"- Fuente: {source}\n"
            f"- Contenido: {context}"
        )
