from __future__ import annotations

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from pypdf import PdfReader

__all__ = [
    "build_gdg_uam_dataset",
    "load_cafeteria_qa",
    "load_pdf_chunks",
]


def build_gdg_uam_dataset(
    cafeteria_path: str | Path,
    pdf_path: str | Path,
    *,
    qa_limit: int = 10,
    chunk_size: int = 450,
    chunk_overlap: int = 100,
) -> DatasetDict:
    """Create the canonical GDG UAM dataset combining Q&A pairs and PDF chunks."""

    qa_examples = load_cafeteria_qa(cafeteria_path, limit=qa_limit)
    pdf_chunks = load_pdf_chunks(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Hugging Face datasets expect lists of dicts with consistent keys.
    qa_dataset = Dataset.from_list(qa_examples)
    pdf_dataset = Dataset.from_list(pdf_chunks)

    return DatasetDict(
        {
            "qa": qa_dataset,
            "documents": pdf_dataset,
        }
    )


def load_cafeteria_qa(path: str | Path, *, limit: int | None = None) -> list[dict[str, str]]:
    """Load the cafeteria FAQ data from JSON, returning well-formed HF records."""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of QA records in {json_path!s}")

    examples: list[dict[str, str]] = []
    for idx, entry in enumerate(data, start=1):
        if limit is not None and len(examples) >= limit:
            break

        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()

        if not question or not answer:
            # Skip malformed or empty rows.
            continue

        examples.append(
            {
                "id": f"qa-{idx:03d}",
                "question": question,
                "answer": answer,
                "source": json_path.name,
            }
        )

    if limit is not None and len(examples) < min(limit, len(data)):
        raise ValueError(f"Only {len(examples)} valid QA pairs found, expected at least {limit}")

    return examples


def load_pdf_chunks(
    path: str | Path,
    *,
    chunk_size: int = 450,
    chunk_overlap: int = 100,
) -> list[dict[str, str | int]]:
    """Extract overlapping textual chunks from a PDF suitable for retrieval."""

    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))

    chunks: list[dict[str, str | int]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        clean_text = _normalize_text(raw_text)

        for chunk_index, chunk in enumerate(
            _chunk_text(text=clean_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            start=1,
        ):
            chunks.append(
                {
                    "id": f"pdf-{page_number:03d}-{chunk_index:02d}",
                    "page": page_number,
                    "text": chunk,
                    "source": f"{pdf_path.name}#page={page_number}",
                }
            )

    if not chunks:
        raise ValueError(f"No textual content could be extracted from {pdf_path!s}")

    return chunks


def _normalize_text(text: str) -> str:
    """Collapse excessive whitespace and normalise hyphenation artifacts."""

    # Replace newline followed by hyphenation (e.g., 'pre-\ncios') with the full word.
    text = text.replace("-\n", "")

    # Replace remaining newlines and tabs with spaces, then collapse any double spaces.
    text = " ".join(text.replace("\t", " ").replace("\n", " ").split())
    return text.strip()


def _chunk_text(*, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split long text into overlapping chunks based on word counts."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step

    return chunks
