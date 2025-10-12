from __future__ import annotations

"""CLI utilities for the GDG UAM RAG workshop project."""

import argparse
from pathlib import Path
from typing import Iterable

from rag import GDGUAMRAG

DATA_DIR = Path(__file__).parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Herramientas CLI para el RAG del taller GDG UAM."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Inicia un modo interactivo para hacer preguntas.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Número de fragmentos del PDF que se mostrarán como evidencia (por defecto 3).",
    )
    return parser


def render_summary(pipeline: GDGUAMRAG) -> None:
    dataset = pipeline.dataset
    qa_split = dataset["qa"]
    doc_split = dataset["documents"]

    print("GDG UAM RAG dataset listo.")
    print(f"- Pares de preguntas/respuestas: {qa_split.num_rows}")
    print(f"- Fragmentos del PDF: {doc_split.num_rows}")

    sample_qa = qa_split[0]
    print("\nEjemplo de QA:")
    print(f"  Pregunta: {sample_qa['question']}")
    print(f"  Respuesta: {sample_qa['answer']}")

    sample_doc = doc_split[0]
    preview = sample_doc["text"][:180] + ("..." if len(sample_doc["text"]) > 180 else "")
    print("\nEjemplo de fragmento del PDF:")
    print(f"  ID: {sample_doc['id']} (página {sample_doc['page']})")
    print(f"  Texto: {preview}")


def interactive_loop(pipeline: GDGUAMRAG, *, top_k_docs: int) -> None:
    _print_welcome()
    while True:
        try:
            question = input("Pregunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego 👋")
            break

        if not question:
            continue

        if question.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego 👋")
            break

        try:
            answer = pipeline.answer(question, top_k_docs=top_k_docs)
        except ValueError as exc:
            print(f"[Error] {exc}")
            continue

        print("\n" + answer.format() + "\n")


def _print_welcome() -> None:
    print("Modo interactivo RAG - GDG UAM")
    print("Escribe tu pregunta sobre la cafetería y pulsa Enter.")
    print("Comandos especiales: 'salir', 'exit', 'quit'.\n")


def main(argv: Iterable[str] | None = None) -> int:
    """Point of entry for CLI execution."""

    parser = build_parser()
    args = parser.parse_args(args=list(argv) if argv is not None else None)

    pipeline = GDGUAMRAG(data_dir=DATA_DIR)

    if args.interactive:
        interactive_loop(pipeline, top_k_docs=args.top_k)
    else:
        render_summary(pipeline)

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution only
    raise SystemExit(main())
