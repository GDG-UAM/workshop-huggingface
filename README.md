# workshop-huggingface

RAG assistant para el taller de Hugging Face en GDG UAM. Construiremos el
proyecto paso a paso:

1. **Estructura base** (este commit): carpetas, `pyproject.toml`, script principal.
2. **Ingesta**: parsing de PDFs/JSON en fragmentos normalizados.
3. **Recuperación**: índice TF-IDF + embeddings con cachés.
4. **Generación y app**: modelo generativo, orquestación y UI.

Cada etapa terminará en un commit/push etiquetado para facilitar el seguimiento.
