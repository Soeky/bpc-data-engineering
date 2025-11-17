#!/usr/bin/env python
"""
generate_clean_text_output.py

Read a BioC JSON file (with a top-level "documents" list) and, for each
document, write a plain-text file containing:

    <title>
    
    <full text / abstract>

Usage:

    uv run generate_clean_text_output.py \
        --input path/to/train.BioC.JSON \
        --output-dir outputs/train_texts
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plain text files (title + text) from BioC JSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the BioC JSON file (e.g. train.BioC.JSON).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory where the .txt files will be written.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text encoding for reading and writing files (default: utf-8).",
    )
    return parser.parse_args()


def load_bioc_json(path: Path, encoding: str = "utf-8") -> Dict[str, Any]:
    with path.open("r", encoding=encoding) as f:
        data = json.load(f)

    # Some BioC files may already be a list of documents; others use {"documents": [...]}
    if isinstance(data, dict) and "documents" in data:
        return data
    elif isinstance(data, list):
        return {"documents": data}
    else:
        raise ValueError(
            f"Unsupported JSON structure in {path}. Expected 'documents' key or a list."
        )


def extract_title_and_body(passages: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Given a list of passages, return a dict with 'title' and 'body'.

    Strategy:
    - Sort passages by "offset"
    - First passage → title
    - Remaining passages → body (joined with blank lines)
    """
    if not passages:
        return {"title": "", "body": ""}

    # Sort passages by offset to ensure correct order
    passages_sorted = sorted(passages, key=lambda p: p.get("offset", 0))

    title = (passages_sorted[0].get("text") or "").strip()
    body_parts = [
        (p.get("text") or "").strip() for p in passages_sorted[1:] if p.get("text")
    ]
    body = "\n\n".join(part for part in body_parts if part)

    return {"title": title, "body": body}


def write_document_text(
    doc_id: str,
    title: str,
    body: str,
    output_dir: Path,
    encoding: str = "utf-8",
) -> None:
    """
    Write a single document to <output_dir>/<doc_id>.txt with:

        <title>

        <body>
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{doc_id}.txt"

    # Compose final text: title + blank line + body (if any body exists)
    if body:
        text = f"{title}\n\n{body}".strip() + "\n"
    else:
        text = (title or "").strip() + "\n"

    with out_path.open("w", encoding=encoding) as f:
        f.write(text)


def main() -> None:
    args = parse_args()

    bioc = load_bioc_json(args.input, encoding=args.encoding)
    documents = bioc.get("documents", [])

    if not isinstance(documents, list):
        raise ValueError("Expected 'documents' to be a list in the BioC JSON file.")

    for doc in documents:
        doc_id = str(doc.get("id", "")).strip()
        if not doc_id:
            # Skip documents without an ID to avoid weird filenames
            continue

        passages = doc.get("passages", [])
        title_body = extract_title_and_body(passages)

        write_document_text(
            doc_id=doc_id,
            title=title_body["title"],
            body=title_body["body"],
            output_dir=args.output_dir,
            encoding=args.encoding,
        )


if __name__ == "__main__":
    main()
