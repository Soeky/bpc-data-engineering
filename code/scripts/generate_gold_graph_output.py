#!/usr/bin/env python
"""
generate_gold_graph_output.py

Read a BioC JSON file and, for each document, export the GOLD
entity graph and relations into a separate JSON file.

For each BioC document with id <doc_id>, the script writes:

    <output_dir>/<doc_id>.json

with structure:

{
  "doc_id": "<string>",
  "title": "<string>",
  "body": "<string>",
  "entities": [
    {
      "id": "<canonical_identifier>",
      "type": "<entity_type>",
      "mentions": [
        {
          "text": "<surface_form>",
          "passage_index": <int>,
          "passage_offset": <int>,
          "char_offset": <int>,
          "length": <int>
        },
        ...
      ]
    },
    ...
  ],
  "relations": [
    {
      "id": "<relation_id>",
      "head_id": "<entity1_identifier>",
      "tail_id": "<entity2_identifier>",
      "type": "<relation_type>",
      "novel": "<Novel|No>"
    },
    ...
  ]
}

Typical usage (with uv):

    uv run generate_gold_graph_output.py \
        --input path/to/train.BioC.JSON \
        --output-dir outputs/train_gold_graphs
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-document gold entity graphs and relations from BioC JSON."
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
        help="Directory where per-document gold graph JSON files will be written.",
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


def extract_title_and_body(passages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Given a list of passages, return (title, body).

    Strategy:
    - Sort passages by "offset"
    - First passage → title
    - Remaining passages → body (joined with blank lines)
    """
    if not passages:
        return "", ""

    passages_sorted = sorted(passages, key=lambda p: p.get("offset", 0))

    title = (passages_sorted[0].get("text") or "").strip()
    body_parts = [
        (p.get("text") or "").strip() for p in passages_sorted[1:] if p.get("text")
    ]
    body = "\n\n".join(part for part in body_parts if part)

    return title, body


def collect_entities(passages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    entities_by_id: Dict[str, Dict[str, Any]] = {}

    for passage_idx, passage in enumerate(passages):
        passage_offset = passage.get("offset", 0)
        annotations = passage.get("annotations", []) or []

        for ann in annotations:
            infons = ann.get("infons", {}) or {}
            identifier = str(infons.get("identifier", "")).strip()
            ent_type = str(infons.get("type", "")).strip()
            text = (ann.get("text") or "").strip()
            locations = ann.get("locations", []) or []

            if not identifier:
                # Skip annotations without a canonical identifier
                continue

            if identifier not in entities_by_id:
                entities_by_id[identifier] = {
                    "id": identifier,
                    "type": ent_type,
                    "mentions": [],
                }

            for loc in locations:
                char_offset = loc.get("offset", 0)
                length = loc.get("length", len(text))

                entities_by_id[identifier]["mentions"].append(
                    {
                        "text": text,
                        "passage_index": passage_idx,
                        "passage_offset": passage_offset,
                        "char_offset": char_offset,
                        "length": length,
                    }
                )

    return entities_by_id


def collect_relations(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    relations_raw = doc.get("relations", []) or []
    relations: List[Dict[str, Any]] = []

    for rel in relations_raw:
        rel_id = str(rel.get("id", "")).strip()
        infons = rel.get("infons", {}) or {}

        entity1 = str(infons.get("entity1", "")).strip()
        entity2 = str(infons.get("entity2", "")).strip()
        rel_type = str(infons.get("type", "")).strip()
        novel_flag = str(infons.get("novel", "")).strip()

        if not entity1 or not entity2 or not rel_type:
            # Skip incomplete relations
            continue

        relations.append(
            {
                "id": rel_id,
                "head_id": entity1,
                "tail_id": entity2,
                "type": rel_type,
                "novel": novel_flag,
            }
        )

    return relations


def main() -> None:
    args = parse_args()
    bioc = load_bioc_json(args.input, encoding=args.encoding)
    documents = bioc.get("documents", [])

    if not isinstance(documents, list):
        raise ValueError("Expected 'documents' to be a list in the BioC JSON file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for doc in documents:
        doc_id = str(doc.get("id", "")).strip()
        if not doc_id:
            # Skip documents without an ID
            continue

        passages = doc.get("passages", []) or []
        title, body = extract_title_and_body(passages)
        entities_by_id = collect_entities(passages)
        relations = collect_relations(doc)

        # Keep only relations whose endpoints appear as entities
        entity_ids = set(entities_by_id.keys())
        relations_filtered = [
            r
            for r in relations
            if (r["head_id"] in entity_ids and r["tail_id"] in entity_ids)
        ]

        record = {
            "doc_id": doc_id,
            "title": title,
            "body": body,
            "entities": list(entities_by_id.values()),
            "relations": relations_filtered,
        }

        out_path = args.output_dir / f"{doc_id}.json"
        with out_path.open("w", encoding=args.encoding) as f:
            json.dump(record, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
