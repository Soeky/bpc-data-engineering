#!/usr/bin/env python
"""
graph_viewer_server.py

Interactive graph viewer for gold relation graphs using Cytoscape.js.

- You pass a directory containing per-document gold graph JSON files
  (one JSON per document; structure like:

    {
      "doc_id": "27993978",
      "title": "...",
      "body": "...",
      "entities": [
        { "id": "26886",
          "type": "GeneOrGeneProduct",
          "mentions": [ { "text": "CenpH", ... }, ... ]
        },
        ...
      ],
      "relations": [
        {
          "id": "R0",
          "head_id": "26886",
          "tail_id": "268697",
          "type": "Association",
          "novel": "Novel"
        },
        ...
      ]
    }

- The server:
    * scans the directory for *.json files
    * builds a per-document graph (nodes + edges)
    * builds a combined FULL graph for the entire directory
    * serves a split-screen UI where you can:
        - choose any document or the FULL graph on the left
        - choose any document or the FULL graph on the right
        - interactively zoom/pan/drag nodes (Cytoscape.js)

Usage (with uv):

    uv run graph_viewer_server.py --dir outputs/train_gold_graphs

Then open your browser at http://127.0.0.1:5000/

Requirements:
    pip install flask
(Everything else is handled by the browser via Cytoscape.js CDN.)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, abort, jsonify, render_template_string, request


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an interactive webserver to inspect gold relation graphs (per doc and full)."
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        required=True,
        help="Directory containing per-document gold graph JSON files.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the webserver (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind the webserver (default: 5000).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding used to read JSON files (default: utf-8).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=300,
        help="Maximum number of nodes to return per graph (for performance).",
    )
    return parser.parse_args()


# ---------- Data loading & transformation ----------

def load_doc_json(path: Path, encoding: str = "utf-8") -> Dict[str, Any]:
    with path.open("r", encoding=encoding) as f:
        return json.load(f)


def doc_to_elements(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single gold doc JSON to Cytoscape.js elements:
    - nodes with data: { id, label, type, num_mentions }
    - edges with data: { id, source, target, type, novel }
    """
    elements: List[Dict[str, Any]] = []

    entities = doc.get("entities", []) or []
    relations = doc.get("relations", []) or []

    # Nodes
    for ent in entities:
        ent_id = str(ent.get("id", "")).strip()
        if not ent_id:
            continue
        ent_type = ent.get("type", "")
        mentions = ent.get("mentions") or []
        label = ent_id
        if mentions:
            # Use first mention text as label
            text = mentions[0].get("text")
            if text:
                label = text
        num_mentions = len(mentions)

        elements.append(
            {
                "data": {
                    "id": ent_id,
                    "label": label,
                    "type": ent_type,
                    "num_mentions": num_mentions,
                }
            }
        )

    entity_ids = {e["data"]["id"] for e in elements}

    # Edges
    for rel in relations:
        rel_id = str(rel.get("id", "")).strip() or None
        head = str(rel.get("head_id", "")).strip()
        tail = str(rel.get("tail_id", "")).strip()
        if head not in entity_ids or tail not in entity_ids:
            continue
        rel_type = rel.get("type", "")
        novel = rel.get("novel", "")

        edge_id = rel_id or f"{head}-{rel_type}->{tail}"

        elements.append(
            {
                "data": {
                    "id": edge_id,
                    "source": head,
                    "target": tail,
                    "type": rel_type,
                    "novel": novel,
                }
            }
        )

    return elements


def build_full_graph_elements(
    docs: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge all documents into a single global graph (nodes merged by id).
    Returns Cytoscape.js elements (nodes + edges).
    """
    # Merge nodes by id
    node_map: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    for doc in docs.values():
        ents = doc.get("entities", []) or []
        rels = doc.get("relations", []) or []

        # entities
        for ent in ents:
            ent_id = str(ent.get("id", "")).strip()
            if not ent_id:
                continue
            ent_type = ent.get("type", "")
            mentions = ent.get("mentions") or []

            if ent_id not in node_map:
                label = ent_id
                if mentions:
                    text = mentions[0].get("text")
                    if text:
                        label = text
                node_map[ent_id] = {
                    "id": ent_id,
                    "label": label,
                    "type": ent_type,
                    "num_mentions": len(mentions),
                }
            else:
                # accumulate number of mentions
                node_map[ent_id]["num_mentions"] += len(mentions)

        # relations
        for rel in rels:
            head = str(rel.get("head_id", "")).strip()
            tail = str(rel.get("tail_id", "")).strip()
            if head not in node_map or tail not in node_map:
                continue
            rel_type = rel.get("type", "")
            novel = rel.get("novel", "")
            rel_id = str(rel.get("id", "")).strip() or f"{head}-{rel_type}->{tail}"

            edges.append(
                {
                    "data": {
                        "id": rel_id,
                        "source": head,
                        "target": tail,
                        "type": rel_type,
                        "novel": novel,
                    }
                }
            )

    nodes = [{"data": v} for v in node_map.values()]
    return nodes + edges


# ---------- Flask app ----------

def create_app(
    json_dir: Path,
    encoding: str = "utf-8",
    max_nodes: int = 300,
) -> Flask:
    app = Flask(__name__)

    # Load all docs in the directory
    json_files = sorted(p for p in json_dir.glob("*.json") if p.is_file())
    if not json_files:
        raise RuntimeError(f"No .json files found in {json_dir}")

    docs_raw: Dict[str, Dict[str, Any]] = {}
    doc_titles: Dict[str, str] = {}

    for path in json_files:
        doc = load_doc_json(path, encoding=encoding)
        doc_id = str(doc.get("doc_id", "") or doc.get("id", "") or path.stem).strip()
        docs_raw[doc_id] = doc
        doc_titles[doc_id] = doc.get("title") or ""

    # Precompute per-doc elements
    doc_elements: Dict[str, List[Dict[str, Any]]] = {
        doc_id: doc_to_elements(doc) for doc_id, doc in docs_raw.items()
    }

    # Precompute full graph
    FULL_ID = "__FULL_GRAPH__"
    FULL_LABEL = "FULL GRAPH (all docs in dir)"
    full_elements = build_full_graph_elements(docs_raw)

    # Build doc options for UI
    doc_options: List[Tuple[str, str]] = [(FULL_ID, FULL_LABEL)]
    for doc_id in sorted(docs_raw.keys()):
        title = doc_titles.get(doc_id, "")
        label = f"{doc_id} - {title[:80]}" if title else doc_id
        doc_options.append((doc_id, label))

    TEMPLATE = """
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Interactive Graph Viewer</title>
        <style>
            body {
                font-family: sans-serif;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }
            .top-bar {
                padding: 0.5rem 1rem;
                background: #f0f0f0;
                border-bottom: 1px solid #ddd;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .split-container {
                display: flex;
                height: calc(100vh - 50px);
            }
            .panel {
                flex: 1;
                border-right: 1px solid #ddd;
                display: flex;
                flex-direction: column;
            }
            .panel:last-child {
                border-right: none;
            }
            .panel-header {
                padding: 0.5rem;
                border-bottom: 1px solid #eee;
                background: #fafafa;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .panel-body {
                flex: 1;
                position: relative;
            }
            #left-graph, #right-graph {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
            }
            select {
                max-width: 90%;
            }
            .doc-title {
                font-size: 0.85rem;
                color: #555;
                margin-left: 0.5rem;
                flex: 1;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
        </style>
        <!-- Cytoscape.js from CDN -->
        <script src="https://unpkg.com/cytoscape@3.28.0/dist/cytoscape.min.js"></script>
    </head>
    <body>
        <div class="top-bar">
            <div><strong>Interactive Graph Viewer</strong> — Directory: {{ json_dir }}</div>
            <div style="font-size:0.85rem;color:#555;">Use mouse wheel to zoom, drag background to pan, drag nodes to rearrange.</div>
        </div>

        <div class="split-container">
            <div class="panel">
                <div class="panel-header">
                    <label for="left-select">Left:</label>
                    <select id="left-select">
                        {% for value, label in doc_options %}
                            <option value="{{ value }}">{{ label }}</option>
                        {% endfor %}
                    </select>
                    <span class="doc-title" id="left-title"></span>
                </div>
                <div class="panel-body">
                    <div id="left-graph"></div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <label for="right-select">Right:</label>
                    <select id="right-select">
                        {% for value, label in doc_options %}
                            <option value="{{ value }}">{{ label }}</option>
                        {% endfor %}
                    </select>
                    <span class="doc-title" id="right-title"></span>
                </div>
                <div class="panel-body">
                    <div id="right-graph"></div>
                </div>
            </div>
        </div>

        <script>
            const FULL_ID = "{{ full_id }}";
            const docTitles = {{ doc_titles | tojson }};

            function initCytoscape(containerId) {
                return cytoscape({
                    container: document.getElementById(containerId),
                    elements: [],
                    style: [
                        {
                            selector: 'node',
                            style: {
                                'background-color': '#888',
                                'label': 'data(label)',
                                'font-size': 8,
                                'text-valign': 'center',
                                'color': '#000',
                                'text-outline-width': 1,
                                'text-outline-color': '#ffffff'
                            }
                        },
                        {
                            selector: 'node[type = "GeneOrGeneProduct"]',
                            style: {
                                'background-color': '#4caf50'
                            }
                        },
                        {
                            selector: 'node[type = "DiseaseOrPhenotypicFeature"]',
                            style: {
                                'background-color': '#f44336'
                            }
                        },
                        {
                            selector: 'node[type = "ChemicalEntity"]',
                            style: {
                                'background-color': '#2196f3'
                            }
                        },
                        {
                            selector: 'edge',
                            style: {
                                'width': 1,
                                'line-color': '#999',
                                'target-arrow-color': '#999',
                                'target-arrow-shape': 'triangle',
                                'curve-style': 'bezier',
                                'label': 'data(type)',
                                'font-size': 7,
                                'text-rotation': 'autorotate'
                            }
                        }
                    ],
                    layout: {
                        name: 'cose',
                        animate: false
                    }
                });
            }

            async function loadGraph(cy, docId, titleElem) {
                titleElem.textContent = '';
                try {
                    const resp = await fetch(`/graph_data?doc_id=${encodeURIComponent(docId)}`);
                    if (!resp.ok) {
                        throw new Error('HTTP ' + resp.status);
                    }
                    const data = await resp.json();
                    const elements = data.elements || [];
                    const title = data.title || '';
                    titleElem.textContent = title;

                    cy.elements().remove();
                    cy.add(elements);

                    if (elements.length === 0) {
                        cy.layout({ name: 'grid' }).run();
                    } else {
                        cy.layout({ name: 'cose', animate: false }).run();
                        cy.fit();
                    }
                } catch (err) {
                    console.error('Failed to load graph:', err);
                    titleElem.textContent = 'Error loading graph';
                    cy.elements().remove();
                    cy.layout({ name: 'grid' }).run();
                }
            }

            document.addEventListener('DOMContentLoaded', () => {
                const leftSelect = document.getElementById('left-select');
                const rightSelect = document.getElementById('right-select');
                const leftTitle = document.getElementById('left-title');
                const rightTitle = document.getElementById('right-title');

                const cyLeft = initCytoscape('left-graph');
                const cyRight = initCytoscape('right-graph');

                // default: both show FULL graph
                leftSelect.value = "{{ full_id }}";
                rightSelect.value = "{{ full_id }}";

                loadGraph(cyLeft, leftSelect.value, leftTitle);
                loadGraph(cyRight, rightSelect.value, rightTitle);

                leftSelect.addEventListener('change', () => {
                    loadGraph(cyLeft, leftSelect.value, leftTitle);
                });

                rightSelect.addEventListener('change', () => {
                    loadGraph(cyRight, rightSelect.value, rightTitle);
                });
            });
        </script>
    </body>
    </html>
    """

    @app.get("/")
    def index():
        return render_template_string(
            TEMPLATE,
            json_dir=str(json_dir),
            doc_options=doc_options,
            full_id=FULL_ID,
            doc_titles=doc_titles,
        )

    @app.get("/graph_data")
    def graph_data():
        doc_id = request.args.get("doc_id")
        if doc_id is None:
            abort(400, "Missing doc_id parameter")

        # Decide which graph to return
        if doc_id == FULL_ID:
            elements = full_elements
            title = "Full graph (all documents)"
        else:
            if doc_id not in docs_raw:
                abort(404, f"Unknown doc_id: {doc_id}")
            elements = doc_elements.get(doc_id, [])
            title = doc_titles.get(doc_id, "")

        # Apply max_nodes limit (nodes + edges)
        if len(elements) > max_nodes:
            # naive truncation; if you want smarter sampling, we could refine this
            elements = elements[:max_nodes]

        return jsonify({"doc_id": doc_id, "title": title, "elements": elements})

    return app


# ---------- main ----------

def main() -> None:
    args = parse_args()
    json_dir: Path = args.dir
    if not json_dir.is_dir():
        raise RuntimeError(f"{json_dir} is not a directory")

    app = create_app(
        json_dir=json_dir,
        encoding=args.encoding,
        max_nodes=args.max_nodes,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
