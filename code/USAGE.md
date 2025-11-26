# Usage Guide

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up environment variables:**
   Create a `.env` file in the `code/` directory with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## Configuration

The configuration is managed in `config.py`. Key settings:

- **Models**: Available models are defined in `Config.AVAILABLE_MODELS`
- **RAG Settings**: 
  - Source directory: `Config.RAG_SOURCE_DIR` (default: `rag_sources/`)
  - Embeddings directory: `Config.RAG_EMBEDDINGS_DIR` (default: `rag_embeddings/`)
- **API Settings**: OpenRouter base URL and API key

## Running the Pipeline

### Basic Usage

```python
from main import main

# Run with default settings (all techniques, default model)
main(split="test")
```

### Custom Model Configuration

```python
# Test different models for each technique
main(
    split="test",
    models={
        "IO": "gpt-4o-mini",
        "CoT": "gpt-4o",
        "RAG": "claude-3-5-sonnet",
        "ReAct": "gpt-4-turbo"
    }
)
```

### Run Specific Techniques

```python
# Only run IO and CoT
main(
    split="dev",
    techniques=["IO", "CoT"],
    models={"IO": "gpt-4o-mini", "CoT": "gpt-4o-mini"}
)
```

## RAG Setup

### Adding Source Files

1. Place your source files (`.txt`, `.md`, or `.json`) in the `rag_sources/` directory
2. The vector store will automatically:
   - Compute hashes for each file
   - Generate embeddings for new/changed files
   - Cache embeddings in `rag_embeddings/` directory
   - Reuse cached embeddings for unchanged files

### Example

```python
# Files in rag_sources/ will be automatically indexed
# When you run RAG prompter, it will use these files for retrieval

from pipeline.retrieval import VectorStore
from config import Config

# Manually add documents (optional)
vector_store = VectorStore()
vector_store.add_documents_from_files(Config.RAG_SOURCE_DIR)

# Or add documents programmatically
vector_store.add_documents([
    {"text": "Your document text here", "metadata": "optional"}
])
```

## Available Models

Models available through OpenRouter (configured in `config.py`):

- `gpt-4o-mini` - OpenAI GPT-4o Mini
- `gpt-4o` - OpenAI GPT-4o
- `gpt-4-turbo` - OpenAI GPT-4 Turbo
- `claude-3-5-sonnet` - Anthropic Claude 3.5 Sonnet
- `claude-3-opus` - Anthropic Claude 3 Opus
- `llama-3.1-70b` - Meta Llama 3.1 70B
- `gemini-pro` - Google Gemini Pro 1.5

## Prompting Techniques

1. **IO (Input/Output)**: Simple zero-shot prompting
2. **CoT (Chain of Thought)**: Step-by-step reasoning
3. **RAG (Retrieval-Augmented Generation)**: Uses external knowledge from `rag_sources/`
4. **ReAct (Reason + Act)**: Reasoning with action steps

## Embedding Caching

The RAG system uses file hashing to cache embeddings:

- **Hash Index**: Stored in `rag_embeddings/hash_index.json`
- **Embeddings**: Stored in `rag_embeddings/embeddings.npy`
- **Documents**: Stored in `rag_embeddings/documents.json`

When a file changes (hash differs), new embeddings are generated. Unchanged files reuse cached embeddings, saving API calls and time.

