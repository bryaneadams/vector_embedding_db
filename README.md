# RAG to MCP

This repo has two tracks:

- notebooks for learning chunking, embeddings, and prompt strategies
- an `app/` directory for a real MCP RAG tool

## Directory Structure

`app/` contains the MCP RAG app code.
`data/` holds the example source documents used for loading and testing.
`database/` contains the Postgres setup, env files, and init SQL.
`setup.ipynb` is the notebook that loads the example documents into Postgres and walks you through a basic RAG architecture for Q&A.

```text
.
├── app
│   ├── __init__.py
│   ├── config.py
│   ├── example.env
│   ├── mcp_server.py
│   ├── prompts.py
│   └── rag.py
├── data
│   └── raw
│       ├── fm
│       └── news
├── database
│   ├── docker-compose.yml
│   ├── example.env
│   └── initdb
│       └── 01-vector.sql
├── setup.ipynb
├── pyproject.toml
└── README.md
```

## Prerequisites

- Docker Desktop running
- Python 3.11+ installed
- Git installed
- a Gemini API key

## First-Time Setup

1. Clone the repo.

2. Create and activate a virtual environment from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install the project:

```bash
pip install -e .
```

4. Copy the example env files into place:

```bash
cp database/example.env database/.env
cp app/example.env app/.env
```
5. Put your Gemini API key into into `.api_key` (this is for `setup.ipynb`)

6. Put your Gemini API key into `app/.env` (this is for the MCP server)

7. Start Postgres:

```bash
cd database
docker compose up -d
```

7. Run `setup.ipynb` to load the example documents from `data/` into Postgres.

## Notebook Workflow

Use `setup.ipynb` and the other notebooks to experiment with:

- chunking strategies
- embedding settings
- retrieval tuning
- prompt strategies

This is the learning/demo side of the repo.

## MCP App Workflow

The `app/` directory is the reusable tool path:

- `app/config.py` loads local settings
- `app/rag.py` handles retrieval and generation
- `app/mcp_server.py` exposes the RAG flow through FastMCP

Run the MCP server from the repo root:

```bash
python -m app.mcp_server
```

## Codex Config

Add this to `~/.codex/config.toml`:

```toml
[mcp_servers.vector_rag]
url = "http://127.0.0.1:8000/mcp"
```

Then start the server and let Codex connect to it.

## Notes

- The database uses pgvector in Postgres.
- The first database startup creates the extension.
- If you want a clean re-run, remove the Docker volume and start again.
