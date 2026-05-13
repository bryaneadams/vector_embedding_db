from fastmcp import FastMCP

from app.config import settings
from app.prompts import instructions
from app.rag import RAGService


mcp = FastMCP(
    "vector-rag",
    instructions=instructions,
)

rag = RAGService(
    db_kwargs=settings.db_kwargs,
    api_key=settings.gemini_api_key,
    table_name=settings.table_name,
    k=settings.top_k,
    answer_model=settings.answer_model,
    embed_model=settings.embedding_model,
    embedding_dims=settings.embedding_dims,
    instructions=instructions,
)


@mcp.tool
def ask(question: str) -> str:
    """Answer a question using retrieved documents."""
    return rag.answer_question(question)


@mcp.tool
def retrieve(question: str) -> list[tuple[str, int, str]]:
    """Return the top-k retrieved documents for a question."""
    return rag.get_top_k_docs(question)


def main() -> None:
    """Run the FastMCP server."""
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
