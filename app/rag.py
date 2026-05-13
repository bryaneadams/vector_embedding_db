import numpy as np
import psycopg
from psycopg import sql
from google import genai
from google.genai import types
from pgvector.psycopg import register_vector


class RAGService:
    def __init__(
        self,
        db_kwargs: dict,
        api_key: str,
        table_name: str = "documents",
        k: int = 3,
        answer_model: str = "gemini-2.5-flash",
        embed_model: str = "gemini-embedding-001",
        embedding_dims: int = 1536,
        instructions: str = "",
    ) -> None:
        """Initialize the RAG service.

        Args:
            db_kwargs (dict[str, str]): Connection keyword arguments for Postgres.
            api_key (str): Gemini API key.
            table_name (str, optional): Table containing documents. Defaults to "documents".
            k (int, optional): Number of documents to retrieve. Defaults to 3.
            answer_model (str, optional): Gemini model used for answer generation.
            embed_model (str, optional): Gemini model used for embeddings.
            embedding_dims (int, optional): Embedding dimensionality.
            instructions (str, optional): Preamble injected into the answer prompt.
        """
        self.db_kwargs = db_kwargs
        self.table_name = table_name
        self.k = k
        self.answer_model = answer_model
        self.embed_model = embed_model
        self.instructions = instructions
        self.embedding_dims = embedding_dims
        self.client = genai.Client(api_key=api_key)

    def _normalize(self, vec: list[float]) -> list[float]:
        """L2-normalize an embedding vector.

        Args:
            vec (list[float]): Input vector.

        Returns:
            list[float]: Normalized vector, or the original vector if its norm is zero.
        """
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        return (arr / norm).tolist() if norm else arr.tolist()

    def embed_query(self, question: str) -> list[float]:
        """Embed a question for retrieval.

        Args:
            question (str): User question.

        Returns:
            list[float]: Normalized query embedding.
        """
        resp = self.client.models.embed_content(
            model=self.embed_model,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="QUESTION_ANSWERING",
                output_dimensionality=self.embedding_dims,
            ),
        )

        values = resp.embeddings[0].values

        if self.embedding_dims != 3072:
            values = self._normalize(values)

        return values

    def get_top_k_docs(self, question: str) -> list[tuple[str, int, str]]:
        """Fetch the top-k matching documents for a question.

        Args:
            question (str): User question.

        Returns:
            list[tuple[str, int, str]]: Retrieved rows as (document_name, page, text).
        """
        query_embedding = self.embed_query(question)

        with psycopg.connect(**self.db_kwargs) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                query = sql.SQL("""
                    SELECT document_name, page, text
                    FROM {table}
                    ORDER BY embedding <#> %s::vector
                    LIMIT %s
                """).format(table=sql.Identifier(self.table_name))
                cur.execute(query, (query_embedding, self.k))
                return cur.fetchall()

    def build_context(self, docs: list[tuple[str, int, str]]) -> str:
        """Build a context block from retrieved documents.

        Args:
            docs (list[tuple[str, int, str]]): Retrieved rows.

        Returns:
            str: Formatted context text.
        """
        if not docs:
            return ""

        return "\n\n".join(
            f"[{document_name} page {page}]\n{text}"
            for document_name, page, text in docs
        )

    def answer_question(self, question: str) -> str:
        """Answer a question using retrieved context.

        Args:
            question (str): User question.

        Returns:
            str: Gemini-generated answer.
        """
        docs = self.get_top_k_docs(question)
        context = self.build_context(docs)

        prompt = f"""{self.instructions}

Context:
{context}

Question:
{question}
"""

        resp = self.client.models.generate_content(
            model=self.answer_model,
            contents=prompt,
        )
        return resp.text or ""
