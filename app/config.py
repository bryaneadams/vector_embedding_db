from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

APP_ENV = dotenv_values(APP_DIR / ".env")
DATABASE_ENV = dotenv_values(ROOT_DIR / "database" / ".env")


def _get(name: str, default: str = "") -> str:
    """Get a config value from the loaded environment files.

    Args:
        name (str): Environment variable name.
        default (str, optional): Fallback value if the key is missing. Defaults to "".

    Returns:
        str: The resolved value.
    """
    value = APP_ENV.get(name) or DATABASE_ENV.get(name)
    return default if value is None else str(value)


def _get_int(name: str, default: int) -> int:
    """Get an integer config value from the loaded environment files.

    Args:
        name (str): Environment variable name.
        default (int): Fallback value if the key is missing.

    Returns:
        int: The resolved integer value.
    """
    value = _get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from local .env files."""

    gemini_api_key: str
    embedding_model: str
    answer_model: str
    embedding_dims: int
    top_k: int
    table_name: str
    postgres_user: str
    postgres_password: str
    postgres_db: str
    db_port: int
    pgadmin_default_email: str
    pgadmin_default_password: str
    pgadmin_port: int

    @property
    def db_kwargs(self) -> dict[str, str]:
        """Return keyword arguments for psycopg connections.

        Returns:
            dict[str, str]: Connection arguments for `psycopg.connect()`.
        """
        return {
            "host": "localhost",
            "port": str(self.db_port),
            "dbname": self.postgres_db,
            "user": self.postgres_user,
            "password": self.postgres_password,
        }


settings = Settings(
    gemini_api_key=_get("GEMINI_API_KEY"),
    embedding_model=_get("EMBEDDING_MODEL", "gemini-embedding-001"),
    answer_model=_get("ANSWER_MODEL", "gemini-2.5-flash"),
    embedding_dims=_get_int("EMBEDDING_DIMS", 1536),
    top_k=_get_int("TOP_K", 3),
    table_name=_get("TABLE_NAME", "documents"),
    postgres_user=_get("POSTGRES_USER", "bryan"),
    postgres_password=_get("POSTGRES_PASSWORD", "bryan_rocks"),
    postgres_db=_get("POSTGRES_DB", "rag_testing"),
    db_port=_get_int("DB_PORT", 5432),
    pgadmin_default_email=_get("PGADMIN_DEFAULT_EMAIL", "admin@admin.com"),
    pgadmin_default_password=_get("PGADMIN_DEFAULT_PASSWORD", "bryan_rocks"),
    pgadmin_port=_get_int("PGADMIN_PORT", 5050),
)
