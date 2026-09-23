import os
from pathlib import Path

# Load .env file manually if python-dotenv is not installed
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                clean_val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), clean_val)

class Settings:
    PROJECT_NAME: str = "Codebase Intelligence Platform"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # MongoDB Settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "codebase_intelligence")

    # Redis Settings (Active State & Short-Term Working Memory Cache)
    REDIS_URI: str = os.getenv("REDIS_URI", "redis://localhost:6379")
    
    # Neo4j Settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # ChromaDB Settings
    CHROMADB_DIR: str = os.getenv("CHROMADB_DIR", "./data/chroma_db")
    
    # LLM Provider Choice ("openai", "groq", "auto")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
    
    # Groq Settings (Ultra-fast LPU inference)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Evaluation judge — a **separate** provider and key from the one that answers questions.
    # RAGAS metrics are LLM-as-judge and cost 15-30 calls per item, so sharing the product's
    # provider would make an evaluation run compete with live answers for the same rate limit.
    # Defaults to Gemini through its OpenAI-compatible endpoint, which needs no extra client
    # code and no paid plan; any OpenAI-compatible endpoint works by changing these four.
    JUDGE_PROVIDER: str = os.getenv("JUDGE_PROVIDER", "gemini")
    JUDGE_API_KEY: str = os.getenv("JUDGE_API_KEY", "")
    JUDGE_BASE_URL: str = os.getenv(
        "JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gemini-2.0-flash")
    # Free tiers limit requests per minute far more tightly than per day, so the harness paces
    # itself rather than firing a dataset's worth of judgements at once.
    JUDGE_MAX_CONCURRENCY: int = int(os.getenv("JUDGE_MAX_CONCURRENCY", "2"))
    JUDGE_TIMEOUT_SECONDS: float = float(os.getenv("JUDGE_TIMEOUT_SECONDS", "60"))
    
    # Authentication (opt-in). When API_KEY is set, every /api route requires it via the
    # X-API-Key header or an Authorization: Bearer token. Left unset, auth is disabled and a
    # warning is logged at startup — this keeps local development frictionless while making
    # the unauthenticated state explicit rather than silent.
    API_KEY: str = os.getenv("API_KEY", "")
    API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")

    # CORS. Comma-separated origins. "*" cannot be combined with credentials (browsers reject
    # it), so credentials are enabled only when an explicit origin list is configured.
    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    )

    @property
    def cors_origin_list(self) -> list:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def cors_allows_any_origin(self) -> bool:
        return "*" in self.cors_origin_list

    # Storage
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    SAMPLE_REPO_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../sample_repo"))

    # Upload & archive extraction limits (guards against zip bombs / path traversal)
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", 250 * 1024 * 1024))        # 250 MB archive
    MAX_ARCHIVE_ENTRIES: int = int(os.getenv("MAX_ARCHIVE_ENTRIES", 20_000))             # entry count
    MAX_ENTRY_BYTES: int = int(os.getenv("MAX_ENTRY_BYTES", 25 * 1024 * 1024))           # 25 MB per file
    MAX_TOTAL_UNCOMPRESSED_BYTES: int = int(os.getenv("MAX_TOTAL_UNCOMPRESSED_BYTES", 1024 * 1024 * 1024))  # 1 GB total

    # Largest single file to index. Minified bundles, vendored libraries and checked-in datasets
    # blow past this and carry no value for code understanding.
    MAX_INDEXED_FILE_BYTES: int = int(os.getenv("MAX_INDEXED_FILE_BYTES", 2 * 1024 * 1024))  # 2 MB

    # Total prompt budget, in tokens. Groq's free tier allows 8,000 tokens per minute, and the
    # completion counts against it too, so the request itself must leave room to be answered.
    # A prompt that consumes the whole budget does not fail outright — it succeeds once and then
    # rate-limits everything for the next minute, which is how one query in four ended up
    # falling back to a template.
    MAX_PROMPT_TOKENS: int = int(os.getenv("MAX_PROMPT_TOKENS", 5000))

    # ── Observability ────────────────────────────────────────────
    # Tracing is off unless both keys are set. It is developer-facing and entirely optional: the
    # product answers identically without it, the same way it runs without Redis or Neo4j.
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3001")

    # How long shutdown waits for background indexing to finish before closing the stores.
    SHUTDOWN_GRACE_SECONDS: float = float(os.getenv("SHUTDOWN_GRACE_SECONDS", 60))

settings = Settings()
