"""Environment-driven configuration shared by the example runners."""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Example runner configuration read from process environment variables."""

    openai_api_key: str | None = field(repr=False)
    openai_base_url: str
    openai_model: str
    chatgpt_backend: str


settings = Settings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
    chatgpt_backend=os.getenv("CHATGPT_BACKEND", "https://chatgpt.com/backend-api/"),
)
