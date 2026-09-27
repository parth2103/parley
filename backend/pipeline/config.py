"""Validate local configuration before accepting calls; never contact providers."""

import os
from dataclasses import dataclass, field

import certifi


def configure_tls() -> None:
    """Supply trusted roots for Python installs without a system CA bundle."""
    # Honor explicitly configured trust stores, including corporate CA bundles.
    if not os.environ.get("SSL_CERT_FILE") and not os.environ.get("SSL_CERT_DIR"):
        os.environ["SSL_CERT_FILE"] = certifi.where()

BILLING_REMINDER = (
    "This project makes real, billed API calls (Anthropic has no free tier). "
    "Set a spend cap in your provider consoles before testing."
)


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class Settings:
    deepgram_api_key: str = field(repr=False)
    anthropic_api_key: str = field(repr=False, default="")
    tts_api_key: str = field(repr=False, default="")
    tts_voice_id: str = ""
    tts_provider: str = "cartesia"
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    llm_provider: str = "anthropic"
    groq_api_key: str = field(repr=False, default="")
    groq_model: str = DEFAULT_GROQ_MODEL

    @classmethod
    def from_env(cls):
        llm_provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
        if llm_provider not in {"anthropic", "groq"}:
            raise RuntimeError("LLM_PROVIDER must be anthropic or groq.")

        tts_provider = os.getenv("TTS_PROVIDER", "cartesia").strip().lower()
        if tts_provider not in {"cartesia", "elevenlabs"}:
            raise RuntimeError("TTS_PROVIDER must be cartesia or elevenlabs.")
        tts_prefix = tts_provider.upper()

        required_keys = ["DEEPGRAM_API_KEY", f"{tts_prefix}_API_KEY", f"{tts_prefix}_VOICE_ID"]
        if llm_provider == "anthropic":
            required_keys.append("ANTHROPIC_API_KEY")
        elif llm_provider == "groq":
            required_keys.append("GROQ_API_KEY")

        values = {name: os.getenv(name, "").strip() for name in required_keys}
        missing = [name for name, value in values.items() if not value or value.startswith("replace-with-")]
        if missing:
            raise RuntimeError(
                "Parley startup refused: missing configuration: " + ", ".join(missing)
                + ". Copy .env.example to .env and fill the selected provider credentials."
            )

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        anthropic_model = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL).strip() or DEFAULT_ANTHROPIC_MODEL
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL

        return cls(
            deepgram_api_key=values["DEEPGRAM_API_KEY"],
            anthropic_api_key=anthropic_api_key,
            tts_api_key=values[f"{tts_prefix}_API_KEY"],
            tts_voice_id=values[f"{tts_prefix}_VOICE_ID"],
            tts_provider=tts_provider,
            anthropic_model=anthropic_model,
            llm_provider=llm_provider,
            groq_api_key=groq_api_key,
            groq_model=groq_model,
        )
