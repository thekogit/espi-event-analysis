"""LLM structured extraction from regulatory announcements."""
from __future__ import annotations

import json
import re
import time
from typing import Optional
from pydantic import ValidationError
from . import config
from .schema import ExtractedEvent

SYSTEM_PROMPT = """Jesteś analitykiem rynku kapitałowego. Czytasz raport bieżący ESPI spółki notowanej na GPW
i zwracasz obiekt JSON o dokładnie następującym schemacie:
{
  "event_type": "BUYBACK | CAPITAL_RAISE | MATERIAL_CONTRACT | MANAGEMENT_CHANGE | HOLDER_CHANGE | FINANCIAL_RESULTS | GUIDANCE | MNA | SHAREHOLDER_MEETING | DIVIDEND | LEGAL_REGULATORY | OTHER",
  "amount": null lub liczba dodatnia,
  "currency": "PLN" | "EUR" | "USD" | "OTHER" lub null,
  "counterparty": null lub string,
  "effective_date": "YYYY-MM-DD" lub null,
  "shareholder_impact": "POSITIVE" | "NEUTRAL" | "NEGATIVE",
  "evidence_quote": "maksymalnie 40 słów skopiowanych dosłownie z treści raportu uzasadniających typ zdarzenia"
}

Zasady:
1. Wybierz dokładnie jeden główny typ zdarzenia (event_type). W razie braku pewności użyj OTHER.
2. Cytat evidence_quote musi być DOSŁOWNYM fragmentem tekstu (max 40 słów).
3. Nie dodawaj markdown ani dodatkowych komentarzy.
"""


def strip_fences(text: str) -> str:
    """Remove markdown code fences from JSON output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation for exact quote matching."""
    return re.sub(r"\s+", " ", text).strip().lower().replace("„", '"').replace("”", '"')


def is_quote_present(quote: str, body: str) -> bool:
    """Verify that the evidence quote appears verbatim within the report body."""
    return bool(quote) and normalize_text(quote) in normalize_text(body)


quote_found = is_quote_present


def make_client():
    """Create an OpenAI-compatible client instance."""
    from openai import OpenAI
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def extract_one(client, model: str, subject: str, body: str, max_retries: int = 2) -> dict:
    """Extract one event with automatic retry on schema validation failure."""
    trimmed_body = body[: config.MAX_BODY_CHARS]
    user_content = f"TEMAT: {subject or '(brak)'}\n\nTREŚĆ RAPORTU:\n{trimmed_body}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    tokens_in = tokens_out = 0
    started = time.perf_counter()
    last_error: Optional[str] = None

    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            last_error = f"api_error: {e}"
            break

        if hasattr(resp, "usage") and resp.usage:
            tokens_in += getattr(resp.usage, "prompt_tokens", 0) or 0
            tokens_out += getattr(resp.usage, "completion_tokens", 0) or 0

        content = resp.choices[0].message.content or ""
        try:
            event = ExtractedEvent.model_validate_json(strip_fences(content))
        except ValidationError as e:
            last_error = f"invalid_json: {e.errors()[:3]}"
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Odpowiedź nie spełnia schematu: {e.errors()[:3]}. Zwróć poprawiony JSON."}
            ]
            continue

        latency = time.perf_counter() - started
        return {
            "ok": True,
            "event": event.model_dump(),
            "quote_found": quote_found(event.evidence_quote, trimmed_body),
            "error": None,
            "latency_s": round(latency, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out
        }

    return {
        "ok": False,
        "event": None,
        "quote_found": False,
        "error": last_error or "max retries exceeded",
        "latency_s": round(time.perf_counter() - started, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out
    }


def extract_event(client, subject: str, body: str, model: Optional[str] = None) -> dict:
    """Convenience single-call extraction wrapper."""
    target_model = model or config.LLM_MODEL
    return extract_one(client, target_model, subject, body)
