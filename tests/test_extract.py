import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from espi.extract import extract_one, quote_found, strip_fences
from espi.schema import ExtractedEvent

BODY = "Zarząd informuje o rekomendacji wypłaty dywidendy w łącznej kwocie 120 000 000 zł."
GOOD = {
    "event_type": "DIVIDEND", "amount": 120000000.0, "currency": "PLN", "counterparty": None,
    "effective_date": None, "shareholder_impact": "POSITIVE",
    "evidence_quote": "rekomendacji wypłaty dywidendy w łącznej kwocie 120 000 000 zł",
}


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create with scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls = 0

    def _create(self, **kwargs):
        self.calls += 1
        content = self.replies.pop(0)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def test_schema_accepts_good_and_rejects_unknown_fields():
    ExtractedEvent.model_validate(GOOD)
    with pytest.raises(ValidationError):
        ExtractedEvent.model_validate({**GOOD, "extra": 1})
    with pytest.raises(ValidationError):
        ExtractedEvent.model_validate({**GOOD, "event_type": "PARTY"})


def test_quote_found_ignores_case_and_spaces():
    assert quote_found("REKOMENDACJI   wypłaty dywidendy", BODY)
    assert not quote_found("wypłaty zaliczki", BODY)


def test_strip_fences():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_retries_after_invalid_json():
    client = FakeClient(["not json at all", "```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```"])
    rec = extract_one(client, "fake-model", "Dywidenda", BODY)
    assert rec["ok"] is True
    assert client.calls == 2
    assert rec["event"]["event_type"] == "DIVIDEND"
    assert rec["quote_found"] is True
    assert rec["tokens_in"] == 200


def test_extract_gives_up_after_retries():
    client = FakeClient(["{}", "{}", "{}"])
    rec = extract_one(client, "fake-model", "x", BODY, max_retries=2)
    assert rec["ok"] is False
    assert rec["error"].startswith("invalid_json")
