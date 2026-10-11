"""Event schema and human-readable metadata definitions."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(str, Enum):
    BUYBACK = "BUYBACK"
    CAPITAL_RAISE = "CAPITAL_RAISE"
    MATERIAL_CONTRACT = "MATERIAL_CONTRACT"
    MANAGEMENT_CHANGE = "MANAGEMENT_CHANGE"
    HOLDER_CHANGE = "HOLDER_CHANGE"
    FINANCIAL_RESULTS = "FINANCIAL_RESULTS"
    GUIDANCE = "GUIDANCE"
    MNA = "MNA"
    SHAREHOLDER_MEETING = "SHAREHOLDER_MEETING"
    DIVIDEND = "DIVIDEND"
    LEGAL_REGULATORY = "LEGAL_REGULATORY"
    OTHER = "OTHER"


EVENT_DISPLAY_NAMES = {
    "BUYBACK": "Share Buyback",
    "CAPITAL_RAISE": "Capital Raise",
    "MATERIAL_CONTRACT": "Material Contract",
    "MANAGEMENT_CHANGE": "Management Change",
    "HOLDER_CHANGE": "Shareholder Stake Change",
    "FINANCIAL_RESULTS": "Financial Results",
    "GUIDANCE": "Earnings Guidance",
    "MNA": "Mergers & Acquisitions",
    "SHAREHOLDER_MEETING": "Shareholder Meeting",
    "DIVIDEND": "Dividend Announcement",
    "LEGAL_REGULATORY": "Legal & Regulatory",
    "OTHER": "Other Corporate Notices",
}


class Impact(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class ExtractedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    amount: Optional[float] = Field(None, description="Primary monetary amount involved, in local currency.")
    currency: Optional[Literal["PLN", "EUR", "USD", "OTHER"]] = None
    counterparty: Optional[str] = Field(None, max_length=200)
    effective_date: Optional[date] = None
    shareholder_impact: Impact
    evidence_quote: str = Field(..., min_length=5, max_length=400)

    @field_validator("evidence_quote")
    @classmethod
    def quote_is_short(cls, v: str) -> str:
        if len(v.split()) > 40:
            raise ValueError("evidence_quote must be at most 40 words")
        return v.strip()

    @field_validator("amount")
    @classmethod
    def amount_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("amount must be >= 0")
        return v
