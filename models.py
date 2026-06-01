from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


IntentType = Literal[
    "pricing_question",
    "booking_request",
    "service_scope_question",
    "complaint",
    "urgent_booking",
    "human_handoff",
    "general_faq",
    "unknown",
]


class IntentResult(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class EmergencyCheck(BaseModel):
    is_emergency: bool
    is_today_booking: bool
    is_tomorrow_booking: bool
    is_complaint: bool
    reason: Optional[str] = None


class BookingDetails(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    service_type: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_slot: Optional[str] = None
    hours: Optional[int] = None
    address: Optional[str] = None
    special_instructions: Optional[str] = None


class BotResponse(BaseModel):
    message: str
    route_to_human: bool = False
    agent_used: str
    booking_details: Optional[BookingDetails] = None