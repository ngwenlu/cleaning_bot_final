from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents import (
    classify_intent,
    emergency_guardrail,
    pricing_agent,
    service_scope_agent,
    booking_agent,
    faq_agent,
    human_handoff_agent,
    response_guardrail,
    summary_agent,
)
from models import BookingDetails, IntentResult, EmergencyCheck, BotResponse, SalesSummary

class ChatState(TypedDict):
    latest_user_message: str
    intent: Optional[IntentResult]
    emergency: Optional[EmergencyCheck]
    booking_details: Optional[BookingDetails]
    response: Optional[BotResponse]
    sales_summary: Optional[SalesSummary]


def intent_node(state: ChatState) -> ChatState:
    state["intent"] = classify_intent(state["latest_user_message"])
    return state


def emergency_node(state: ChatState) -> ChatState:
    state["emergency"] = emergency_guardrail(
        message=state["latest_user_message"],
        intent=state["intent"],
    )
    return state


def route_after_emergency(state: ChatState) -> str:
    emergency = state["emergency"]
    intent = state["intent"]

    if emergency and emergency.is_emergency:
        return "human_handoff"

    if intent is None:
        return "human_handoff"

    if intent.intent == "pricing_question":
        return "pricing"

    if intent.intent == "booking_request":
        return "booking"

    if intent.intent == "service_scope_question":
        return "service_scope"

    if intent.intent in ["complaint", "urgent_booking", "human_handoff"]:
        return "human_handoff"

    return "faq"


def pricing_node(state: ChatState) -> ChatState:
    state["response"] = pricing_agent(state["latest_user_message"])
    return state


def service_scope_node(state: ChatState) -> ChatState:
    state["response"] = service_scope_agent(state["latest_user_message"])
    return state


def booking_node(state: ChatState) -> ChatState:
    response = booking_agent(
        message=state["latest_user_message"],
        existing_details=state.get("booking_details"),
    )
    state["booking_details"] = response.booking_details
    state["response"] = response
    return state


def faq_node(state: ChatState) -> ChatState:
    state["response"] = faq_agent(state["latest_user_message"])
    return state


def human_handoff_node(state: ChatState) -> ChatState:
    reason = "This request requires human assistance."

    if state.get("emergency") and state["emergency"].reason:
        reason = state["emergency"].reason
    elif state.get("intent"):
        reason = state["intent"].reasoning

    state["response"] = human_handoff_agent(reason)
    return state


def guardrail_node(state: ChatState) -> ChatState:
    state["response"] = response_guardrail(state["response"])
    return state
    
def summary_node(state: ChatState) -> ChatState:
    state["sales_summary"] = summary_agent(
        latest_user_message=state["latest_user_message"],
        bot_response=state.get("response"),
        existing_summary=state.get("sales_summary"),
        booking_details=state.get("booking_details"),
        emergency=state.get("emergency"),
        intent=state.get("intent"),
    )
    return state


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("intent", intent_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("pricing", pricing_node)
    graph.add_node("service_scope", service_scope_node)
    graph.add_node("booking", booking_node)
    graph.add_node("faq", faq_node)
    graph.add_node("human_handoff", human_handoff_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("summary", summary_node)

    graph.set_entry_point("intent")

    graph.add_edge("intent", "emergency")
    
    graph.add_conditional_edges(
        "emergency",
        route_after_emergency,
        {
            "pricing": "pricing",
            "service_scope": "service_scope",
            "booking": "booking",
            "faq": "faq",
            "human_handoff": "human_handoff",
        },
    )

    graph.add_edge("pricing", "guardrail")
    graph.add_edge("service_scope", "guardrail")
    graph.add_edge("booking", "guardrail")
    graph.add_edge("faq", "guardrail")
    graph.add_edge("human_handoff", "guardrail")
    graph.add_edge("guardrail", "summary")

    graph.add_edge("summary", END)

    return graph.compile()


chat_graph = build_graph()
