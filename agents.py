from __future__ import annotations

import json
from datetime import date, timedelta

from langchain_core.prompts import ChatPromptTemplate

from config import llm
from knowledge_base import (
    COMPANY_KB,
    PRICING_KB,
    SERVICE_KB,
    BOOKING_RULES_KB,
    HANDOFF_KB,
)
from models import IntentResult, EmergencyCheck, BookingDetails, BotResponse, SalesSummary

def kb_to_prompt_text(kb: dict) -> str:
    """
    Converts Python dictionaries into safe JSON text for LangChain prompts.

    Why this is needed:
    LangChain treats {something} as a prompt variable.
    Python dicts contain curly braces, so raw dicts can cause KeyError.
    """
    return json.dumps(kb, indent=2, ensure_ascii=False)


def classify_intent(message: str) -> IntentResult:
    structured_llm = llm.with_structured_output(IntentResult)

    system_prompt = """
<agent>
    <name>Intent Router Agent</name>
</agent>

<role>
Classify the latest user message into exactly one intent.
</role>

<objective>
Every new customer message must be routed correctly before any reply is written.
Intent can change abruptly, so classify only the latest user message.
</objective>

<intents>
- pricing_question
- booking_request
- service_scope_question
- complaint
- urgent_booking
- human_handoff
- general_faq
- unknown
</intents>

<rules>
1. Classify only the latest user message.
2. Do not assume the previous intent is still active.
3. Complaints include damaged property, poor cleaning, late cleaner, refund requests, anger, or dissatisfaction.
4. Same-day or next-day booking requests are urgent_booking.
5. Pricing questions are pricing_question.
6. Requests to arrange cleaning are booking_request.
7. If unclear, use unknown.
</rules>

<validation>
Before returning, check:
1. Did I classify only the latest message?
2. Did I ignore old intent if the topic changed?
3. Did I detect complaints correctly?
4. Did I detect today/tomorrow booking correctly?
5. Did I choose exactly one intent?

If any validation check fails, correct the intent before returning.
</validation>

<output_contract>
Return IntentResult only.
</output_contract>
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{message}"),
        ]
    )

    return structured_llm.invoke(prompt.format_messages(message=message))


def emergency_guardrail(message: str, intent: IntentResult) -> EmergencyCheck:
    structured_llm = llm.with_structured_output(EmergencyCheck)

    today = date.today()
    tomorrow = today + timedelta(days=1)

        handoff_kb_text = kb_to_prompt_text(HANDOFF_KB)

    system_prompt = f"""
<agent>
    <name>Emergency Guardrail Agent</name>
</agent>

<role>
Detect whether the customer must be routed to a human immediately.
</role>

<objective>
Protect the business from urgent, sensitive, or complaint-related cases.
</objective>

<dates>
    <today>{today.isoformat()}</today>
    <tomorrow>{tomorrow.isoformat()}</tomorrow>
</dates>

<emergency_triggers>
1. Customer is making a complaint.
2. Customer wants booking today.
3. Customer wants booking tomorrow.
</emergency_triggers>

<complaint_examples>
- cleaner was late
- cleaner did not show up
- cleaner damaged something
- poor cleaning
- refund request
- angry customer
- unhappy customer
</complaint_examples>

<handoff_rules>
{handoff_kb_text}
</handoff_rules>

<rules>
1. If the message is a complaint, set is_emergency = true.
2. If the booking date is today, set is_emergency = true.
3. If the booking date is tomorrow, set is_emergency = true.
4. If any emergency trigger is true, route to human.
5. Do not continue normal booking flow for emergency cases.
</rules>

<validation>
Before returning, check:
1. Is the message a complaint?
2. Is the requested date today?
3. Is the requested date tomorrow?
4. If any answer is yes, is_emergency must be true.
5. If is_emergency is true, provide a clear reason.
6. Never mark same-day or next-day booking as non-emergency.

If any validation check fails, correct the emergency decision before returning.
</validation>

<output_contract>
Return EmergencyCheck only.
</output_contract>
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
<latest_user_message>
{message}
</latest_user_message>

<intent_result>
{intent}
</intent_result>
""",
            ),
        ]
    )

    return structured_llm.invoke(
        prompt.format_messages(
            message=message,
            intent=intent.model_dump_json(),
        )
    )


def pricing_agent(message: str) -> BotResponse:
    response = f"""
For general cleaning, our pricing is:

- ${PRICING_KB["three_hour_rate"]}/hour for a 3-hour booking
- ${PRICING_KB["four_hour_rate"]}/hour for bookings of 4 hours or more

The minimum booking is {PRICING_KB["minimum_hours"]} hours.
"""

    return BotResponse(
        message=response.strip(),
        route_to_human=False,
        agent_used="pricing_agent",
    )


def service_scope_agent(message: str) -> BotResponse:
    response = f"""
We provide general cleaning services all over Singapore.

We do not handle dangerous, hazardous, professional, specialist, industrial, or high-risk cleaning jobs.

Our company address is {COMPANY_KB["address"]}.
"""

    return BotResponse(
        message=response.strip(),
        route_to_human=False,
        agent_used="service_scope_agent",
    )


def booking_agent(
    message: str,
    existing_details: BookingDetails | None = None,
) -> BotResponse:
    structured_llm = llm.with_structured_output(BookingDetails)
    existing_details = existing_details or BookingDetails()

    booking_rules_kb_text = kb_to_prompt_text(BOOKING_RULES_KB)
    service_kb_text = kb_to_prompt_text(SERVICE_KB)

    system_prompt = f"""
<agent>
    <name>Booking Collection Agent</name>
</agent>

<role>
Collect booking details for human staff.
</role>

<objective>
Gather required booking information without confirming the booking.
</objective>

<knowledge>
    <booking_rules>
{booking_rules_kb_text}
    </booking_rules>

    <service_rules>
{service_kb_text}
    </service_rules>
</knowledge>

<critical_rules>
1. Never confirm a booking.
2. Never reserve a slot.
3. Never promise cleaner availability.
4. Never say "see you then".
5. Only collect information.
6. Human staff must confirm all bookings.
</critical_rules>

<booking_constraints>
- Service hours: 9am to 9pm
- Latest start time: 6pm
- Minimum booking: 3 hours
- Slots: morning, afternoon, evening
- Service type: general cleaning only
</booking_constraints>

<required_fields>
- name
- phone
- service_type
- preferred_date
- preferred_slot
- hours
- address
- special_instructions
</required_fields>

<rules>
1. Extract details from the latest message.
2. Merge with existing booking details.
3. Do not erase existing details unless the user corrects them.
4. Ask only for missing details.
5. If all details are collected, say the team will follow up.
6. Do not confirm the booking.
</rules>

<validation>
Before returning, check:
1. Did I accidentally confirm the booking?
2. Did I imply the slot is reserved?
3. Did I promise availability?
4. Did I ask for information already provided?
5. Did I collect only general cleaning requests?
6. Is the booking at least 3 hours?
7. Is the requested start time no later than 6pm?
8. If the date is today or tomorrow, this should have been routed to human.

If any validation check fails, correct the output before returning.
</validation>

<output_contract>
Return BookingDetails only.
</output_contract>
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
<existing_booking_details>
{existing_details}
</existing_booking_details>

<latest_user_message>
{message}
</latest_user_message>
""",
            ),
        ]
    )

    updated_details = structured_llm.invoke(
        prompt.format_messages(
            existing_details=existing_details.model_dump_json(),
            message=message,
        )
    )

    missing_fields = [
        field
        for field, value in updated_details.model_dump().items()
        if value in [None, ""]
    ]

    if missing_fields:
        next_field = missing_fields[0].replace("_", " ")
        reply = f"Could I get your {next_field}?"
    else:
        reply = (
            "Thank you. I’ll pass these details to our team to check availability "
            "and follow up. This is not a confirmed booking yet."
        )

    return BotResponse(
        message=reply,
        route_to_human=False,
        agent_used="booking_agent",
        booking_details=updated_details,
    )


def faq_agent(message: str) -> BotResponse:
    company_kb_text = kb_to_prompt_text(COMPANY_KB)
    pricing_kb_text = kb_to_prompt_text(PRICING_KB)
    service_kb_text = kb_to_prompt_text(SERVICE_KB)
    booking_rules_kb_text = kb_to_prompt_text(BOOKING_RULES_KB)

    system_prompt = f"""
<agent>
    <name>FAQ Agent</name>
</agent>

<role>
Answer general customer questions.
</role>

<knowledge>
    <company>
{company_kb_text}
    </company>

    <pricing>
{pricing_kb_text}
    </pricing>

    <service>
{service_kb_text}
    </service>

    <booking_rules>
{booking_rules_kb_text}
    </booking_rules>
</knowledge>

<rules>
1. Use only provided knowledge.
2. Do not invent company policies.
3. Do not confirm bookings.
4. If unsure, route to human.
</rules>

<validation>
Before replying, check:
1. Is the answer fully supported by the knowledge base?
2. Did I avoid inventing policies?
3. Did I avoid confirming bookings?
4. Is this actually an FAQ, or should it be routed elsewhere?
5. Is the message a complaint, today booking, or tomorrow booking?

If any validation check fails, correct the response or say a human staff member will follow up.
</validation>

<output_contract>
Return a concise customer-facing answer.
</output_contract>
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{message}"),
        ]
    )

    result = llm.invoke(prompt.format_messages(message=message))

    return BotResponse(
        message=result.content,
        route_to_human=False,
        agent_used="faq_agent",
    )


def human_handoff_agent(reason: str) -> BotResponse:
    message = f"""
Thanks for sharing this. I’ll pass this to our team so a human staff member can assist you directly.

Reason: {reason}
"""

    return BotResponse(
        message=message.strip(),
        route_to_human=True,
        agent_used="human_handoff_agent",
    )


def response_guardrail(response: BotResponse) -> BotResponse:
    forbidden_phrases = [
        "your booking is confirmed",
        "booking confirmed",
        "confirmed your booking",
        "see you then",
        "we have booked you",
        "your slot is reserved",
        "cleaner will arrive",
    ]

    lowered = response.message.lower()

    if any(phrase in lowered for phrase in forbidden_phrases):
        response.message = (
            "Thank you. I’ll pass these details to our team to check availability "
            "and follow up. This is not a confirmed booking yet."
        )
        response.route_to_human = True
        response.agent_used = f"{response.agent_used}+response_guardrail"

    return response
    
def summary_agent(
    latest_user_message: str,
    bot_response: BotResponse | None,
    existing_summary: SalesSummary | None = None,
    booking_details: BookingDetails | None = None,
    emergency: EmergencyCheck | None = None,
    intent: IntentResult | None = None,
) -> SalesSummary:
    structured_llm = llm.with_structured_output(SalesSummary)

    existing_summary = existing_summary or SalesSummary()

    system_prompt = """
<agent>
    <name>Salesperson Summary Agent</name>
</agent>

<role>
Maintain an internal salesperson-only summary.
</role>

<objective>
Update customer information and summarize the conversation so far.
This summary is only for staff and must not be shown to the customer.
</objective>

<rules>
1. Update the summary every turn.
2. Preserve existing information unless the customer corrects it.
3. Extract customer details from the latest message.
4. Include urgent flags such as complaint, today booking, or tomorrow booking.
5. Include what the customer wants.
6. Include missing information needed for follow-up.
7. Do not invent information.
</rules>

<validation>
Before returning, check:
1. Did I preserve known customer information?
2. Did I update new information from the latest message?
3. Did I mark human handoff if needed?
4. Did I summarize the conversation clearly for a salesperson?
5. Did I avoid inventing missing details?
</validation>

<output_contract>
Return SalesSummary only.
</output_contract>
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
<existing_sales_summary>
{existing_summary}
</existing_sales_summary>

<latest_user_message>
{latest_user_message}
</latest_user_message>

<latest_bot_response>
{bot_response}
</latest_bot_response>

<booking_details>
{booking_details}
</booking_details>

<intent>
{intent}
</intent>

<emergency>
{emergency}
</emergency>
""",
            ),
        ]
    )

    return structured_llm.invoke(
        prompt.format_messages(
            existing_summary=existing_summary.model_dump_json(),
            latest_user_message=latest_user_message,
            bot_response=bot_response.model_dump_json() if bot_response else "",
            booking_details=booking_details.model_dump_json() if booking_details else "",
            intent=intent.model_dump_json() if intent else "",
            emergency=emergency.model_dump_json() if emergency else "",
        )
    )
