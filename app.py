import streamlit as st

from graph import chat_graph
from models import BookingDetails, SalesSummary, ComplaintDetails


st.set_page_config(
    page_title="Cleaning Company Sales Assistant",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 Cleaning Company Sales Assistant")

if "complaint_details" not in st.session_state:
    st.session_state.complaint_details = ComplaintDetails()
    
if "messages" not in st.session_state:
    st.session_state.messages = []

if "booking_details" not in st.session_state:
    st.session_state.booking_details = BookingDetails()

if "sales_summary" not in st.session_state:
    st.session_state.sales_summary = SalesSummary()


# -----------------------------
# Salesperson-only summary panel
# -----------------------------

st.subheader("Salesperson Summary")

summary = st.session_state.sales_summary

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Customer name:** {summary.customer_name or 'Unknown'}")
        st.write(f"**Phone:** {summary.phone or 'Unknown'}")
        st.write(f"**Service requested:** {summary.service_requested or 'Unknown'}")
        st.write(f"**Preferred date:** {summary.preferred_date or 'Unknown'}")
        st.write(f"**Preferred time:** {summary.preferred_time or 'Unknown'}")

    with col2:
        st.write(f"**Hours:** {summary.hours or 'Unknown'}")
        st.write(f"**Customer address:** {summary.customer_address or 'Unknown'}")
        st.write(f"**Urgency status:** {summary.urgency_status or 'Normal'}")
        st.write(f"**Route to human:** {summary.route_to_human}")

    st.markdown("**Conversation summary:**")
    st.write(summary.conversation_summary or "No conversation yet.")

    st.markdown("**Missing info:**")
    st.write(", ".join(summary.missing_info) if summary.missing_info else "None")


st.divider()


# -----------------------------
# Chat history
# -----------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------------
# Chat input
# -----------------------------

user_message = st.chat_input("Ask about cleaning services, pricing, or booking...")

if user_message:
    st.session_state.messages.append(
        {"role": "user", "content": user_message}
    )

    with st.chat_message("user"):
        st.write(user_message)

    state = {
        "latest_user_message": user_message,
        "intent": None,
        "emergency": None,
        "booking_details": st.session_state.booking_details,
        "response": None,
        "sales_summary": st.session_state.sales_summary,
        "complaint_details": st.session_state.complaint_details,
    }

    result = chat_graph.invoke(state)

    bot_response = result["response"]

    if bot_response.complaint_details:
        st.session_state.complaint_details = bot_response.complaint_details

    if bot_response.booking_details:
        st.session_state.booking_details = bot_response.booking_details

    if result.get("sales_summary"):
        st.session_state.sales_summary = result["sales_summary"]

    st.session_state.messages.append(
        {"role": "assistant", "content": bot_response.message}
    )

    with st.chat_message("assistant"):
        st.write(bot_response.message)

        with st.expander("Debug info"):
            st.write("Agent used:", bot_response.agent_used)
            st.write("Route to human:", bot_response.route_to_human)
            st.write("Intent:", result["intent"])
            st.write("Emergency:", result["emergency"])
            st.write("Booking details:", result["booking_details"])
            st.write("Sales summary:", result["sales_summary"])

    st.rerun()
