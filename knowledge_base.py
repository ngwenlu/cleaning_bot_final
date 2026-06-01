COMPANY_KB = {
    "company_name": "Cleaning Company",
    "address": "865 Mountbatten Road",
    "coverage": "All over Singapore",
}

PRICING_KB = {
    "currency": "SGD",
    "minimum_hours": 3,
    "three_hour_rate": 21,
    "four_hour_minimum": 4,
    "four_hour_rate": 18,
}

SERVICE_KB = {
    "allowed_services": ["general cleaning"],
    "not_allowed_services": [
        "dangerous places",
        "hazardous cleaning",
        "professional cleaning",
        "specialist cleaning",
        "high-risk cleaning",
        "industrial cleaning",
    ],
}

BOOKING_RULES_KB = {
    "service_start_time": "09:00",
    "service_end_time": "21:00",
    "latest_start_time": "18:00",
    "minimum_hours": 3,
    "requires_specific_time": True,
    "never_confirm_booking": True,
}

HANDOFF_KB = {
    "route_to_human_immediately": [
        "complaints",
        "same-day bookings",
        "next-day bookings",
        "dangerous cleaning requests",
        "requests to confirm booking",
    ],
}

DATE_RULES_KB = {
    "booking_date_must_be_future": True,
    "max_booking_months_ahead": 6,
    "complaint_session_date_must_be_past_or_today": True,
}
