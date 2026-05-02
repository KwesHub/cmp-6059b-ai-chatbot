import collections
import collections.abc
for _name in ["Mapping", "MutableMapping", "Iterable",
              "MutableSet", "Sequence"]:
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

from experta import *
from config import (
    INTENT_BOOK_TICKET,
    INTENT_PREDICT_DELAY,
    INTENT_UNKNOWN
)
from task1.ticket_api import find_cheapest_ticket
from task2.predict import predict_delay
from engine.knowledge_base import get_kb


# ─── Rules for book_ticket intent ────────────────────────

class TrainBotRules(KnowledgeEngine):

    def __init__(self):
        super().__init__()
        self.response = None

    # ── Step 1: Ask for origin ────────────────────────────
    @Rule(Fact(intent=INTENT_BOOK_TICKET),
          NOT(Fact(origin=W())))
    def ask_origin(self):
        self.response = (
            "Sure, I can help you find the cheapest train ticket! "
            "Where are you travelling from?"
        )

    # ── Step 2: Ask for destination ───────────────────────
    @Rule(Fact(intent=INTENT_BOOK_TICKET),
          Fact(origin=W()),
          NOT(Fact(destination=W())))
    def ask_destination(self):
        self.response = "Great — and where are you travelling to?"

    # ── Step 3: Ask for date ──────────────────────────────
    @Rule(Fact(intent=INTENT_BOOK_TICKET),
          Fact(origin=W()),
          Fact(destination=W()),
          NOT(Fact(date=W())))
    def ask_date(self):
        self.response = (
            "What date are you planning to travel? "
            "For example: 15th July."
        )

    # ── Step 4: Ask single or return ─────────────────────
    @Rule(Fact(intent=INTENT_BOOK_TICKET),
          Fact(origin=W()),
          Fact(destination=W()),
          Fact(date=W()),
          NOT(Fact(ticket_type=W())))
    def ask_ticket_type(self):
        self.response = "Would you like a single or return ticket?"

    # ── Step 5: Search for ticket ─────────────────────────
    @Rule(Fact(intent=INTENT_BOOK_TICKET),
          Fact(origin=MATCH.origin),
          Fact(destination=MATCH.destination),
          Fact(date=MATCH.date),
          Fact(ticket_type=MATCH.ticket_type),
          Fact(origin_crs=MATCH.origin_crs),
          Fact(destination_crs=MATCH.destination_crs))
    def search_ticket(self, origin, destination, date,
                      ticket_type, origin_crs, destination_crs):
        result = find_cheapest_ticket(
            origin_crs=origin_crs,
            destination_crs=destination_crs,
            date_string=date
        )
        if result["found"]:
            self.response = (
                f"Great news! The cheapest ticket from {origin} to "
                f"{destination} on {date} is {result['price']} "
                f"({result['ticket_type']}).\n"
                f"Departs: {result['departure'][:16].replace('T', ' ')}\n"
                f"Arrives: {result['arrival'][:16].replace('T', ' ')}\n"
                f"Book here: {result['booking_url']}"
            )
        else:
            self.response = (
                f"I couldn't find a fare right now. "
                f"You can search directly at: {result['booking_url']}"
            )

    # ─── Rules for predict_delay intent ──────────────────

    # ── Step 1: Ask for current station ──────────────────
    @Rule(Fact(intent=INTENT_PREDICT_DELAY),
          NOT(Fact(current_station=W())))
    def ask_current_station(self):
        self.response = (
            "I can help predict your arrival time. "
            "Which station is your train at currently?"
        )

    # ── Step 2: Ask for destination ───────────────────────
    @Rule(Fact(intent=INTENT_PREDICT_DELAY),
          Fact(current_station=W()),
          NOT(Fact(destination=W())))
    def ask_delay_destination(self):
        self.response = "Where are you heading to?"

    # ── Step 3: Ask for planned arrival time ─────────────
    @Rule(Fact(intent=INTENT_PREDICT_DELAY),
          Fact(current_station=W()),
          Fact(destination=W()),
          NOT(Fact(planned_arrival=W())))
    def ask_planned_arrival(self):
        self.response = (
            "What was the originally scheduled arrival time "
            "at your destination? For example: 11:30."
        )

    # ── Step 4: Predict delay ─────────────────────────────
    @Rule(Fact(intent=INTENT_PREDICT_DELAY),
          Fact(current_station=MATCH.current_station),
          Fact(destination=MATCH.destination),
          Fact(planned_arrival=MATCH.planned_arrival),
          Fact(current_station_crs=MATCH.crs),
          Fact(day_of_week=MATCH.dow),
          Fact(month=MATCH.month))
    def predict_arrival(self, current_station, destination,
                        planned_arrival, crs, dow, month):
        result = predict_delay(
            current_station_crs=crs,
            destination_crs="WAT",
            planned_arrival_time=planned_arrival,
            planned_departure_time=planned_arrival,
            direction="WEY2WAT",
            day_of_week=dow,
            month=month
        )
        delay = result["predicted_delay_minutes"]
        arrival = result["predicted_arrival"]
        confidence = result["confidence"]

        if delay > 0:
            delay_text = f"approximately {abs(delay):.0f} minutes late"
        elif delay < 0:
            delay_text = f"approximately {abs(delay):.0f} minutes early"
        else:
            delay_text = "on time"

        self.response = (
            f"Based on current conditions, your train is predicted to "
            f"arrive at {destination} {delay_text}.\n"
            f"Estimated arrival: {arrival}\n"
            f"Confidence: {confidence}"
        )

    # ─── Knowledge Base Query (for general questions) ──────
    @Rule(Fact(query=MATCH.question))
    def kb_query(self, question):
        """Search KB for answers to general questions."""
        kb = get_kb()
        match = kb.search_qa(question, threshold=0.6)

        if match:
            self.response = match["answer"]
        else:
            self.response = kb.get_fallback_response(0)

    # ─── Unknown intent fallback ──────────────────────────
    @Rule(Fact(intent=INTENT_UNKNOWN))
    def unknown_intent(self):
        self.response = (
            "I'm sorry, I didn't quite understand that. "
            "I can help you find a train ticket or predict "
            "your arrival time if your train is delayed. "
            "Which would you like help with?"
        )

    def get_response(self) -> str:
        return self.response or (
            "I'm not sure how to help with that. "
            "Could you rephrase your question?"
        )