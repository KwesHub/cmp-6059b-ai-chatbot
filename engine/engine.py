import collections
import collections.abc

# ─── Compatibility fix for Python 3.10+ ─────────────────
# experta uses older collection classes that were moved in
# Python 3.10. This patch fixes that before importing experta.
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


# ─── Facts ───────────────────────────────────────────────
# Facts are the things the engine "knows" about the
# current conversation. We declare them as we learn more
# from the user.

class UserRequest(Fact):
    """Everything we know about what the user wants."""
    pass
    # Fields we'll use:
    # intent       : book_ticket / predict_delay / unknown
    # origin       : station name
    # destination  : station name
    # origin_crs   : 3-letter CRS code
    # destination_crs : 3-letter CRS code
    # date         : travel date string
    # time         : travel time string
    # ticket_type  : single / return


class ConversationState(Fact):
    """Tracks where we are in the conversation."""
    pass
    # Fields we'll use:
    # stage  : greeting / collecting / searching / predicting / done
    # turns  : number of exchanges so far


# ─── Engine ──────────────────────────────────────────────
class TrainBotEngine(KnowledgeEngine):
    """
    The main reasoning engine for the chatbot.
    Rules are defined in engine/rules.py and loaded
    into this class in Milestone 5 card 3.
    For now this is the skeleton — we verify it
    runs correctly before adding rules.
    """

    def __init__(self):
        super().__init__()
        self.response = None  # stores what the bot should say next

    def get_response(self) -> str:
        """Return the bot's response after running the engine."""
        return self.response or "I'm not sure how to help with that. Could you rephrase?"


# ─── Engine runner ───────────────────────────────────────
def run_engine(intent: str, entities: dict) -> str:
    """
    Initialise the engine with what we know from NLP,
    run it, and return the bot's response.

    This is the function app.py will call.
    """
    engine = TrainBotEngine()
    engine.reset()

    # Declare what we know as Facts
    engine.declare(UserRequest(
        intent=intent,
        origin=entities.get("origin"),
        destination=entities.get("destination"),
        origin_crs=entities.get("origin_crs"),
        destination_crs=entities.get("destination_crs"),
        date=entities.get("date"),
        time=entities.get("time"),
    ))

    engine.declare(ConversationState(
        stage="collecting",
        turns=0
    ))

    engine.run()
    return engine.get_response()


# ─── Manual test ─────────────────────────────────────────
if __name__ == "__main__":
    # Test 1 — book ticket, all info present
    print("Test 1: book ticket with full info")
    response = run_engine(
        intent=INTENT_BOOK_TICKET,
        entities={
            "origin": "Norwich",
            "destination": "London Liverpool Street",
            "origin_crs": "NRW",
            "destination_crs": "LST",
            "date": "15th July",
            "time": None
        }
    )
    print(f"Bot: {response}\n")

    # Test 2 — book ticket, missing destination
    print("Test 2: book ticket, missing destination")
    response = run_engine(
        intent=INTENT_BOOK_TICKET,
        entities={
            "origin": "Norwich",
            "destination": None,
            "origin_crs": "NRW",
            "destination_crs": None,
            "date": None,
            "time": None
        }
    )
    print(f"Bot: {response}\n")

    # Test 3 — unknown intent
    print("Test 3: unknown intent")
    response = run_engine(
        intent=INTENT_UNKNOWN,
        entities={}
    )
    print(f"Bot: {response}\n")