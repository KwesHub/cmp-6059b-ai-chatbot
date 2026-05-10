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
from engine.rules import TrainBotRules


def run_engine(intent: str, entities: dict, extra_facts: dict = None) -> str:
    """
    Fire the experta rule engine with the current intent and known facts.
    experta uses forward-chaining: it matches declared facts against rules
    and fires the first rule whose conditions are satisfied.
    """
    engine = TrainBotRules()
    engine.reset()

    # declare intent as a fact so the rules can match on it
    engine.declare(Fact(intent=intent))

    # declare entity facts -- each one can trigger different rules
    if entities.get("origin"):
        engine.declare(Fact(origin=entities["origin"]))
    if entities.get("destination"):
        engine.declare(Fact(destination=entities["destination"]))
    if entities.get("origin_crs"):
        engine.declare(Fact(origin_crs=entities["origin_crs"]))
    if entities.get("destination_crs"):
        engine.declare(Fact(destination_crs=entities["destination_crs"]))
    if entities.get("date"):
        engine.declare(Fact(date=entities["date"]))
    if entities.get("time"):
        engine.declare(Fact(time=entities["time"]))

    # declare any facts collected across previous turns in the conversation
    if extra_facts:
        for key, value in extra_facts.items():
            if value is not None:
                engine.declare(Fact(**{key: value}))

    engine.run()
    return engine.get_response()


if __name__ == "__main__":
    print("Test 1: book_ticket -- no info yet")
    r = run_engine(INTENT_BOOK_TICKET, {})
    print(f"Bot: {r}\n")

    print("Test 2: book_ticket -- has origin only")
    r = run_engine(INTENT_BOOK_TICKET, {"origin": "Norwich", "origin_crs": "NRW"})
    print(f"Bot: {r}\n")

    print("Test 3: book_ticket -- has origin and destination")
    r = run_engine(INTENT_BOOK_TICKET, {
        "origin": "Norwich", "origin_crs": "NRW",
        "destination": "London Liverpool Street", "destination_crs": "LST"
    })
    print(f"Bot: {r}\n")

    print("Test 4: predict_delay -- no info yet")
    r = run_engine(INTENT_PREDICT_DELAY, {})
    print(f"Bot: {r}\n")

    print("Test 5: unknown intent")
    r = run_engine(INTENT_UNKNOWN, {})
    print(f"Bot: {r}\n")
