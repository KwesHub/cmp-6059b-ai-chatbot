import spacy
from config import SPACY_MODEL, INTENT_BOOK_TICKET, INTENT_PREDICT_DELAY, INTENT_ADD_RULE, INTENT_UNKNOWN

# ─── Load spaCy model ────────────────────────────────────
nlp = spacy.load(SPACY_MODEL)


def get_intent(user_input: str) -> str:
    """
    Analyse the user's message and return one of the intents:
    - INTENT_BOOK_TICKET    → user wants to find/book a train ticket
    - INTENT_PREDICT_DELAY  → user wants to predict a delay
    - INTENT_ADD_RULE       → user wants to teach the bot a new rule
    - INTENT_UNKNOWN        → couldn't work it out
    """
    doc = nlp(user_input.lower())

    # ─── Extract root verb + direct object pairs ─────────
    root = None
    dobj = None

    for token in doc:
        if token.dep_ == "ROOT":
            root = token.lemma_
        if token.dep_ == "dobj":
            dobj = token.lemma_

    # ─── Keyword matching for add_rule intent ────────────
    ka_verbs = {"add", "teach", "learn", "remember", "save", "store", "create"}
    ka_nouns = {"rule", "fact", "knowledge", "answer", "response", "information"}

    all_words = {token.lemma_ for token in doc}

    if (root in ka_verbs and all_words & ka_nouns) or \
       ("add" in all_words and "rule" in all_words) or \
       ("teach" in all_words) or \
       ("remember this" in user_input.lower()):
        return INTENT_ADD_RULE

    # ─── Keyword matching for book_ticket intent ─────────
    ticket_verbs = {"find", "book", "buy", "get", "search", "check", "want", "need", "look", "help"}
    ticket_nouns = {"ticket", "fare", "price", "cheap", "cost", "journey", "travel", "book"}

    # ─── Keyword matching for predict_delay intent ────────
    delay_verbs = {"predict", "estimate", "check", "calculate", "know", "find"}
    delay_nouns = {"delay", "late", "arrival", "arrive", "time", "delayed", "stuck", "when"}

    # Check root verb and dobj against keyword sets
    if root in ticket_verbs or dobj in ticket_nouns:
        return INTENT_BOOK_TICKET

    if root in delay_verbs or dobj in delay_nouns:
        return INTENT_PREDICT_DELAY

    # ─── Fallback: scan all tokens for keywords ───────────
    if all_words & ticket_nouns:
        return INTENT_BOOK_TICKET

    if all_words & delay_nouns:
        return INTENT_PREDICT_DELAY

    return INTENT_UNKNOWN


# ─── Manual test ─────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "I want to find the cheapest ticket from Norwich to London",
        "Can you help me book a train?",
        "My train is delayed, when will it arrive?",
        "How late will I be?",
        "What time does my train get to London Waterloo?",
        "hello there",
        "asdfghjkl",
    ]
    for t in tests:
        print(f"'{t}'\n  → {get_intent(t)}\n")