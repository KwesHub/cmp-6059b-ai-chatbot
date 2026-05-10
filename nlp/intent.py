import spacy
from config import SPACY_MODEL, INTENT_BOOK_TICKET, INTENT_PREDICT_DELAY, INTENT_ADD_RULE, INTENT_UNKNOWN

# load the pre-trained English spaCy model once at import time
nlp = spacy.load(SPACY_MODEL)


def get_intent(user_input: str) -> str:
    """Work out what the user wants: book a ticket, predict a delay, add a rule, or unknown."""
    doc = nlp(user_input.lower())

    # spaCy gives each word a dependency label; ROOT is the main verb,
    # dobj is the direct object -- e.g. in "find me a ticket", root=find, dobj=ticket
    root = None
    dobj = None

    for token in doc:
        if token.dep_ == "ROOT":
            root = token.lemma_
        if token.dep_ == "dobj":
            dobj = token.lemma_

    # collect all lemmas so we can do simple keyword lookups below
    all_words = {token.lemma_ for token in doc}

    # check for "add a rule / teach me something" intent first,
    # because some words (e.g. "learn") overlap with booking phrases
    ka_verbs = {"add", "teach", "learn", "remember", "save", "store", "create"}
    ka_nouns = {"rule", "fact", "knowledge", "answer", "response", "information"}

    if (root in ka_verbs and all_words & ka_nouns) or \
       ("add" in all_words and "rule" in all_words) or \
       ("teach" in all_words) or \
       ("remember this" in user_input.lower()):
        return INTENT_ADD_RULE

    # keyword sets for the two main tasks
    ticket_verbs = {"find", "book", "buy", "get", "search", "check", "want", "need", "look", "help"}
    ticket_nouns = {"ticket", "fare", "price", "cheap", "cost", "journey", "travel", "book"}

    delay_verbs = {"predict", "estimate", "check", "calculate", "know", "find"}
    delay_nouns = {"delay", "late", "arrival", "arrive", "time", "delayed", "stuck", "when"}

    # first try the root verb and direct object -- more accurate than a bag-of-words check
    if root in ticket_verbs or dobj in ticket_nouns:
        return INTENT_BOOK_TICKET

    if root in delay_verbs or dobj in delay_nouns:
        return INTENT_PREDICT_DELAY

    # fallback: scan all tokens for any matching keyword
    if all_words & ticket_nouns:
        return INTENT_BOOK_TICKET

    if all_words & delay_nouns:
        return INTENT_PREDICT_DELAY

    return INTENT_UNKNOWN


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
        print(f"'{t}'\n  -> {get_intent(t)}\n")
