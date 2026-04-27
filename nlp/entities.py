import spacy
from config import SPACY_MODEL, INTENT_BOOK_TICKET, INTENT_PREDICT_DELAY
from database import get_connection

# ─── Load spaCy model ────────────────────────────────────
nlp = spacy.load(SPACY_MODEL)


# ─── Station code lookup from database ───────────────────
def get_crs_code(station_name: str) -> str | None:
    """
    Look up the 3-letter CRS code for a station name.
    Returns None if the station isn't in our database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT crs_code FROM station_codes
        WHERE LOWER(station_name) = LOWER(?)
    """, (station_name,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def find_stations_in_text(user_input: str) -> list:
    """
    Fallback: scan the sentence against the station_codes table.
    Catches stations spaCy misses.
    Returns list of (station_name, crs_code) tuples in order
    of appearance in the sentence.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT station_name, crs_code FROM station_codes")
    all_stations = cursor.fetchall()
    conn.close()

    found = []
    text_lower = user_input.lower()

    for station_name, crs_code in all_stations:
        if station_name.lower() in text_lower:
            pos = text_lower.index(station_name.lower())
            found.append((pos, station_name, crs_code))

    found.sort(key=lambda x: x[0])
    return [(name, code) for _, name, code in found]

def resolve_london(station_name: str, intent: str = None) -> tuple:
    """
    'London' is ambiguous — map it to the most likely station
    based on context. Default to London Waterloo for delay
    prediction (SWR route) and London Liverpool Street for
    ticket searches (Greater Anglia route from Norwich).
    """
    if station_name.lower() == "london":
        if intent == INTENT_PREDICT_DELAY:
            return ("London Waterloo", "WAT")
        else:
            return ("London Liverpool Street", "LST")
    return (station_name, None)
# ─── Main entity extractor ───────────────────────────────
def extract_entities(user_input: str, intent: str = None) -> dict:
    """
    Extract named entities from user input.

    Returns a dictionary with:
    - origin          : station name or None
    - destination     : station name or None
    - origin_crs      : CRS code or None
    - destination_crs : CRS code or None
    - date            : date string or None
    - time            : time string or None
    - missing         : list of required fields that are missing
    """
    doc = nlp(user_input)

    result = {
        "origin": None,
        "destination": None,
        "origin_crs": None,
        "destination_crs": None,
        "date": None,
        "time": None,
        "missing": []
    }

    # ─── Extract locations (GPE first, then DB fallback) ──
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    if len(locations) >= 1:
        result["origin"] = locations[0]
        result["origin_crs"] = get_crs_code(locations[0])
    if len(locations) >= 2:
        result["destination"] = locations[1]
        result["destination_crs"] = get_crs_code(locations[1])
        
# ─── Resolve ambiguous "London" ───────────────────────
    if result["origin"] and result["origin"].lower() == "london":
        result["origin"], result["origin_crs"] = resolve_london(
            result["origin"], intent
        )
    if result["destination"] and result["destination"].lower() == "london":
        result["destination"], result["destination_crs"] = resolve_london(
            result["destination"], intent
        )
    # DB fallback for anything spaCy missed
    db_stations = find_stations_in_text(user_input)

    if db_stations and result["origin"] is None:
        result["origin"] = db_stations[0][0]
        result["origin_crs"] = db_stations[0][1]

    if len(db_stations) >= 2 and result["destination"] is None:
        result["destination"] = db_stations[1][0]
        result["destination_crs"] = db_stations[1][1]

    # Fix CRS code if spaCy found name but DB lookup returned None
    if result["origin"] and result["origin_crs"] is None:
        result["origin_crs"] = get_crs_code(result["origin"])

    if result["destination"] and result["destination_crs"] is None:
        result["destination_crs"] = get_crs_code(result["destination"])

    # ─── Extract date and time ────────────────────────────
    for ent in doc.ents:
        if ent.label_ == "DATE" and not result["date"]:
            result["date"] = ent.text
        if ent.label_ == "TIME" and not result["time"]:
            result["time"] = ent.text

    # ─── Work out what's missing ──────────────────────────
    if intent == INTENT_BOOK_TICKET:
        required = ["origin", "destination", "date"]
    elif intent == INTENT_PREDICT_DELAY:
        required = ["origin", "destination"]
    else:
        required = ["origin", "destination"]

    for field in required:
        if result[field] is None:
            result["missing"].append(field)

    return result


# ─── Manual test ─────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("I want to travel from Norwich to London on Tuesday",
         INTENT_BOOK_TICKET),
        ("I want to go from Norwich to Oxford",
         INTENT_BOOK_TICKET),
        ("I want to travel from Norwich on the 15th of July",
         INTENT_BOOK_TICKET),
        ("My train from Weymouth is delayed at Southampton",
         INTENT_PREDICT_DELAY),
        ("hello there",
         INTENT_BOOK_TICKET),
        ("I want to go from Norwich to Peterborough on Friday",
         INTENT_BOOK_TICKET),
    ]

    for sentence, intent in tests:
        result = extract_entities(sentence, intent)
        print(f"Input:   '{sentence}'")
        print(f"Intent:  {intent}")
        print(f"Origin:  {result['origin']} ({result['origin_crs']})")
        print(f"Dest:    {result['destination']} ({result['destination_crs']})")
        print(f"Date:    {result['date']}")
        print(f"Time:    {result['time']}")
        print(f"Missing: {result['missing']}")
        print()