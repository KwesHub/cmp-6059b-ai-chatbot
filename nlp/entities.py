import spacy
from config import SPACY_MODEL, INTENT_BOOK_TICKET, INTENT_PREDICT_DELAY
from database import get_connection

# load the pre-trained English spaCy model once at import time
nlp = spacy.load(SPACY_MODEL)


def get_crs_code(station_name: str) -> str | None:
    """Look up the 3-letter CRS code for an exact station name."""
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
    Scan user input against the station_codes table using three passes.
    Priority: exact match > station name inside text > text inside station name.
    Returns list of (station_name, crs_code).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT station_name, crs_code FROM station_codes")
    all_stations = cursor.fetchall()
    conn.close()

    text_lower = user_input.strip().lower()

    # Pass 1: exact match -- fastest and most precise
    for station_name, crs_code in all_stations:
        if not _is_likely_rail_station(station_name):
            continue
        if station_name.lower() == text_lower:
            return [(station_name, crs_code)]

    # Pass 2: station name appears as a whole word inside the sentence.
    # The word-boundary regex (\b) prevents "Iver" matching inside "Liverpool".
    import re as _re
    found = []
    for station_name, crs_code in all_stations:
        if not _is_likely_rail_station(station_name):
            continue
        station_lower = station_name.lower()
        if station_lower != text_lower and _re.search(
            r'\b' + _re.escape(station_lower) + r'\b', text_lower
        ):
            pos = text_lower.index(station_lower)
            found.append((pos, station_name, crs_code))

    if found:
        found.sort(key=lambda x: x[0])
        return [(name, code) for _, name, code in found]

    # Pass 3: user typed a partial name (e.g. "Southampton" matches "Southampton Central")
    if len(text_lower) >= 4:
        partial = []
        for station_name, crs_code in all_stations:
            if not _is_likely_rail_station(station_name):
                continue
            if text_lower in station_name.lower() and station_name.lower() != text_lower:
                partial.append((station_name, crs_code))
        return partial

    return []


# these keywords identify non-rail stops that share names with real stations
_NON_RAIL_KEYWORDS = (
    "(bus)", "arena", "docks", "ferry", "(tramlink)",
    "tram", "metro", "underground", "tube"
)


def _is_likely_rail_station(name: str) -> bool:
    """Return True if the station name looks like a National Rail stop."""
    nl = name.lower()
    return not any(kw in nl for kw in _NON_RAIL_KEYWORDS)


def find_stations_fuzzy(query: str, limit: int = 3) -> list:
    """
    Return up to `limit` station matches for an ambiguous query.
    Starts-with matches are ranked before contains matches.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT station_name, crs_code FROM station_codes")
    all_stations = cursor.fetchall()
    conn.close()

    q = query.lower().strip()

    # exact match takes priority
    for name, crs in all_stations:
        if name.lower() == q and _is_likely_rail_station(name):
            return [(name, crs)]

    starts = []
    contains = []
    for name, crs in all_stations:
        if not _is_likely_rail_station(name):
            continue
        nl = name.lower()
        if nl.startswith(q):
            starts.append((name, crs))
        elif q in nl:
            # only include if the query appears at a word boundary, not mid-word
            import re as _re
            if _re.search(r'\b' + _re.escape(q), nl):
                contains.append((name, crs))

    results = starts + contains
    return results[:limit]


def find_station_by_typo(query: str) -> tuple:
    """
    Use difflib to find the closest station name for a misspelled query.
    e.g. "norich" -> ("Norwich", "NRW")
    Returns (station_name, crs_code) or (None, None) if no close match found.
    """
    import difflib
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT station_name, crs_code FROM station_codes")
    all_stations = cursor.fetchall()
    conn.close()

    station_names = [s[0] for s in all_stations]
    name_to_crs = {s[0]: s[1] for s in all_stations}

    # try matching the full query against all station names
    matches = difflib.get_close_matches(
        query.lower(),
        [n.lower() for n in station_names],
        n=1,
        cutoff=0.6
    )
    if matches:
        for name in station_names:
            if name.lower() == matches[0]:
                return name, name_to_crs[name]

    # also try matching just the first word -- helps with single-word city inputs
    first_words = {}
    for name in station_names:
        fw = name.split()[0].lower()
        if fw not in first_words:
            first_words[fw] = (name, name_to_crs[name])

    fw_matches = difflib.get_close_matches(
        query.lower(), list(first_words.keys()), n=1, cutoff=0.6
    )
    if fw_matches:
        return first_words[fw_matches[0]]

    return None, None


def resolve_london(station_name: str, intent: str = None) -> tuple:
    """
    "London" is ambiguous -- route it to the correct terminus based on context.
    For delay prediction the route is Weymouth -> London Waterloo, so we default
    to Waterloo. For ticket booking we default to Liverpool Street.
    """
    sl = station_name.strip().lower()

    # explicit Waterloo variants
    if sl in ("london waterloo", "waterloo", "london waterloo station",
              "waterloo london", "wat"):
        return ("London Waterloo", "WAT")

    # explicit Liverpool Street variants
    if sl in ("london liverpool street", "liverpool street",
              "liverpool st", "london liverpool st", "lst"):
        return ("London Liverpool Street", "LST")

    # bare "london" -- pick terminus by intent
    if sl == "london":
        if intent == INTENT_PREDICT_DELAY:
            return ("Waterloo London", "WAT")
        else:
            return ("London Liverpool Street", "LST")

    return (station_name, None)


def extract_entities(user_input: str, intent: str = None) -> dict:
    """
    Extract origin, destination, date and time from user input.
    spaCy handles GPE (geopolitical entity) recognition; the station DB is used
    as a fallback for any station names that spaCy doesn't tag as GPE.
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

    # spaCy labels place names as GPE -- take the first as origin, second as destination
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    if len(locations) >= 1:
        result["origin"] = locations[0]
        result["origin_crs"] = get_crs_code(locations[0])
    if len(locations) >= 2:
        result["destination"] = locations[1]
        result["destination_crs"] = get_crs_code(locations[1])

    # resolve "London" to the right terminus before any DB lookup
    if result["origin"] and result["origin"].lower() == "london":
        result["origin"], result["origin_crs"] = resolve_london(
            result["origin"], intent
        )
    if result["destination"] and result["destination"].lower() == "london":
        result["destination"], result["destination_crs"] = resolve_london(
            result["destination"], intent
        )

    # DB fallback for station names that spaCy didn't recognise as GPE
    db_stations = find_stations_in_text(user_input)

    if db_stations and result["origin"] is None:
        result["origin"] = db_stations[0][0]
        result["origin_crs"] = db_stations[0][1]

    if len(db_stations) >= 2 and result["destination"] is None:
        result["destination"] = db_stations[1][0]
        result["destination_crs"] = db_stations[1][1]

    # fill in any CRS codes that are still missing after name resolution
    if result["origin"] and result["origin_crs"] is None:
        result["origin_crs"] = get_crs_code(result["origin"])

    if result["destination"] and result["destination_crs"] is None:
        result["destination_crs"] = get_crs_code(result["destination"])

    # extract date and time from spaCy DATE/TIME entities
    for ent in doc.ents:
        if ent.label_ == "DATE" and not result["date"]:
            result["date"] = ent.text
        if ent.label_ == "TIME" and not result["time"]:
            result["time"] = ent.text

    # build the list of required fields that are still missing
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


if __name__ == "__main__":
    tests = [
        ("I want to travel from Norwich to London on Tuesday", INTENT_BOOK_TICKET),
        ("I want to go from Norwich to Oxford", INTENT_BOOK_TICKET),
        ("I want to travel from Norwich on the 15th of July", INTENT_BOOK_TICKET),
        ("My train from Weymouth is delayed at Southampton", INTENT_PREDICT_DELAY),
        ("hello there", INTENT_BOOK_TICKET),
        ("I want to go from Norwich to Peterborough on Friday", INTENT_BOOK_TICKET),
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
