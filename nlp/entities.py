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
    Scan user input against station_codes table.
    Priority:
      1. Exact match (case-insensitive) — e.g. "Norwich" → only Norwich (NRW)
      2. Station name contained in text — e.g. full sentence with station name
      3. Text contained in station name — e.g. "Southampton" → Southampton Central
    Returns list of (station_name, crs_code).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT station_name, crs_code FROM station_codes")
    all_stations = cursor.fetchall()
    conn.close()

    text_lower = user_input.strip().lower()

    # Pass 1: exact match
    for station_name, crs_code in all_stations:
        if not _is_likely_rail_station(station_name):
            continue
        if station_name.lower() == text_lower:
            return [(station_name, crs_code)]

    # Pass 2: station name is a substring of the user's text (sentence input)
    # Use word-boundary match to prevent "Iver" matching inside "liverpool" etc.
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

    # Pass 3: user's text is a prefix/substring of a station name
    # e.g. "Southampton" → "Southampton Central"
    if len(text_lower) >= 4:
        partial = []
        for station_name, crs_code in all_stations:
            if not _is_likely_rail_station(station_name):
                continue
            if text_lower in station_name.lower() and station_name.lower() != text_lower:
                partial.append((station_name, crs_code))
        return partial

    return []


# Keywords that indicate non-rail stops (trams, buses, arenas, etc.)
_NON_RAIL_KEYWORDS = (
    "(bus)", "arena", "docks", "ferry", "(tramlink)",
    "tram", "metro", "underground", "tube"
)


def _is_likely_rail_station(name: str) -> bool:
    """Return True if the station name looks like an actual National Rail stop."""
    nl = name.lower()
    return not any(kw in nl for kw in _NON_RAIL_KEYWORDS)


def find_stations_fuzzy(query: str, limit: int = 3) -> list:
    """
    Return up to `limit` National Rail station matches for an ambiguous query.
    Filters out bus stops, arenas, tram stops etc.
    Priority:
      1. Starts-with match (e.g. "London" → "London Liverpool Street" first)
      2. Contains match   (e.g. "Street" → "London Liverpool Street")
    Returns list of (station_name, crs_code).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT station_name, crs_code FROM station_codes")
    all_stations = cursor.fetchall()
    conn.close()

    q = query.lower().strip()

    # Check exact match first
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
            # Only include if query appears as a word boundary, not mid-word
            # e.g. "london" matches "London Liverpool Street" but not "Caldon Low"
            import re as _re
            if _re.search(r'\b' + _re.escape(q), nl):
                contains.append((name, crs))

    results = starts + contains
    return results[:limit]


def find_station_by_typo(query: str) -> tuple:
    """
    Use difflib to find the closest station name for a misspelled query.
    Returns (station_name, crs_code) if a close match is found, else (None, None).
    e.g. "norich" → ("Norwich", "NRW")
    """
    import difflib
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT station_name, crs_code FROM station_codes")
    all_stations = cursor.fetchall()
    conn.close()

    station_names = [s[0] for s in all_stations]
    name_to_crs = {s[0]: s[1] for s in all_stations}

    # Try matching against full names and also first words of multi-word names
    matches = difflib.get_close_matches(
        query.lower(),
        [n.lower() for n in station_names],
        n=1,
        cutoff=0.6
    )
    if matches:
        # Find the original-case name
        for name in station_names:
            if name.lower() == matches[0]:
                return name, name_to_crs[name]

    # Also try first word only (e.g. "norich" against "Norwich")
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
    'London' is ambiguous — map it to the most likely station
    based on context. Also handles common Waterloo/Liverpool St
    variants whose names are reversed in the station DB.
    """
    sl = station_name.strip().lower()

    # Explicit Waterloo variants
    if sl in ("london waterloo", "waterloo", "london waterloo station",
              "waterloo london", "wat"):
        return ("London Waterloo", "WAT")

    # Explicit Liverpool Street variants
    if sl in ("london liverpool street", "liverpool street",
              "liverpool st", "london liverpool st", "lst"):
        return ("London Liverpool Street", "LST")

    # Bare "london" — route by intent
    if sl == "london":
        if intent == INTENT_PREDICT_DELAY:
            return ("Waterloo London", "WAT")
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