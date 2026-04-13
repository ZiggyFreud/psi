import json
import re

with open("dealers_and_reps.json", "r") as f:
    _data = json.load(f)

_entries = {entry["id"]: entry for entry in _data["entries"]}
_state_index = _data["state_to_rep_index"]

REP_INTENT_KEYWORDS = [
    "rep", "representative", "dealer", "contact", "who covers",
    "who is", "who's", "sales", "territory", "region", "area",
    "covers", "covering", "find", "locate", "get in touch"
]

# Short keywords that need whole-word matching to avoid false positives
# e.g. "rep" matches "recommend", "find" matches "finders", "area" matches "area" (ok but be safe)
WHOLE_WORD_KEYWORDS = {"rep", "find", "area", "sales", "locate", "covers", "covering"}

_state_aliases = {
    "new mexico": "New Mexico", "nm": "New Mexico",
    "south carolina": "South Carolina", "sc": "South Carolina",
    "north carolina": "North Carolina", "nc": "North Carolina",
    "new york": "New York", "ny": "New York",
    "new jersey": "New Jersey", "nj": "New Jersey",
    "new hampshire": "New Hampshire", "nh": "New Hampshire",
    "west virginia": "West Virginia", "wv": "West Virginia",
    "north dakota": "North Dakota", "nd": "North Dakota",
    "south dakota": "South Dakota", "sd": "South Dakota",
    "rhode island": "Rhode Island", "ri": "Rhode Island",
    "connecticut": "Connecticut", "ct": "Connecticut",
    "massachusetts": "Massachusetts", "ma": "Massachusetts",
    "michigan upper peninsula": "Michigan Upper Peninsula",
    "michigan": "Michigan", "mi": "Michigan",
    "minnesota": "Minnesota", "mn": "Minnesota",
    "mississippi": "Mississippi", "ms": "Mississippi",
    "missouri": "Missouri", "mo": "Missouri",
    "montana": "Montana", "mt": "Montana",
    "nebraska": "Nebraska", "ne": "Nebraska",
    "nevada": "Nevada", "nv": "Nevada",
    "ohio": "Ohio", "oh": "Ohio",
    "oklahoma": "Oklahoma", "ok": "Oklahoma",
    "oregon": "Oregon", "or": "Oregon",
    "pennsylvania": "Pennsylvania", "pa": "Pennsylvania",
    "tennessee": "Tennessee", "tn": "Tennessee",
    "texas": "Texas", "tx": "Texas",
    "utah": "Utah", "ut": "Utah",
    "vermont": "Vermont", "vt": "Vermont",
    "virginia": "Virginia", "va": "Virginia",
    "washington": "Washington", "wa": "Washington",
    "wisconsin": "Wisconsin", "wi": "Wisconsin",
    "wyoming": "Wyoming", "wy": "Wyoming",
    "alabama": "Alabama", "al": "Alabama",
    "alaska": "Alaska", "ak": "Alaska",
    "arizona": "Arizona", "az": "Arizona",
    "arkansas": "Arkansas", "ar": "Arkansas",
    "california": "California", "ca": "California",
    "colorado": "Colorado", "co": "Colorado",
    "delaware": "Delaware", "de": "Delaware",
    "florida": "Florida", "fl": "Florida",
    "georgia": "Georgia", "ga": "Georgia",
    "hawaii": "Hawaii", "hi": "Hawaii",
    "idaho": "Idaho", "id": "Idaho",
    "illinois": "Illinois", "il": "Illinois",
    "indiana": "Indiana", "in": "Indiana",
    "iowa": "Iowa", "ia": "Iowa",
    "kansas": "Kansas", "ks": "Kansas",
    "kentucky": "Kentucky", "ky": "Kentucky",
    "louisiana": "Louisiana", "la": "Louisiana",
    "maine": "Maine", "me": "Maine",
    "maryland": "Maryland", "md": "Maryland"
}

NO_REP_RESPONSE = (
    "For assistance finding a representative in your area, "
    "please call us at 1-800-947-9422."
)

def has_rep_intent(message: str) -> bool:
    msg = message.lower()
    for keyword in REP_INTENT_KEYWORDS:
        if keyword in WHOLE_WORD_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', msg):
                return True
        else:
            if keyword in msg:
                return True
    return False

def detect_state(message: str) -> str | None:
    msg = message.lower()
    for alias in sorted(_state_aliases.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, msg):
            return _state_aliases[alias]
    return None

def format_rep(entry: dict) -> str:
    phones = " | ".join(
        f"{p['type'].capitalize()}: {p['number']}" for p in entry["phone"]
    )
    lines = [
        f"{entry['company']}",
        f"Contact: {entry['contact_name']}",
        f"Type: {entry['type'].capitalize()}",
        f"Phone: {phones}",
        f"Email: {entry['email']}",
    ]
    if entry.get("region_notes"):
        lines.append(f"Note: {entry['region_notes']}")
    return "\n".join(lines)

# Reverse map full names back to abbreviations for index lookup
_full_to_abbrev = {v: k.upper() for k, v in _state_aliases.items() if len(k) == 2}

def lookup_rep(message: str) -> str | None:
    if not has_rep_intent(message):
        return None
    state = detect_state(message)
    if not state:
        return None

    # Try full name first, then abbreviation
    rep_ids = _state_index.get(state, [])
    if not rep_ids:
        abbrev = _full_to_abbrev.get(state, "")
        rep_ids = _state_index.get(abbrev, [])

    if not rep_ids:
        return NO_REP_RESPONSE

    rep_blocks = [format_rep(_entries[rid]) for rid in rep_ids if rid in _entries]
    return "\n\n".join(rep_blocks)
