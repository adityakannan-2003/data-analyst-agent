"""Spot columns of an uploaded CSV that look like personal data: names, emails, phone numbers,
addresses, dates of birth, ID and card numbers, and so on (the rules are in evals/privacy/LABELS.md).

The hosted page uses it to warn the visitor and to keep their questions about that file out of the
usage log. Rules only, so it's instant, free, and the file goes nowhere for the check. Measured on
labelled columns in evals/privacy/.
"""

import re

import pandas as pd

KINDS = {
    "name": "names", "email": "email addresses", "phone": "phone numbers", "address": "street addresses",
    "postcode": "postcodes", "birth_date": "dates of birth", "gov_id": "ID numbers", "financial": "card or bank numbers",
    "ip_address": "IP addresses", "location": "home locations", "handle": "usernames",
    "text_with_contact": "text with emails or phone numbers",
}

EMAIL = r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+"
PHONE = (r"(?:\+\d{1,3}[\s.-]?\(?\d{1,5}\)?(?:[\s.-]?\d{2,5}){1,4}"  # +44 7700 900123, +1 555 123 4567, +91-98765...
         r"|\(\d{3}\)\s?\d{3}[\s.-]\d{4}"  # (555) 123-4567
         r"|\d{3}[.-]\d{3}[.-]\d{4}"  # 555-123-4567
         r"|0\d{2,4}\s\d{3,4}\s?\d{3,6})")  # 07700 900123, 020 7946 0123
STRONG = {  # a whole value in one of these shapes is personal on its own
    "email": EMAIL,
    "phone": PHONE,
    "ip_address": (r"(?:\d{1,3}\.){3}\d{1,3}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){7}"  # IPv4, full IPv6
                   r"|(?:[0-9a-fA-F]{1,4}:){1,6}:(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?"),  # IPv6 with ::
    "gov_id": (r"\d{3}-\d{2}-\d{4}"  # US SSN
               r"|[ABCEGHJ-PRSTW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]"  # UK National Insurance
               r"|\d{4}\s\d{4}\s\d{4}"  # Aadhaar
               r"|[A-Z]{5}\d{4}[A-Z]"),  # PAN
    "financial": r"[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?",  # IBAN; card numbers are checked with Luhn
    "postcode": r"[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}",  # UK, with the space
    "address": r"\d{1,5}[A-Za-z]?\s+(?:[\w'.-]+\s+){0,4}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Close|Way|Court|Ct"
               r"|Place|Pl|Boulevard|Blvd|Crescent|Terrace|Gardens|Square|Row|Hill|Grove|Park|View|Mews|Walk)\b.*",
    "handle": r"@\w[\w.]{1,30}",
}
STRONG = {kind: re.compile(pattern, re.IGNORECASE if kind == "address" else 0) for kind, pattern in STRONG.items()}
PHONE_IN_TEXT = re.compile(rf"(?<![\w-]){PHONE}(?![\w-])")

# Column-name words. A hint only counts if the values agree with it.
HINTS = {
    "phone": {"phone", "mobile", "tel", "telephone", "cell", "cellphone", "whatsapp", "msisdn", "fax", "contactno"},
    "address": {"address", "addr", "street", "address1", "address2", "addressline"},
    "postcode": {"postcode", "postal", "zip", "zipcode", "plz", "pincode", "postalcode"},
    "birth_date": {"dob", "birth", "birthday", "birthdate", "dateofbirth", "born"},
    "gov_id": {"ssn", "socialsecurity", "nationalinsurance", "nino", "ninumber", "passport", "aadhaar", "aadhar", "pan",
               "taxid", "tin", "licence", "license", "nationalid"},
    "financial": {"iban", "creditcard", "cardnumber", "ccnum", "accountnumber", "accountno", "acct", "sortcode", "bankaccount"},
    "handle": {"handle", "username", "screenname", "twitter", "instagram", "tiktok", "user", "nickname"},
}
PERSON = {"first", "last", "firstname", "lastname", "fname", "lname", "surname", "forename", "givenname", "fullname",
          "customer", "client", "employee", "manager", "staff", "patient", "student", "attendee", "member", "donor",
          "tenant", "landlord", "owner", "recipient", "contact", "passenger", "guest", "assignee", "reporter", "rep",
          "buyer", "seller", "teacher", "driver", "candidate", "applicant", "person", "worker", "agent", "volunteer"}
GENERIC_NAME = {"name", "names", "nombre", "nom", "yourname"}
LATLON = {"lat", "latitude", "lon", "lng", "longitude"}
HOME = {"home", "residence", "residential", "house"}
BUSINESS = {"store", "shop", "office", "branch", "warehouse", "server", "company", "business", "supplier", "vendor",
            "merchant", "site", "depot", "venue", "hq", "headquarters", "product", "item"}
# Common given names, for spotting names without a helpful column name. A short list on purpose:
# shortcut: misses rarer names, so a column of them is only caught by its column name; a full name list would fix it.
FIRST_NAMES = set("""james john robert michael william david richard joseph thomas charles christopher daniel matthew
anthony mark donald steven paul andrew joshua kenneth kevin brian george timothy ronald edward jason jeffrey ryan jacob
gary nicholas eric jonathan stephen larry justin scott brandon benjamin samuel gregory alexander frank patrick raymond
jack dennis jerry tyler aaron jose adam nathan henry douglas zachary peter kyle noah ethan jeremy walter christian keith
roger terry austin sean gerald carl harold dylan arthur lawrence jordan jesse bryan billy bruce gabriel joe logan alan
juan albert willie elijah wayne randy vincent mason roy ralph bobby russell bradley philip eugene mary patricia jennifer
linda elizabeth barbara susan jessica sarah karen lisa nancy betty sandra margaret ashley kimberly emily donna michelle
carol amanda melissa deborah stephanie dorothy rebecca sharon laura cynthia amy kathleen angela shirley brenda emma anna
pamela nicole samantha katherine christine helen debra rachel carolyn janet maria catherine heather diane olivia julie
joyce victoria ruth virginia lauren kelly christina joan evelyn judith andrea hannah megan cheryl jacqueline martha
madison teresa gloria sara janice ann kathryn abigail sophia frances jean alice judy isabella julia grace amber denise
danielle marilyn beverly charlotte natalie theresa diana brittany doris kayla alexis lori marie oliver harry george
charlie leo freddie alfie archie theo oscar arthur muhammad mohammed ahmed ali omar hassan fatima aisha amelia isla ava
mia ivy lily sophie poppy freya florence evie ella rosie chloe zoe priya rahul amit raj ravi anil sunil vijay arjun
aditya rohan vikram neha pooja anjali divya kavya sneha ananya deepak suresh ramesh krishna lakshmi carlos luis miguel
jorge pedro javier diego alejandro pablo sofia lucia valentina camila mariana daniela gabriela wei li ming yan chen
mei hiroshi yuki kenji akira chidi emeka ngozi chinedu oluwaseun tunde kwame kofi ama yaw ivan olga dmitri natasha
anna katarzyna piotr tomasz agnieszka luca giulia marco francesca matteo hans klaus anke jan lars sven ingrid""".split())
TITLES = {"mr", "mrs", "miss", "ms", "dr", "master", "mx", "prof", "sir"}


def name_words(column: str) -> set[str]:
    """Words of a column name, plus the whole name squashed: "Date of Birth" -> {date, of, birth, dateofbirth}."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(column)).lower()
    words = [w for w in re.split(r"[^a-z0-9]+", spaced) if w]
    return set(words) | {"".join(words)} | {"".join(words[i:i + 2]) for i in range(len(words) - 1)}


def share(values: list[str], test) -> float:
    return sum(map(test, values)) / len(values)


def phone_digits_ok(text: str) -> bool:
    return 9 <= len(re.sub(r"\D", "", text)) <= 15  # so "+0000" in a timestamp isn't a phone number


def has_contact(value: str) -> bool:
    return bool(re.search(EMAIL, value)) or any(phone_digits_ok(m.group()) for m in PHONE_IN_TEXT.finditer(value))


def luhn_ok(value: str) -> bool:
    number = re.sub(r"[\s-]", "", value)
    if not (number.isdigit() and 15 <= len(number) <= 19):  # cards are 15 to 19 digits; 13-digit barcodes aren't cards
        return False
    total = sum(int(d) if i % 2 == 0 else (int(d) * 2 - 9 if int(d) > 4 else int(d) * 2) for i, d in enumerate(reversed(number)))
    return total % 10 == 0


def looks_like_name(value: str) -> bool:
    words = [w.strip(".,'’-") for w in value.lower().split()]
    return 1 <= len(words) <= 5 and any(w in FIRST_NAMES or w in TITLES for w in words)


def kind_of(column: str, values: list[str]) -> str | None:
    """The kind of personal data a column holds, or None. `values` are non-empty strings."""
    if not values:
        return None
    words = name_words(column)
    business = bool(words & BUSINESS)

    for kind, pattern in STRONG.items():
        if share(values, lambda v: bool(pattern.fullmatch(v)) and (kind != "phone" or phone_digits_ok(v))) >= 0.5:
            return None if business and kind in ("phone", "ip_address", "postcode", "address") else kind
    if share(values, luhn_ok) >= 0.5:
        return "financial"
    if business:
        return None
    if words & {"id", "uuid", "hash", "hashed", "key", "token"} and not words & HINTS["gov_id"]:
        return None  # a pseudonymous id: out of scope (LABELS.md)
    if any(map(has_contact, values)):
        return "text_with_contact"

    digits = [re.sub(r"\D", "", v) for v in values]
    has_letters = share(values, lambda v: bool(re.search(r"[A-Za-z]", v)))
    distinct = len(set(values))
    if words & HINTS["phone"] and share(digits, lambda d: 7 <= len(d) <= 15) >= 0.5 and has_letters < 0.2:
        return "phone"
    if words & HINTS["address"] and share(values, lambda v: bool(re.search(r"\d", v) and re.search(r"[A-Za-z]{3}", v))) >= 0.5:
        return "address"
    if words & HINTS["postcode"] and share(values, lambda v: bool(re.fullmatch(r"\d{4,6}(-\d{4})?|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}", v.strip().upper()))) >= 0.5:
        return "postcode"
    if words & HINTS["birth_date"] and pd.to_datetime(pd.Series(values), errors="coerce", format="mixed").notna().mean() >= 0.5:
        return "birth_date"
    if words & HINTS["gov_id"] and share(digits, lambda d: len(d) >= 6) >= 0.5:
        return "gov_id"
    if words & HINTS["financial"] and share(digits, lambda d: len(d) >= 6) >= 0.5:
        return "financial"
    if words & LATLON and words & HOME:
        return "location"
    if words & HINTS["handle"] and share(values, lambda v: bool(re.fullmatch(r"@?[A-Za-z][\w.]{2,30}", v))) >= 0.5 and distinct >= 8:
        return "handle"
    alphabetic = share(values, lambda v: bool(re.fullmatch(r"[^\W\d_][^\d@]*", v.strip()))) >= 0.8
    name_share = share(values, looks_like_name)
    # distinct >= 8 keeps out categories ("email", "phone", "sms"); a small file of real names gets in by its names
    if alphabetic and (distinct >= 8 or name_share >= 0.5):
        if words & PERSON:
            return "name"
        if name_share >= (0.3 if words & GENERIC_NAME else 0.5):
            return "name"
    return None


def personal_columns(df: pd.DataFrame, rows: int = 500) -> dict[str, str]:
    """{column: kind} for the columns of df that look like personal data, judged on the first `rows` rows.
    shortcut: a column whose personal values only start further down is missed; sample more rows if that matters."""
    found = {}
    for column in df.columns:
        values = [str(v).strip() for v in df[column].head(rows).dropna() if str(v).strip()]
        kind = kind_of(column, values)
        if kind:
            found[column] = kind
    return found


def describe(found: dict[str, str]) -> str:
    """'Customer (names), email (email addresses)' for the warning on the page."""
    return ", ".join(f"**{column}** ({KINDS[kind]})" for column, kind in found.items())
