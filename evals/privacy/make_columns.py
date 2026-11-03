"""Build the privacy filter's labelled set: about 200 columns, each a name and 30 example values, labelled
by the rules in LABELS.md. Names follow real exports (Shopify, Mailchimp, Stripe, Google Forms, HR, CRM);
the values are made up, apart from every column of the three sample datasets, which are real. Look-alikes
are planted on purpose: product "Name" columns, store postcodes, opt-in flags, 13-digit barcodes.

    uv run python evals/privacy/make_columns.py

Writes columns.jsonl (seeded, so it rebuilds the same) and spotcheck.md. 40% of columns are the dev
split, used to build the filter; the other 60% are the test split it's measured on.
"""

import hashlib
import json
import random
import string
import uuid
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
N = 30

FIRST = """Aaliyah Aarav Abigail Adebayo Aditi Ahmed Aiko Alejandro Alessia Amara Amelia Ananya Andrei Anika Arjun
Astrid Beatriz Benjamin Bilal Bronwen Callum Camila Chen Chiara Chloe Connor Dariusz Deepika Diego Dmitri Eilidh
Elena Eleanor Emeka Emily Ethan Fatima Felix Finn Freya Gabriel Grace Hamza Hannah Haruto Hiroshi Ibrahim Imogen
Ines Isla Ivan Jack Jasmine Javier Jessica Joao Joseph Julia Kai Karan Katarzyna Kemi Kofi Lakshmi Laura Leila Liam
Lucia Luca Mai Malik Maria Mateo Matilda Mehmet Mia Michael Mohammed Nadia Naledi Niamh Noah Nkechi Olga Oliver
Olivia Omar Oscar Pablo Priya Rafael Rahul Rania Rhys Rosa Rohan Saanvi Samuel Santiago Sara Shreya Siobhan Sofia
Sophie Tariq Thabo Theo Tomasz Valentina Vikram Wei Wiktoria Xin Yara Yusuf Zainab Zara Zoe""".split()
LAST = """Abbott Adeyemi Agarwal Ahmed Ali Andersen Bailey Banerjee Becker Bianchi Brown Campbell Chatterjee Chen
Clarke Costa Davies Delgado Dubois Edwards Evans Fernandes Fischer Garcia Gonzalez Gupta Hall Hansen Hughes Ivanova
Iyer Jackson Jensen Johnson Jones Kaur Kelly Khan Kim Kowalski Kumar Lee Lewis Lopez Martin Mensah Murphy Nakamura
Nguyen Novak Obi O'Brien Okafor Patel Pereira Petrov Reddy Ricci Roberts Rossi Sanchez Santos Schmidt Shah Silva
Singh Smith Sullivan Suzuki Tanaka Taylor Thomas Thompson Walker Wang Watson Williams Wilson Wright Yilmaz Zhang""".split()
STREETS = ["High Street", "Station Road", "Church Lane", "Maple Avenue", "Oak Street", "Park Road", "Elm Drive",
           "Victoria Road", "Mill Lane", "Cedar Court", "Kings Way", "Willow Close", "Main St", "Washington Ave",
           "Lake Shore Dr", "2nd Street", "Sunset Blvd", "Brigade Road", "Queens Crescent", "Harbour View"]
CITIES = ["London", "Manchester", "Leeds", "Bristol", "Glasgow", "New York", "Chicago", "Austin", "Seattle", "Mumbai",
          "Bengaluru", "Pune", "Paris", "Berlin", "Madrid", "Toronto", "Sydney", "Lagos", "Nairobi", "Dublin"]
DOMAINS = ["gmail.com", "yahoo.co.uk", "outlook.com", "hotmail.com", "icloud.com", "proton.me", "btinternet.com"]
PRODUCTS = ["Iced Latte", "Cold Brew", "Flat White", "Blue Ceramic Mug", "Paper Chain Kit", "Jumbo Bag Red Retrospot",
            "Wireless Mouse", "Yoga Mat", "Garden Hose 15m", "Silver Necklace", "Running Shoes", "Desk Lamp",
            "Green Tea 50 Bags", "Notebook A5", "Phone Case", "USB-C Cable", "Herb Planter", "Scented Candle"]
COMPANIES = ["Northwind Traders", "Hartley & Sons Ltd", "Acme Corp", "Blue Fin Logistics", "Greenleaf Foods",
             "Initech", "Kestrel Analytics", "Oakridge Partners", "Pioneer Freight", "Summit Health Group"]


def pick(r, xs):
    return r.choice(xs)


def full_name(r):
    return f"{pick(r, FIRST)} {pick(r, LAST)}"


def email(r):
    first, last = pick(r, FIRST).lower(), pick(r, LAST).lower().replace("'", "")
    local = r.choice([f"{first}.{last}", f"{first[0]}{last}", f"{first}{r.randint(1, 99)}", f"{last}.{first[0]}"])
    return f"{local}@{pick(r, DOMAINS)}"


def work_email(r):
    return f"{pick(r, FIRST).lower()}.{pick(r, LAST).lower().replace(chr(39), '')}@{pick(r, ['northwind.com', 'acme.co.uk', 'kestrel.io'])}"


def digits(r, n):
    return "".join(r.choice(string.digits) for _ in range(n))


def phone(r, style=None):
    style = style or r.choice(["us", "uk", "in"])
    if style == "us":
        a, b, c = digits(r, 3), digits(r, 3), digits(r, 4)
        return r.choice([f"({a}) {b}-{c}", f"{a}-{b}-{c}", f"+1 {a} {b} {c}", f"{a}.{b}.{c}"])
    if style == "uk":
        return r.choice([f"07{digits(r, 3)} {digits(r, 6)}", f"+44 7{digits(r, 3)} {digits(r, 6)}", f"020 {digits(r, 4)} {digits(r, 4)}"])
    return r.choice([f"+91 9{digits(r, 4)} {digits(r, 5)}", f"9{digits(r, 9)}", f"+91-9{digits(r, 9)}"])


def street(r):
    return f"{r.randint(1, 250)} {pick(r, STREETS)}" + r.choice(["", "", f", Flat {r.randint(1, 20)}", f" Apt {r.randint(1, 40)}"])


def uk_postcode(r):
    area = r.choice(["SW", "E", "N", "M", "LS", "BS", "G", "EH", "B", "CF", "NW", "SE"])
    return f"{area}{r.randint(1, 20)} {r.randint(1, 9)}{r.choice('ABDEFGHJLNPQRSTUWXYZ')}{r.choice('ABDEFGHJLNPQRSTUWXYZ')}"


def birth_date(r, fmt=None):
    d = pd.Timestamp("1940-01-01") + pd.Timedelta(days=r.randint(0, 24000))
    return d.strftime(fmt or r.choice(["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]))


def event_date(r, fmt="%Y-%m-%d"):
    return (pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=r.randint(0, 600000))).strftime(fmt)


def luhn(prefix: str, length: int, r) -> str:
    body = prefix + digits(r, length - len(prefix) - 1)
    total = 0
    for i, d in enumerate(reversed(body)):
        n = int(d) * (2 if i % 2 == 0 else 1)
        total += n - 9 if n > 9 else n
    return body + str((10 - total % 10) % 10)


def card(r):
    number = luhn(r.choice(["4", "51", "37"]), 16, r)
    return r.choice([number, " ".join(number[i:i + 4] for i in range(0, 16, 4)), "-".join(number[i:i + 4] for i in range(0, 16, 4))])


def ean13(r):
    body = "50" + digits(r, 10)
    check = (10 - sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(body)) % 10) % 10
    return body + str(check)


def ipv4(r, private=False):
    return f"10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}" if private else \
        f"{r.randint(23, 223)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"


def note_with_contact(r):
    return r.choice([f"Please call me on {phone(r, 'uk')}", f"email me at {email(r)}", "Leave at the back door",
                     "No preference", f"Reach me on {phone(r, 'us')} after 5pm", "Gift wrap please", "",
                     "Allergic to nuts", f"cc {email(r)} on the invoice", "Ring the bell twice"])


def ni_number(r):
    letters = "ABCEGHJKLMNPRSTWXYZ"
    return f"{r.choice(letters)}{r.choice(letters)}{digits(r, 6)}{r.choice('ABCD')}"


P = True  # personal
SPECS = [
    # (source, column name, value generator, personal, kind)
    # --- names
    ("shopify-orders", "Billing Name", full_name, P, "name"),
    ("shopify-orders", "Shipping Name", full_name, P, "name"),
    ("crm", "Full Name", full_name, P, "name"),
    ("crm", "full_name", full_name, P, "name"),
    ("crm", "Contact", full_name, P, "name"),
    ("mailchimp", "FNAME", lambda r: pick(r, FIRST), P, "name"),
    ("mailchimp", "LNAME", lambda r: pick(r, LAST), P, "name"),
    ("hr", "first_name", lambda r: pick(r, FIRST), P, "name"),
    ("hr", "last_name", lambda r: pick(r, LAST), P, "name"),
    ("hr", "Employee", full_name, P, "name"),
    ("hr", "Manager", full_name, P, "name"),
    ("hr", "EMPLOYEE_NAME", lambda r: full_name(r).upper(), P, "name"),
    ("titanic", "Name", lambda r: f"{pick(r, LAST)}, {r.choice(['Mr.', 'Mrs.', 'Miss.', 'Master.'])} {pick(r, FIRST)}", P, "name"),
    ("clinic", "Patient", lambda r: f"{pick(r, LAST)}, {pick(r, FIRST)}", P, "name"),
    ("school", "Student Name", full_name, P, "name"),
    ("events", "Attendee", full_name, P, "name"),
    ("google-forms", "Your name", full_name, P, "name"),
    ("google-forms", "Name", full_name, P, "name"),
    ("sales", "Customer", full_name, P, "name"),
    ("sales", "sales_rep", full_name, P, "name"),
    ("export", "col_2", full_name, P, "name"),
    ("export", "Unnamed: 1", full_name, P, "name"),
    ("spanish-crm", "Nombre", full_name, P, "name"),
    ("lettings", "Tenant", full_name, P, "name"),
    ("lettings", "Landlord", full_name, P, "name"),
    ("support", "Assignee", lambda r: pick(r, FIRST), P, "name"),
    ("donations", "Donor", full_name, P, "name"),
    ("gym", "member", full_name, P, "name"),
    # --- emails
    ("mailchimp", "Email Address", email, P, "email"),
    ("shopify-orders", "Email", email, P, "email"),
    ("crm", "email", email, P, "email"),
    ("crm", "Contact Email", work_email, P, "email"),
    ("stripe", "Customer Email", email, P, "email"),
    ("app", "user_email", email, P, "email"),
    ("app", "login", email, P, "email"),
    ("export", "col_4", email, P, "email"),
    ("export", "Unnamed: 3", email, P, "email"),
    ("spanish-crm", "Correo", email, P, "email"),
    ("hr", "E-mail", work_email, P, "email"),
    ("events", "EMAIL", lambda r: email(r).upper(), P, "email"),
    ("support", "Reporter", email, P, "email"),
    ("app", "username", email, P, "email"),
    # --- phones
    ("shopify-orders", "Phone", lambda r: phone(r), P, "phone"),
    ("shopify-orders", "Billing Phone", lambda r: phone(r, "us"), P, "phone"),
    ("crm", "Mobile", lambda r: phone(r, "uk"), P, "phone"),
    ("crm", "phone_number", lambda r: phone(r, "us"), P, "phone"),
    ("crm", "Tel", lambda r: phone(r, "uk"), P, "phone"),
    ("hr", "cell", lambda r: phone(r, "us"), P, "phone"),
    ("delivery", "contact_no", lambda r: phone(r, "in"), P, "phone"),
    ("delivery", "WhatsApp", lambda r: phone(r, "in"), P, "phone"),
    ("export", "field_9", lambda r: phone(r, "uk"), P, "phone"),
    ("export", "Unnamed: 5", lambda r: phone(r, "us"), P, "phone"),
    ("clinic", "Emergency Contact", lambda r: phone(r, "uk"), P, "phone"),
    ("app", "msisdn", lambda r: f"44{digits(r, 10)}", P, "phone"),
    # --- addresses and postcodes
    ("shopify-orders", "Billing Address1", street, P, "address"),
    ("shopify-orders", "Shipping Street", street, P, "address"),
    ("crm", "Address", lambda r: f"{street(r)}, {pick(r, CITIES)}", P, "address"),
    ("hr", "home_address", lambda r: f"{street(r)}, {pick(r, CITIES)}", P, "address"),
    ("delivery", "drop_location", street, P, "address"),
    ("export", "col_6", lambda r: f"{street(r)}, {pick(r, CITIES)}", P, "address"),
    ("lettings", "Property", street, P, "address"),
    ("shopify-orders", "Billing Zip", lambda r: digits(r, 5), P, "postcode"),
    ("shopify-orders", "Shipping Zip", lambda r: uk_postcode(r), P, "postcode"),
    ("crm", "postcode", uk_postcode, P, "postcode"),
    ("hr", "zip_code", lambda r: digits(r, 5), P, "postcode"),
    ("export", "Unnamed: 7", uk_postcode, P, "postcode"),
    ("german-crm", "PLZ", lambda r: digits(r, 5), P, "postcode"),
    # --- dates of birth
    ("hr", "dob", birth_date, P, "birth_date"),
    ("clinic", "Date of Birth", lambda r: birth_date(r, "%d/%m/%Y"), P, "birth_date"),
    ("crm", "Birthday", lambda r: birth_date(r, "%B %d, %Y"), P, "birth_date"),
    ("school", "DOB", lambda r: birth_date(r, "%Y-%m-%d"), P, "birth_date"),
    ("app", "birthdate", lambda r: birth_date(r, "%m/%d/%Y"), P, "birth_date"),
    # --- government ids
    ("hr", "ssn", lambda r: f"{digits(r, 3)}-{digits(r, 2)}-{digits(r, 4)}", P, "gov_id"),
    ("payroll", "SSN", lambda r: digits(r, 9), P, "gov_id"),
    ("payroll", "NI Number", ni_number, P, "gov_id"),
    ("hr", "national_insurance_no", lambda r: f"{ni_number(r)[:2]} {r.randint(10, 99)} {r.randint(10, 99)} {r.randint(10, 99)} {r.choice('ABCD')}", P, "gov_id"),
    ("travel", "passport_number", lambda r: r.choice(["", "K", "P"]) + digits(r, 8), P, "gov_id"),
    ("kyc", "Aadhaar", lambda r: f"{digits(r, 4)} {digits(r, 4)} {digits(r, 4)}", P, "gov_id"),
    ("kyc", "PAN", lambda r: "".join(r.choice(string.ascii_uppercase) for _ in range(5)) + digits(r, 4) + r.choice(string.ascii_uppercase), P, "gov_id"),
    ("rentals", "driving_licence", lambda r: f"{pick(r, LAST).upper()[:5]:9<5}{digits(r, 6)}{''.join(r.choice(string.ascii_uppercase) for _ in range(2))}{digits(r, 1)}", P, "gov_id"),
    ("export", "id_number", lambda r: f"{digits(r, 3)}-{digits(r, 2)}-{digits(r, 4)}", P, "gov_id"),
    # --- financial
    ("payments", "card_number", card, P, "financial"),
    ("payments", "Credit Card", card, P, "financial"),
    ("payments", "cc_num", lambda r: luhn("4", 16, r), P, "financial"),
    ("payroll", "IBAN", lambda r: f"GB{digits(r, 2)} {r.choice(['WEST', 'BARC', 'LOYD', 'HBUK'])} {digits(r, 4)} {digits(r, 4)} {digits(r, 4)} {digits(r, 2)}", P, "financial"),
    ("payroll", "account_number", lambda r: digits(r, 8), P, "financial"),
    ("export", "col_11", lambda r: luhn("5", 16, r), P, "financial"),
    # --- ip addresses, locations, handles
    ("web-logs", "client_ip", ipv4, P, "ip_address"),
    ("web-logs", "ip_address", ipv4, P, "ip_address"),
    ("web-logs", "remote_addr", lambda r: r.choice([ipv4(r), f"2a02:{r.randint(1000, 9999)}:{r.randint(100, 999)}::{r.randint(1, 99)}"]), P, "ip_address"),
    ("survey", "IP", ipv4, P, "ip_address"),
    ("delivery", "home_lat", lambda r: f"{r.uniform(51.3, 51.7):.6f}", P, "location"),
    ("delivery", "home_lon", lambda r: f"{r.uniform(-0.5, 0.2):.6f}", P, "location"),
    ("social", "twitter_handle", lambda r: f"@{pick(r, FIRST).lower()}{r.choice(['_', ''])}{pick(r, LAST).lower()[:4]}{r.randint(1, 99)}", P, "handle"),
    ("social", "Instagram", lambda r: f"@{pick(r, FIRST).lower()}.{pick(r, LAST).lower()[:5]}", P, "handle"),
    ("forum", "user", lambda r: f"{pick(r, FIRST).lower()}{pick(r, LAST).lower()[:3]}{r.randint(1, 999)}", P, "handle"),
    ("forum", "screen_name", lambda r: f"{pick(r, FIRST).lower()}_{r.randint(70, 99)}", P, "handle"),
    # --- free text with contact details
    ("shopify-orders", "Notes", note_with_contact, P, "text_with_contact"),
    ("support", "message", note_with_contact, P, "text_with_contact"),
    ("google-forms", "Anything else?", note_with_contact, P, "text_with_contact"),
    ("crm", "contact", lambda r: r.choice([email(r), phone(r)]), P, "email"),

    # ===== not personal
    ("products", "name", lambda r: pick(r, PRODUCTS), False, None),
    ("products", "Name", lambda r: pick(r, PRODUCTS), False, None),
    ("products", "product_name", lambda r: pick(r, PRODUCTS), False, None),
    ("sports", "Name", lambda r: pick(r, ["Manchester United", "Real Madrid", "Chicago Bulls", "Mumbai Indians", "Leeds Rhinos", "Toronto Raptors"]), False, None),
    ("suppliers", "Name", lambda r: pick(r, COMPANIES), False, None),
    ("suppliers", "company_name", lambda r: pick(r, COMPANIES), False, None),
    ("suppliers", "Company", lambda r: pick(r, COMPANIES), False, None),
    ("stores", "store_name", lambda r: f"{pick(r, CITIES)} {r.choice(['Central', 'North', 'Station', 'Mall'])}", False, None),
    ("stores", "branch", lambda r: pick(r, CITIES), False, None),
    ("stores", "city", lambda r: pick(r, CITIES), False, None),
    ("stores", "Country", lambda r: pick(r, ["United Kingdom", "France", "Germany", "India", "USA", "Spain", "Ireland"]), False, None),
    ("stores", "region", lambda r: pick(r, ["North", "South", "East", "West", "Midlands"]), False, None),
    ("stores", "State", lambda r: pick(r, ["CA", "TX", "NY", "WA", "IL", "MA"]), False, None),
    ("stores", "store_zip", lambda r: digits(r, 5), False, None),
    ("stores", "office_postcode", uk_postcode, False, None),
    ("stores", "store_lat", lambda r: f"{r.uniform(51.3, 51.7):.4f}", False, None),
    ("stores", "latitude", lambda r: f"{r.uniform(-40, 60):.2f}", False, None),
    ("stores", "Store Address", street, False, None),
    ("stores", "Store Phone", lambda r: f"020 {digits(r, 4)} {digits(r, 4)}", False, None),
    ("products", "category", lambda r: pick(r, ["Coffee", "Tea", "Food", "Merch", "Gifts", "Home"]), False, None),
    ("products", "brand", lambda r: pick(r, ["Acme", "Zenith", "Orbit", "Nova", "Apex"]), False, None),
    ("products", "sku", lambda r: f"SKU-{digits(r, 5)}", False, None),
    ("products", "barcode", ean13, False, None),
    ("products", "ISBN", lambda r: "978" + digits(r, 10), False, None),
    ("products", "price", lambda r: f"${r.uniform(1, 80):.2f}", False, None),
    ("products", "description", lambda r: f"{pick(r, PRODUCTS)}, {r.choice(['pack of 2', 'large', 'limited edition', 'new'])}", False, None),
    ("orders", "Order ID", lambda r: f"#{r.randint(1001, 9999)}", False, None),
    ("orders", "order_number", lambda r: digits(r, 10), False, None),
    ("orders", "invoice", lambda r: f"INV-2026-{r.randint(1, 9999):04d}", False, None),
    ("orders", "transaction_id", lambda r: str(uuid.UUID(int=r.getrandbits(128))), False, None),
    ("orders", "quantity", lambda r: str(r.randint(1, 12)), False, None),
    ("orders", "Total", lambda r: f"{r.uniform(5, 500):.2f}", False, None),
    ("orders", "currency", lambda r: pick(r, ["GBP", "USD", "EUR", "INR"]), False, None),
    ("orders", "order_date", event_date, False, None),
    ("orders", "Created at", lambda r: event_date(r, "%Y-%m-%d %H:%M:%S +0000"), False, None),
    ("orders", "ship_date", lambda r: event_date(r, "%d/%m/%Y"), False, None),
    ("orders", "payment_method", lambda r: pick(r, ["card", "cash", "app", "paypal"]), False, None),
    ("orders", "Card Type", lambda r: pick(r, ["Visa", "Mastercard", "Amex"]), False, None),
    ("stripe", "Card Last4", lambda r: digits(r, 4), False, None),
    ("orders", "tracking_number", lambda r: f"1Z{digits(r, 3)}AA{digits(r, 10)}", False, None),
    ("orders", "status", lambda r: pick(r, ["paid", "refunded", "pending", "shipped"]), False, None),
    ("orders", "Fulfillment Status", lambda r: pick(r, ["fulfilled", "unfulfilled", "partial"]), False, None),
    ("customers", "customer_id", lambda r: str(r.randint(10000, 99999)), False, None),
    ("customers", "user_id", lambda r: "u_" + hashlib.md5(str(r.random()).encode()).hexdigest()[:12], False, None),
    ("customers", "hashed_email", lambda r: hashlib.sha256(email(r).encode()).hexdigest(), False, None),
    ("customers", "session_id", lambda r: uuid.UUID(int=r.getrandbits(128)).hex, False, None),
    ("customers", "gender", lambda r: pick(r, ["F", "M", "F", "M", "Non-binary", ""]), False, None),
    ("customers", "age", lambda r: str(r.randint(18, 85)), False, None),
    ("customers", "age_group", lambda r: pick(r, ["18-24", "25-34", "35-44", "45-54", "55+"]), False, None),
    ("customers", "signup_date", event_date, False, None),
    ("customers", "last_login", lambda r: event_date(r, "%Y-%m-%dT%H:%M:%SZ"), False, None),
    ("customers", "loyalty_tier", lambda r: pick(r, ["bronze", "silver", "gold"]), False, None),
    ("customers", "email_opt_in", lambda r: pick(r, ["True", "False"]), False, None),
    ("customers", "has_phone", lambda r: pick(r, ["yes", "no"]), False, None),
    ("customers", "contact_method", lambda r: pick(r, ["email", "phone", "sms", "post"]), False, None),
    ("customers", "preferred_contact", lambda r: pick(r, ["Email", "Phone", "No contact"]), False, None),
    ("customers", "address_type", lambda r: pick(r, ["billing", "shipping"]), False, None),
    ("customers", "phone_model", lambda r: pick(r, ["iPhone 15", "Pixel 8", "Galaxy S24", "iPhone 13", "Moto G"]), False, None),
    ("customers", "email_status", lambda r: pick(r, ["sent", "opened", "bounced", "clicked"]), False, None),
    ("customers", "dob_verified", lambda r: pick(r, ["TRUE", "FALSE"]), False, None),
    ("customers", "lifetime_value", lambda r: f"{r.uniform(0, 3000):.2f}", False, None),
    ("hr", "job_title", lambda r: pick(r, ["Data Analyst", "Engineer", "Sales Manager", "Nurse", "Teacher"]), False, None),
    ("hr", "department", lambda r: pick(r, ["Sales", "HR", "Finance", "IT", "Operations"]), False, None),
    ("hr", "salary", lambda r: str(r.randint(22, 140) * 1000), False, None),
    ("hr", "start_date", event_date, False, None),
    ("hr", "tenure_years", lambda r: f"{r.uniform(0, 30):.1f}", False, None),
    ("hr", "performance_rating", lambda r: str(r.randint(1, 5)), False, None),
    ("web-logs", "server_ip", lambda r: ipv4(r, private=True), False, None),
    ("web-logs", "ip_version", lambda r: pick(r, ["4", "6"]), False, None),
    ("web-logs", "user_agent", lambda r: pick(r, ["Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5)", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "curl/8.4.0"]), False, None),
    ("web-logs", "path", lambda r: pick(r, ["/", "/pricing", "/blog/launch", "/login", "/api/v1/items"]), False, None),
    ("web-logs", "status_code", lambda r: pick(r, ["200", "200", "301", "404", "500"]), False, None),
    ("web-logs", "response_ms", lambda r: str(r.randint(5, 900)), False, None),
    ("web-logs", "version", lambda r: f"{r.randint(1, 4)}.{r.randint(0, 20)}.{r.randint(0, 9)}", False, None),
    ("web-logs", "referrer_domain", lambda r: pick(r, ["google.com", "news.ycombinator.com", "linkedin.com", "bing.com"]), False, None),
    ("web-logs", "website", lambda r: f"https://{pick(r, COMPANIES).split()[0].lower()}.example", False, None),
    ("support", "subject", lambda r: pick(r, ["Order not arrived", "Refund request", "Wrong size", "Login problem"]), False, None),
    ("support", "priority", lambda r: pick(r, ["low", "medium", "high"]), False, None),
    ("reviews", "review_text", lambda r: pick(r, ["Great coffee, slow service", "Loved it", "Too sweet for me", "Would buy again", "Arrived broken"]), False, None),
    ("reviews", "rating", lambda r: str(r.randint(1, 5)), False, None),
    ("reviews", "comments", lambda r: pick(r, ["Delivered late", "", "Good value", "Packaging could be better"]), False, None),
    ("travel", "flight", lambda r: f"{pick(r, ['BA', 'AI', 'LH', 'UA'])}{r.randint(100, 2999)}", False, None),
    ("travel", "destination", lambda r: pick(r, CITIES), False, None),
    ("travel", "Ticket", lambda r: r.choice([f"A/5 {digits(r, 5)}", f"PC {digits(r, 5)}", digits(r, 6)]), False, None),
    ("travel", "Cabin", lambda r: r.choice(["", f"C{r.randint(1, 140)}", f"B{r.randint(1, 99)}"]), False, None),
    ("travel", "Fare", lambda r: f"{r.uniform(5, 250):.4f}", False, None),
    ("sensors", "temperature_c", lambda r: f"{r.uniform(-5, 35):.1f}", False, None),
    ("sensors", "device_serial", lambda r: f"SN{digits(r, 8)}", False, None),
    ("sensors", "timestamp", lambda r: event_date(r, "%Y-%m-%d %H:%M"), False, None),
    ("google-forms", "Timestamp", lambda r: event_date(r, "%m/%d/%Y %H:%M:%S"), False, None),
    ("google-forms", "How did you hear about us?", lambda r: pick(r, ["Friend", "Instagram", "Google", "Flyer"]), False, None),
    ("misc", "colour", lambda r: f"#{digits(r, 6)}", False, None),
    ("misc", "error_code", lambda r: f"E{r.randint(1000, 1999)}", False, None),
    ("misc", "duration", lambda r: f"00:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}", False, None),
    ("misc", "percent", lambda r: f"{r.uniform(0, 100):.1f}%", False, None),
    ("misc", "species", lambda r: pick(r, ["Adelie", "Gentoo", "Chinstrap"]), False, None),
    ("misc", "team", lambda r: pick(r, ["Alpha", "Bravo", "Charlie", "Delta"]), False, None),
]

SAMPLES = [("coffee_sales.csv", {}), ("online_retail.csv", {"nrows": 20000}), ("stocks.csv", {})]


def build() -> list[dict]:
    columns = []
    for i, (source, name, make, personal, kind) in enumerate(SPECS):
        r = random.Random(f"{i}:{name}")
        columns.append({"source": source, "column": name, "values": [make(r) for _ in range(N)], "personal": personal, "kind": kind})
    for file, read_args in SAMPLES:
        df = pd.read_csv(ROOT / "data" / file, dtype=str, comment="#", **read_args)
        for name in df.columns:
            values = df[name].dropna().sample(N, random_state=0).tolist()
            columns.append({"source": f"sample:{file}", "column": name, "values": values, "personal": False, "kind": None})
    # stratified split: 40% of the personal columns and 40% of the rest go to dev
    rng = random.Random(0)
    for label in (True, False):
        group = [c for c in columns if c["personal"] is label]
        rng.shuffle(group)
        for j, c in enumerate(group):
            c["split"] = "dev" if j < round(0.4 * len(group)) else "test"
    for j, c in enumerate(columns, 1):
        c["id"] = f"c{j:03d}"
    return [{k: c[k] for k in ("id", "split", "source", "column", "personal", "kind", "values")} for c in columns]


def main() -> None:
    columns = build()
    names = [(c["source"], c["column"]) for c in columns]
    assert len(names) == len(set(names)), "each (source, column) once"
    (HERE / "columns.jsonl").write_text("".join(json.dumps(c) + "\n" for c in columns))
    spot = random.Random(1).sample(columns, 25)
    lines = ["# Spot-check: 25 labelled columns (seed 1)\n",
             "Rules: [LABELS.md](LABELS.md). Put an x in the last column where you disagree with the label.\n",
             "| id | column | first values | label | disagree? |", "|---|---|---|---|---|"]
    for c in spot:
        shown = " · ".join(v.replace("|", "/") for v in c["values"][:4])
        label = f"personal ({c['kind']})" if c["personal"] else "not personal"
        lines.append(f"| {c['id']} | `{c['column']}` | {shown} | {label} | |")
    (HERE / "spotcheck.md").write_text("\n".join(lines) + "\n")
    personal = sum(c["personal"] for c in columns)
    print(f"{len(columns)} columns ({personal} personal, {len(columns) - personal} not); "
          f"dev {sum(c['split'] == 'dev' for c in columns)}, test {sum(c['split'] == 'test' for c in columns)}")


if __name__ == "__main__":
    main()
