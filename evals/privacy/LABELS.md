# Labelling rules for the privacy filter

Each labelled column is a column name and 30 example values. Its label answers one question: **would this column let someone identify or contact a specific person?** These rules were written before the filter, and the labels in `columns.jsonl` follow them.

## Personal (label `true`)

| Kind | What counts |
|---|---|
| `name` | A person's full, first or last name, in any format ("Okafor, Mr. Emeka") |
| `email` | Any email address, including work addresses |
| `phone` | Any phone number, in any national format |
| `address` | A street address or address line of a person (home, billing, shipping) |
| `postcode` | The postcode or ZIP code of a person's address |
| `birth_date` | A date of birth |
| `gov_id` | A government number: SSN, National Insurance, passport, Aadhaar, PAN, tax ID, driving licence |
| `financial` | A full card number, IBAN or bank account number |
| `ip_address` | The IP address of a visitor or user |
| `location` | A precise location of a person (home coordinates) |
| `handle` | A username or social media handle |
| `text_with_contact` | Free text where some values contain someone's email or phone number |

## Not personal (label `false`)

- Products, prices, quantities, categories, stores, companies, cities, countries, regions.
- Dates and times of events (orders, sign-ups, logins), as opposed to dates of birth.
- Demographics on their own: gender, age, age band, job title, salary.
- Pseudonymous IDs: customer numbers, hashed emails, session and user ids. On their own they don't say who someone is. This is a scope choice: under GDPR they still count as personal data, but warning about every customer number would make the warning noise.
- The postcode, coordinates or IP address of a business, store, office or server.
- Partial card numbers (last 4 digits) and card brands.
- Columns *about* contact details that don't hold any: opt-in flags, contact method ("email", "phone"), phone models, address types.

## Borderline cases, decided

- A generic company email (support@...) is still an email: `true`. The filter can't tell them apart, and neither could a visitor's file.
- Free text counts as personal if any of its 30 values contains an email or phone number.
- A column called "Name" is labelled by what it holds: people's names `true`; products, teams or companies `false`.

## Spot-check

`spotcheck.md` lists 25 columns picked at random (seed 1), with the first values and the label. Mark any label you disagree with; if more than 2 of 25 are wrong, the rules or the labels need another pass before the numbers are trusted.
