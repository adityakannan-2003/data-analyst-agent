"""Tests for the upload privacy filter. Its precision and recall are measured in evals/privacy/."""

import pandas as pd

from privacy import describe, personal_columns


def test_flags_personal_columns_and_leaves_look_alikes_alone():
    n = 10
    df = pd.DataFrame({
        "Customer": ["Jane Smith", "Raj Patel", "Maria Lopez", "Tom Brown", "Anna Novak"] * 2,
        "contact": [f"user{i}@example.com" for i in range(n)],
        "Mobile": [f"07700 9001{i:02d}" for i in range(n)],
        "card": [f"4111 1111 1111 {1111 + 0 * i}" for i in range(n)],  # passes the Luhn check
        "notes": ["Leave at door"] * (n - 1) + ["call me on +44 7700 900123"],
        "product": ["Iced Latte", "Cold Brew"] * 5,
        "store_postcode": ["SW1A 1AA"] * n,
        "email_opt_in": ["True", "False"] * 5,
        "barcode": ["5012345678900"] * n,
        "duration": ["00:03:12"] * n,
        "created_at": ["2026-03-01 10:00:00 +0000"] * n,
        "customer_id": [str(10000 + i) for i in range(n)],
    })
    found = personal_columns(df)
    assert found == {"Customer": "name", "contact": "email", "Mobile": "phone", "card": "financial", "notes": "text_with_contact"}
    assert "**contact** (email addresses)" in describe(found)


def test_empty_and_blank_columns_are_not_flagged():
    assert personal_columns(pd.DataFrame({"email": [None, "", "  "], "x": [1, 2, 3]})) == {}
