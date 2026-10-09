"""The eval's answer key, computed from the data rather than typed in.

Each value is computed with the cleaning a careful analyst would do: drop exact
duplicate rows, turn "$4.75" prices into numbers, merge product spellings, and
count refunds (negative quantities) as negative revenue. Because the key is
computed, it stays right if the sample data is regenerated.

`facts()` returns the values as display strings, which fill the {placeholders}
in evals/cases.json.
"""

from pathlib import Path

import pandas as pd

ICED = ["Cold Brew", "Iced Latte"]


def load_clean(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(csv_path)
    df = raw.drop_duplicates().copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["product"] = df["product"].str.strip().str.title()
    df["price"] = df["unit_price"].str.lstrip("$").astype(float)
    df["revenue"] = df["quantity"] * df["price"]
    return raw, df


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{abs(x):.0%}"


def facts(csv_path: Path) -> dict[str, str]:
    raw, df = load_clean(csv_path)
    sales = df[df["quantity"] > 0]
    f = {}

    # Revenue by store, weekends
    by_store = df.groupby("store")["revenue"].sum()
    f["downtown_revenue"] = money(by_store["Downtown"])
    f["downtown_revenue_lo"], f["downtown_revenue_hi"] = money(by_store["Downtown"] * 0.99), money(by_store["Downtown"] * 1.01)
    weekend = df["ts"].dt.dayofweek >= 5
    days = df.groupby(weekend)["ts"].apply(lambda s: s.dt.date.nunique())
    downtown = df[df["store"] == "Downtown"].groupby(weekend)["revenue"].sum() / days
    f["downtown_weekday_per_day"], f["downtown_weekend_per_day"] = money(downtown[False]), money(downtown[True])
    f["downtown_weekend_drop"] = pct(downtown[True] / downtown[False] - 1)

    # Seasonality
    iced = sales[sales["product"].isin(ICED)].groupby(sales["ts"].dt.to_period("M"))["quantity"].sum()
    f["iced_ratio"] = f"{iced.max() / iced.min():.1f}"
    lows = iced[iced <= iced.min() * 1.02]  # months this close to the minimum are effectively tied
    f["iced_low"] = " or ".join(m.strftime("%B") for m in lows.index)
    f["iced_low_detail"] = (
        "effectively tied at " + " and ".join(f"{n:,}" for n in lows) + " units" if len(lows) > 1 else f"{lows.iloc[0]:,} units"
    )

    # Price increase: average monthly units, Oct-Mar vs Apr-Sep
    def change(product: str) -> float:
        units = sales[sales["product"] == product].groupby(sales["ts"] >= "2026-04-01")["quantity"].sum()
        return units[True] / units[False] - 1

    f["latte_drop"], f["espresso_drop"] = pct(change("Latte")), pct(change("Espresso"))

    # Matcha launch
    matcha = df[df["product"] == "Matcha Latte"]
    f["matcha_launch"] = matcha["ts"].min().strftime("%-d %B %Y")
    monthly = matcha.groupby(matcha["ts"].dt.to_period("M"))["revenue"].sum()
    f["matcha_avg_on_sale"], f["matcha_avg_12"] = money(monthly.mean()), money(monthly.sum() / 12)
    # The two products either side of the Matcha Latte's months-on-sale average (both sold all year)
    by_product = df.groupby([df["ts"].dt.to_period("M"), "product"])["revenue"].sum().unstack()
    f["espresso_avg"], f["muffin_avg"] = money(by_product["Espresso"].mean()), money(by_product["Muffin"].mean())

    # Totals and cleaning
    total, gross = df["revenue"].sum(), sales["revenue"].sum()
    raw_price = raw["unit_price"].str.lstrip("$").astype(float)
    f["total_net"], f["total_lo"], f["total_hi"] = money(total), money(total - 100), money(total + 100)
    f["total_gross"] = money(gross)
    f["total_no_dedupe"] = money((raw["quantity"] * raw_price).sum())
    refunds = -df.loc[df["quantity"] < 0, "revenue"].sum()
    f["refunds_total"], f["refunds_lo"], f["refunds_hi"] = f"${refunds:,.2f}", money(refunds * 0.98), money(refunds * 1.02)
    f["refund_rows"] = str(int((df["quantity"] < 0).sum()))
    f["raw_spellings"], f["products"] = str(raw["product"].nunique()), str(df["product"].nunique())
    lattes = df.loc[df["product"] == "Latte", "quantity"].sum()
    f["latte_units"], f["latte_lo"], f["latte_hi"] = f"{lattes:,}", f"{int(lattes * 0.99):,}", f"{int(lattes * 1.01):,}"
    f["latte_no_norm"] = f"{int(raw.loc[raw['product'] == 'Latte', 'quantity'].sum()):,}"
    f["duplicates"] = str(len(raw) - len(df))

    # Rows with no store, before and after recovering the store from other lines of the same order
    no_store = df["store"].isna()
    f["missing_rows"], f["missing_rev"] = str(int(no_store.sum())), money(df.loc[no_store, "revenue"].sum())
    filled = df["store"].fillna(df.groupby("order_id")["store"].transform("first"))
    f["missing_rows_after_fill"] = str(int(filled.isna().sum()))
    f["missing_rev_after_fill"] = money(df.loc[filled.isna(), "revenue"].sum())
    return f
