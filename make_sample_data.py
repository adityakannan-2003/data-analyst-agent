"""Generate data/coffee_sales.csv: a year of sales at a three-store coffee chain.

The data is synthetic, with patterns planted on purpose so you can check
whether the agent's answers are right (see "Sample data" in the README):

- Downtown is busy on weekdays and quiet on weekends; Riverside is the reverse.
- Campus sales collapse over the summer and the winter break.
- Iced drinks rise in summer; hot drinks rise in winter.
- Latte and Cappuccino went up $0.50 on 2026-04-01, and their volume dipped.
- Matcha Latte launched on 2026-03-01.

It is also messy in the ways real exports are: prices stored as "$4.75"
strings, inconsistent product spelling, missing store names, duplicate rows,
and refunds recorded as negative quantities.

Run: uv run python make_sample_data.py
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
START, END = date(2025, 10, 1), date(2026, 9, 30)
PRICE_INCREASE = date(2026, 4, 1)
MATCHA_LAUNCH = date(2026, 3, 1)

PRODUCTS = {  # name: (base price, base share of orders, kind)
    "Espresso": (3.00, 0.10, "hot"),
    "Latte": (4.75, 0.22, "hot"),
    "Cappuccino": (4.50, 0.15, "hot"),
    "Cold Brew": (4.25, 0.10, "iced"),
    "Iced Latte": (5.00, 0.10, "iced"),
    "Matcha Latte": (5.25, 0.08, "hot"),
    "Croissant": (3.50, 0.13, "food"),
    "Muffin": (3.25, 0.12, "food"),
}

STORES = {  # name: (orders per weekday, weekend multiplier, peak hour)
    "Downtown": (34, 0.4, 8),
    "Riverside": (22, 1.6, 11),
    "Campus": (26, 0.7, 12),
}


def season(day: date) -> float:
    """+1 at the peak of summer (mid-July), -1 in mid-January."""
    return float(np.cos(2 * np.pi * (day.timetuple().tm_yday - 196) / 365))


def store_volume(store: str, day: date) -> float:
    weekday_orders, weekend_mult, _ = STORES[store]
    volume = weekday_orders * (weekend_mult if day.weekday() >= 5 else 1.0)
    if store == "Campus":
        summer_break = date(day.year, 6, 1) <= day <= date(day.year, 8, 25)
        winter_break = (day.month == 12 and day.day >= 20) or (day.month == 1 and day.day <= 10)
        if summer_break:
            volume *= 0.3
        elif winter_break:
            volume *= 0.25
    return volume


def product_weights(day: date) -> tuple[list[str], np.ndarray]:
    s = season(day)
    names, weights = [], []
    for name, (_, share, kind) in PRODUCTS.items():
        if name == "Matcha Latte" and day < MATCHA_LAUNCH:
            continue
        w = share
        if kind == "iced":
            w *= np.exp(0.9 * s)  # roughly 2.5x in summer, 0.4x in winter
        elif kind == "hot":
            w *= np.exp(-0.25 * s)
        if name in ("Latte", "Cappuccino") and day >= PRICE_INCREASE:
            w *= 0.85  # demand dips after the price increase
        names.append(name)
        weights.append(w)
    weights = np.array(weights)
    return names, weights / weights.sum()


def price(name: str, day: date) -> float:
    base = PRODUCTS[name][0]
    if name in ("Latte", "Cappuccino") and day >= PRICE_INCREASE:
        base += 0.50
    return base


def messy_name(name: str, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.01:
        return name.lower()
    if r < 0.02:
        return name + " "
    if r < 0.025:
        return name.upper()
    return name


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    order_id = 100000
    day = START
    while day <= END:
        names, weights = product_weights(day)
        for store, (_, _, peak) in STORES.items():
            for _ in range(rng.poisson(store_volume(store, day))):
                order_id += 1
                hour = int(rng.normal(peak, 2.2))
                while not 6 <= hour <= 19:  # redraw: clipping would pile orders up at opening and closing time
                    hour = int(rng.normal(peak, 2.2))
                ts = datetime(day.year, day.month, day.day, hour, int(rng.integers(0, 60)))
                payment = rng.choice(["card", "app", "cash"], p=[0.6, 0.3, 0.1])
                loyalty = "yes" if rng.random() < (0.45 if payment == "app" else 0.2) else "no"
                for name in rng.choice(names, size=rng.choice([1, 2], p=[0.75, 0.25]), replace=False, p=weights):
                    qty = int(rng.choice([1, 2, 3], p=[0.85, 0.12, 0.03]))
                    if rng.random() < 0.004:
                        qty = -qty  # refund
                    rows.append({
                        "order_id": order_id,
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                        "store": "" if rng.random() < 0.015 else store,
                        "product": messy_name(str(name), rng),
                        "quantity": qty,
                        "unit_price": f"${price(str(name), day):.2f}",
                        "payment_method": payment,
                        "loyalty_member": loyalty,
                    })
        day += timedelta(days=1)

    df = pd.DataFrame(rows)
    dupes = df.sample(frac=0.005, random_state=SEED)  # duplicated export rows
    return pd.concat([df, dupes]).sort_values(["timestamp", "order_id"], kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "coffee_sales.csv"
    out.parent.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
