"""Answer key for the real-data eval, computed from the UCI Online Retail dataset.

Real data has no planted answers, and some questions have more than one fair reading.
So each value is computed under the cleaning choices a careful analyst might make, and
the checks in evals/retail_cases.json accept the fair readings and reject the traps.

The default cleaning: drop exact duplicate rows; treat invoices starting with "C" as
cancellations; count "sales" as non-cancelled lines with a positive quantity and price.
Lines whose stock code doesn't start with five digits (postage, fees, discounts, manual
adjustments) aren't products.

Get the data first: uv run python scripts/fetch_online_retail.py
"""

from pathlib import Path

import pandas as pd


def load(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(csv_path, dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"])
    raw["revenue"] = raw["Quantity"] * raw["UnitPrice"]
    df = raw.drop_duplicates().copy()
    df["cancelled"] = df["InvoiceNo"].str.startswith("C")
    df["is_product"] = df["StockCode"].str.match(r"^\d{5}")
    return raw, df


def pounds(x: float) -> str:
    return f"£{x:,.0f}"


def millions(x: float) -> str:
    return f"£{x / 1e6:.2f} million"


def facts(csv_path: Path) -> dict[str, str]:
    raw, df = load(csv_path)
    sales = df[~df["cancelled"] & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    f = {}

    def by_month(frame):
        return frame.groupby(frame["InvoiceDate"].dt.to_period("M"))["revenue"].sum()

    # Months: the last one is partial, and the weakest full month depends on cancellations
    net_months, sales_months = by_month(df), by_month(sales)
    f["best_month"] = net_months.idxmax().strftime("%B %Y")
    f["best_month_net"], f["best_month_sales"] = millions(net_months.max()), millions(sales_months.max())
    f["last_day"] = df["InvoiceDate"].max().strftime("%-d %B %Y")
    f["partial_month"] = df["InvoiceDate"].max().strftime("%B %Y")
    full_net, full_sales = net_months.iloc[:-1], sales_months.iloc[:-1]
    f["weakest_full_net"], f["weakest_full_sales"] = full_net.idxmin().strftime("%B %Y"), full_sales.idxmin().strftime("%B %Y")

    # Days of the week
    weekdays = df.groupby(df["InvoiceDate"].dt.day_name())["revenue"].sum()
    f["best_weekday"] = weekdays.idxmax()
    f["saturday_rows"] = str(int((raw["InvoiceDate"].dt.day_name() == "Saturday").sum()))

    # Countries
    countries = df.groupby("Country")["revenue"].sum()
    f["top_country"] = countries.drop("United Kingdom").idxmax()
    f["top_country_revenue"] = pounds(countries.drop("United Kingdom").max())
    shares = [frame.loc[frame["Country"] == "United Kingdom", "revenue"].sum() / frame["revenue"].sum()
              for frame in (df, sales, df[df["is_product"]])]
    f["uk_share"], f["uk_share_lo"], f["uk_share_hi"] = f"{shares[0]:.0%}", f"{min(shares) - 0.005:.1%}", f"{max(shares) + 0.005:.1%}"

    # Products
    units = df[df["is_product"]].groupby("Description")["Quantity"].sum()
    gross_units = sales.groupby("Description")["Quantity"].sum()
    f["top_units_product"], f["top_units"] = units.idxmax(), f"{units.max():,}"
    f["top_units_gross"] = f"{gross_units[units.idxmax()]:,}"
    f["trap_units_product"], f["trap_units"] = gross_units.idxmax(), f"{gross_units.max():,}"
    # Grouping by stock code instead of name can change the winner: one code is sold under two names
    products = df[df["is_product"]]
    by_code = products.groupby("StockCode")["Quantity"].sum()
    names = products.loc[products["StockCode"] == by_code.idxmax(), "Description"].dropna().unique()
    f["top_units_code_product"] = f"{' / '.join(sorted(names))} (stock code {by_code.idxmax()})"
    f["top_units_code"] = f"{by_code.max():,}"
    product_revenue = df[df["is_product"]].groupby("Description")["revenue"].sum()
    f["top_revenue_product"], f["top_revenue"] = product_revenue.idxmax(), pounds(product_revenue.max())
    f["top_revenue_gross"] = pounds(sales.groupby("Description")["revenue"].sum()[product_revenue.idxmax()])
    f["postage_revenue"] = pounds(df.loc[df["StockCode"] == "DOT", "revenue"].sum())

    # Totals
    total, gross, cancelled = df["revenue"].sum(), sales["revenue"].sum(), -df.loc[df["cancelled"], "revenue"].sum()
    f["total_net"], f["total_lo"], f["total_hi"] = millions(total), millions(total * 0.995), millions(total * 1.005)
    f["total_gross"], f["cancelled_value"] = millions(gross), pounds(cancelled)

    # Customers and orders
    buyers = sales["CustomerID"].dropna().nunique()
    f["customers"], f["customers_any"] = f"{buyers:,}", f"{df['CustomerID'].dropna().nunique():,}"
    f["missing_id_share"] = f"{raw['CustomerID'].isna().mean():.0%}"
    orders_per_customer = sales.dropna(subset=["CustomerID"]).groupby("CustomerID")["InvoiceNo"].nunique()
    repeat = (orders_per_customer >= 2).mean()
    f["repeat_share"], f["repeat_lo"], f["repeat_hi"] = (f"{x:.1%}" for x in (repeat, repeat - 0.02, repeat + 0.02))
    order_value = sales.groupby("InvoiceNo")["revenue"].sum()
    with_id = sales.dropna(subset=["CustomerID"]).groupby("InvoiceNo")["revenue"].sum()
    f["aov"], f["aov_with_id"], f["aov_median"] = f"£{order_value.mean():.0f}", f"£{with_id.mean():.0f}", f"£{order_value.median():.0f}"
    f["aov_lo"], f["aov_hi"] = f"£{min(order_value.mean(), with_id.mean()) * 0.98:.0f}", f"£{max(order_value.mean(), with_id.mean()) * 1.02:.0f}"
    f["line_value"] = f"£{sales['revenue'].mean():.0f}"
    cancelled_invoices = df.loc[df["cancelled"], "InvoiceNo"].nunique()
    f["cancelled_orders"] = f"{cancelled_invoices:,}"
    f["cancelled_share"] = f"{cancelled_invoices / df['InvoiceNo'].nunique():.0%}"
    f["cancelled_lines"] = f"{int(raw['InvoiceNo'].str.startswith('C').sum()):,}"

    # Data problems
    f["duplicates"] = f"{len(raw) - len(df):,}"
    f["zero_price_lines"] = f"{int((raw['UnitPrice'] <= 0).sum()):,}"
    return f
