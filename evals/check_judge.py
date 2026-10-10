"""Check the judge before trusting it: correct answers must pass and bad answers must fail.

    uv run python evals/check_judge.py                  # the coffee eval
    uv run python evals/check_judge.py --suite retail   # the real-data eval

For every case, a hand-written correct answer (filled with the answer key's values) should pass
all checks. If one fails, the check's wording is broken, not the agent. For a sample of cases,
three bad answers (empty, "I don't know", and a correct answer to a different question) should
fail, and so should answers with a specific mistake or trap. Costs about $0.10 per suite.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import anthropic
from dotenv import load_dotenv

import run_eval
from run_eval import JUDGE_MODEL, ROOT, cost, judge

COFFEE_CORRECT = {
    "top-store": "Downtown brought in the most revenue: {downtown_revenue} for the year, ahead of Riverside and Campus.",
    "weekend-leader": "Riverside does the most business on weekends, by both revenue and number of orders.",
    "downtown-weekends": "Downtown is much quieter on weekends. Revenue per day falls from {downtown_weekday_per_day} on weekdays to {downtown_weekend_per_day} on weekends, about {downtown_weekend_drop} lower.",
    "seasonal-store": "Campus varies the most through the year. It's quietest from June to August, during the summer break, with another dip over the winter break.",
    "iced-season": "Iced drinks (Cold Brew and Iced Latte) sell best in July and worst in January. July sales are about {iced_ratio} times January's.",
    "price-increase": "Only a little. Latte units fell about {latte_drop} from Oct-Mar to Apr-Sep, but espresso, whose price didn't change, fell {espresso_drop} over the same months as customers switched to iced drinks. Relative to espresso, lattes lost about 15%, so the price increase explains only a small part of the drop.",
    "matcha-launch": "The Matcha Latte was first sold on {matcha_launch}.",
    "product-averages": "Average monthly revenue: Latte $3,432, Cappuccino $2,254, Iced Latte $1,879, Cold Brew $1,528, Croissant $1,508, Muffin $1,268, Espresso $1,071. The Matcha Latte launched in March 2026, so I averaged it over its months on sale: {matcha_avg_on_sale} a month. A 12-month average ({matcha_avg_12}) would understate it.",
    "busiest-hours": "Counting orders: Downtown is busiest in the 8am hour, Riverside in the 11am hour, and Campus in the 11am hour.",
    "total-revenue": "Total net revenue for the year was {total_net}, after removing {duplicates} duplicate rows and subtracting refunds.",
    "product-count": "The shop sells {products} products: Espresso, Latte, Cappuccino, Cold Brew, Iced Latte, Matcha Latte, Croissant and Muffin.",
    "latte-units": "We sold {latte_units} lattes over the year, net of refunds.",
    "missing-store": "{missing_rev} of revenue, across {missing_rows} rows, has no store recorded.",
    "refunds": "Refunds totalled {refunds_total} across {refund_rows} refund lines.",
    "data-problems": "Yes. There are {duplicates} duplicate rows, prices are stored as text like \"$4.75\", product names are spelled inconsistently (different capitalization and trailing spaces), {missing_rows} rows have no store, and refunds appear as negative quantities.",
    "profit-margin": "The data has no cost information, only sale prices, so the profit margin on lattes can't be calculated from it.",
    "customer-age": "The data has no information about customers' ages, so I can't work out the average age of loyalty members.",
    "latte-price": "The latte price went from $4.75 to $5.25 on 1 April 2026. Weighted by lattes sold, the average price for the year was {latte_avg_price}.",
    "order-value": "The average order value was {avg_order_value}, adding up all the lines of each order.",
    "food-orders": "{food_order_share} of orders include a croissant or muffin.",
    "loyal-cash": "{cash_among_loyal} of loyalty members' orders are paid in cash.",
    "half-year-growth": "Among products sold all year, Cold Brew grew the most: {cold_brew_growth} in units, ahead of Iced Latte at {iced_latte_growth}. The Matcha Latte shows {matcha_growth}, but that's inflated because it only went on sale in March 2026.",
}
COFFEE_BAD_SAMPLE = ["top-store", "price-increase", "total-revenue", "product-averages", "customer-age"]
# Plausible answers with one specific mistake each, taken from real agent runs. All should fail.
COFFEE_SUBTLE_MISTAKES = [
    ("product-averages", "Matcha Latte brings in the least, about {matcha_avg_12} a month averaged over the full year. "
     "It had no sales before March 2026, so it probably launched then. Averaged over just its 7 months on sale, "
     "it makes about {matcha_avg_on_sale} a month, which would put it just behind Espresso."),  # wrong ranking
    ("product-averages", "- Matcha Latte: {matcha_avg_12} over 12 months, or $1,031 over its 7 selling months\n\n"
     "Matcha Latte has no sales from Oct 2025 to Feb 2026, so it looks like it launched in March 2026. "
     "I counted those months as $0 in the {matcha_avg_12} figure. If you only want the months it was on sale, use $1,031."),  # wrong figure
    # The trap each held-out question sets
    ("latte-price", "The average latte price was $5.00: it was $4.75 until April and $5.25 after."),
    ("order-value", "The average order value is {avg_line_value}."),
    ("food-orders", "{food_line_share} of orders include food."),
    ("loyal-cash", "{loyal_among_cash} of loyalty members' orders are paid in cash."),
    ("half-year-growth", "The Matcha Latte grew the most, up {matcha_growth} from the first half of the year to the second."),
]


RETAIL_CORRECT = {
    "best-month": "{best_month} had the highest sales: {best_month_net} after cancellations.",
    "weakest-month": "The data ends on {last_day}, so {partial_month} is incomplete and its low total isn't comparable. The weakest full month was {weakest_full_net}, after subtracting cancellations.",
    "weekdays": "{best_weekday} brings in the most sales. There are no Saturday sales in the data at all, so Saturday can't be compared.",
    "top-country": "The {top_country} brings in the most revenue outside the UK, about {top_country_revenue}.",
    "uk-share": "About {uk_share} of revenue comes from the UK.",
    "top-units": "{top_units_product} sold the most units: {top_units} after cancellations. {trap_units_product} shows {trap_units} units, but that was one order cancelled the same day.",
    "top-revenue-product": "{top_revenue_product} brought in the most revenue, about {top_revenue} after cancellations. DOTCOM POSTAGE shows {postage_revenue}, but it's a postage charge, not a product.",
    "repeat-customers": "{repeat_share} of customers with a customer ID ordered more than once, counting orders that weren't cancelled.",
    "total-revenue": "Total revenue after cancellations was {total_net}: gross sales of {total_gross} minus {cancelled_value} of cancellations, after removing {duplicates} duplicate rows.",
    "customers": "{customers} customers bought something. About {missing_id_share} of rows have no customer ID, so the real number of customers is likely higher.",
    "order-value": "The average order value is {aov}, adding up each invoice's lines and leaving out cancellations. The median is {aov_median}.",
    "cancellations": "{cancelled_orders} orders were cancelled (invoices starting with \"C\"), about {cancelled_share} of all invoices.",
    "data-problems": "Yes: cancelled orders with negative quantities, about {missing_id_share} of rows with no customer ID, {duplicates} duplicate rows, lines that aren't products (postage, Amazon fees, discounts, manual adjustments), and {zero_price_lines} lines with zero or negative prices.",
    "profit-margin": "The data has no cost information, only selling prices, so the profit margin on the Regency cakestand can't be calculated from it.",
    "conversion-rate": "The data only has orders, with no visitor or traffic information, so the website's conversion rate can't be calculated from it.",
}
RETAIL_BAD_SAMPLE = ["best-month", "top-units", "total-revenue", "customers", "conversion-rate"]
# The trap each retail question sets. All should fail.
RETAIL_SUBTLE_MISTAKES = [
    ("weakest-month", "{partial_month} had the lowest sales."),
    ("weekdays", "{best_weekday} brings in the most sales. Saturday is the quietest day."),
    ("top-units", "{trap_units_product} sold the most units: {trap_units}."),
    ("top-revenue-product", "DOTCOM POSTAGE brought in the most revenue: {postage_revenue}."),
    ("total-revenue", "Total revenue was {total_gross}."),
    ("customers", "The shop has {customers} customers."),
    ("order-value", "The average order value is {line_value}."),
    ("cancellations", "{cancelled_lines} orders were cancelled."),
    ("profit-margin", "The profit margin on the Regency cakestand is about 45%."),
]
SUITES = {
    "coffee": (COFFEE_CORRECT, COFFEE_BAD_SAMPLE, COFFEE_SUBTLE_MISTAKES),
    "retail": (RETAIL_CORRECT, RETAIL_BAD_SAMPLE, RETAIL_SUBTLE_MISTAKES),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that the judge passes correct answers and fails bad ones.")
    parser.add_argument("--suite", default="coffee", choices=sorted(SUITES), help="Which eval's checks to test.")
    suite = parser.parse_args().suite
    CORRECT, BAD_SAMPLE, SUBTLE_MISTAKES = SUITES[suite]
    load_dotenv(ROOT / ".env")
    run_eval.use_suite(suite)
    values = run_eval.KEY.facts(run_eval.CSV)
    cases = {c["id"]: c for c in run_eval.load_cases()}
    correct = {cid: CORRECT[cid].format_map(values) for cid in cases}
    ids = list(cases)
    trials = [(cid, "correct", correct[cid], True) for cid in ids]
    for cid in BAD_SAMPLE:
        other = ids[(ids.index(cid) + 3) % len(ids)]
        trials += [(cid, "empty", "", False), (cid, "I don't know", "I don't know.", False),
                   (cid, f"answer to {other}", correct[other], False)]
    trials += [(cid, "subtle mistake", text.format_map(values), False) for cid, text in SUBTLE_MISTAKES]

    client = anthropic.Anthropic(max_retries=8)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda t: judge(client, JUDGE_MODEL, cases[t[0]], t[2]), trials))

    wrong, spend = 0, 0.0
    for (cid, kind, _, should_pass), (verdicts, usage) in zip(trials, results):
        spend += cost(JUDGE_MODEL, usage) or 0
        passed = all(v["met"] for v in verdicts.values())
        if passed != should_pass:
            wrong += 1
            print(f"WRONG  {cid} ({kind}): judged {'pass' if passed else 'fail'}")
            for check_id, v in verdicts.items():
                print(f"         {'✓' if v['met'] else '✗'} {check_id}: {v['reason']}")
    n_good = len(ids)
    n_bad = len(trials) - n_good
    good_ok = sum(all(v["met"] for v in r[0].values()) for r in results[:n_good])
    bad_ok = sum(not all(v["met"] for v in r[0].values()) for r in results[n_good:])
    print(f"Correct answers passed: {good_ok}/{n_good}. Bad answers failed: {bad_ok}/{n_bad}. Judge cost ${spend:.2f}.")
    raise SystemExit(1 if wrong else 0)


if __name__ == "__main__":
    main()
