"""Check the judge before trusting it: correct answers must pass and bad answers must fail.

    uv run python evals/check_judge.py

For every case, a hand-written correct answer (filled with the answer key's values) should pass
all checks. If one fails, the check's wording is broken, not the agent. For a sample of cases,
three bad answers (empty, "I don't know", and a correct answer to a different question) should
fail, and so should answers with a specific mistake an agent actually made. Costs about $0.10.
"""

from concurrent.futures import ThreadPoolExecutor

import anthropic
from dotenv import load_dotenv

from answer_key import facts
from run_eval import CSV, JUDGE_MODEL, ROOT, cost, judge, load_cases

CORRECT = {
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
}
BAD_SAMPLE = ["top-store", "price-increase", "total-revenue", "product-averages", "customer-age"]
# Plausible answers with one specific mistake each, taken from real agent runs. All should fail.
SUBTLE_MISTAKES = {
    "product-averages": "Matcha Latte brings in the least, about {matcha_avg_12} a month averaged over the full year. "
    "It had no sales before March 2026, so it probably launched then. Averaged over just its 7 months on sale, "
    "it makes about {matcha_avg_on_sale} a month, which would put it just behind Espresso.",
}


def main() -> None:
    load_dotenv(ROOT / ".env")
    values = facts(CSV)
    cases = {c["id"]: c for c in load_cases()}
    correct = {cid: CORRECT[cid].format_map(values) for cid in cases}
    ids = list(cases)
    trials = [(cid, "correct", correct[cid], True) for cid in ids]
    for cid in BAD_SAMPLE:
        other = ids[(ids.index(cid) + 3) % len(ids)]
        trials += [(cid, "empty", "", False), (cid, "I don't know", "I don't know.", False),
                   (cid, f"answer to {other}", correct[other], False)]
    trials += [(cid, "subtle mistake", text.format_map(values), False) for cid, text in SUBTLE_MISTAKES.items()]

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
