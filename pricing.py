"""What an answer cost, from the token counts the API reports."""

# Dollars per million tokens (Claude API list prices, October 2026). Cache writes cost 1.25x input.
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_read": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
}


def usage_dict(u) -> dict:
    """Usage in the API's shape, where input_tokens excludes cache reads and writes."""
    if hasattr(u, "cache_read_tokens"):  # agent.Usage, which totals all input
        return {
            "input_tokens": u.input_tokens - u.cache_read_tokens - u.cache_write_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": u.cache_read_tokens,
            "cache_creation_input_tokens": u.cache_write_tokens,
        }
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": u.cache_read_input_tokens or 0,
        "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
    }


def cost(model: str, usage: dict) -> float | None:
    """Token cost in dollars, or None for a model without a price here.

    Code execution time is billed separately ($0.05 per container-hour after 1,550 free hours a
    month per organization), so it isn't counted."""
    price = PRICES.get(model)
    if not price or not usage:
        return None
    return (
        usage["input_tokens"] * price["input"]
        + usage["cache_creation_input_tokens"] * price["input"] * 1.25
        + usage["cache_read_input_tokens"] * price["cache_read"]
        + usage["output_tokens"] * price["output"]
    ) / 1e6
