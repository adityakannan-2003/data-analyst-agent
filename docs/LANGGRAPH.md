# LangGraph vs. the hand-written loop

The agent's loop in [agent.py](../agent.py) is plain Python with no framework. [graph_agent.py](../graph_agent.py) rebuilds that loop in LangGraph and reuses everything else: the system prompt, tools, sandbox and tool code. Tests check that both loops send the API requests with the same content ([tests/test_loops_match.py](../tests/test_loops_match.py)), so the loop is the only difference. Both ran on the same 37 eval questions: 22 about the made-up coffee chain and 15 about the real retail data.

**Result: no measurable difference, so ship the hand-written loop.** Both loops passed all 37 questions on both runs. Steps and time per answer matched, and so did cost once the batches that filled the prompt cache are left out: $0.0183 against $0.0186 per coffee answer, and $0.0136 against $0.0133 per retail answer (LangGraph first). The LangGraph version needs 2.3 times as much code: 150 lines against 66.

## Results

Same model (Sonnet 5.5), effort, tools, grader and day, with two runs per question. To print these tables again: `uv run python evals/compare_loops.py`.

| Per answer, unless stated | Coffee: LangGraph | Coffee: hand-written | Retail: LangGraph | Retail: hand-written |
|---|---|---|---|---|
| Runs | 44 | 44 | 30 | 30 |
| Questions passed on every run | 22 of 22 | 22 of 22 | 15 of 15 | 15 of 15 |
| Model turns | 3.48 | 3.50 | 2.43 | 2.47 |
| Tool calls | 2.57 | 2.64 | 1.47 | 1.47 |
| Agent cost, all runs | $0.0200 | $0.0186 | $0.0136 | $0.0152 |
| Agent cost, without each suite's first batch | $0.0183 | $0.0186 | $0.0136 | $0.0133 |
| Time | 11.2 s | 11.6 s | 9.3 s | 9.1 s |

Each suite's first batch paid extra to fill the prompt cache, which is why there are two cost rows.

**No measurable gap in pass rate, steps or time.** Compared question by question, which cancels out how hard each question is, the gaps in turns, tool calls and time are small, and each 95% interval includes zero. The cost intervals over all runs don't; the paragraph under the table explains why.

| LangGraph minus hand-written, per answer (95% interval) | Coffee | Retail |
|---|---|---|
| Model turns | −0.02 (−0.23 to +0.18) | −0.03 (−0.17 to +0.10) |
| Tool calls | −0.07 (−0.27 to +0.14) | 0.00 (−0.13 to +0.13) |
| Agent cost, all runs | +$0.0014 (+$0.0001 to +$0.0026) | −$0.0017 (−$0.0023 to −$0.0010) |
| Agent cost, without each suite's first batch | −$0.0003 (−$0.0025 to +$0.0014) | +$0.0003 (−$0.0006 to +$0.0012) |
| Time | −0.4 s (−1.2 to +0.3) | +0.2 s (−0.3 to +0.7) |

**The cost gaps come from run order, not from the loop.** A question's first request (the system prompt, tools and the question) has the same content in both loops, and the API keeps it cached for 5 minutes after each use; no question went that long between runs. Only the JSON key order differs, which the cache ignores. So the first time each question was asked, its first turn wrote about 1,500 tokens to the cache (1,600 on retail), and every later run of it, by either loop, read them back. LangGraph asked the coffee questions first and the hand-written loop asked the retail questions first, so each paid for the cache on one suite. Without those first batches, the gaps are $0.0003 per answer, and both intervals include zero.

| Lines of code only one loop needs | Hand-written | LangGraph |
|---|---|---|
| Total | 66 | 150 |
| The loop | 66 (`ask` 43, `_call_model` 23) | 81 (`GraphAgent`) |
| The graph | none | 11 (`build_graph` 8, `route` 3) |
| Translating LangChain's objects | none | 58 (`TurnPrinter` 26, `api_usage` 12, `api_message` 12, `read_tool_input` 8) |

Lines are counted without blank lines, comments or docstrings. Code both loops use (the prompt, tools, tool code, sandbox and terminal printer) isn't counted.

## Which loop to ship

The hand-written one:

- **Same results.** The loop changed nothing the evals measure: pass rate, steps, cost or time. Both loops scored 100%, so the evals show that LangGraph broke nothing on the paths they cover. They can't show which loop answers better, and they don't reach the rare cases under [Where the loops still differ](#where-the-loops-still-differ).
- **Less code to read and own.** 66 lines against 150. The graph itself is 11; most of the extra lines translate LangChain's objects into what the rest of the program uses.
- **Fewer surprises.** Matching the hand-written loop took eight fixes found by replaying API streams through both loops, and three more from a code review. A few differences can't be fixed without overriding LangChain's private methods (see [Where the loops still differ](#where-the-loops-still-differ)).
- **No extra packages.** LangChain and LangGraph add 21 packages. Here they sit in their own dependency group, which the public app doesn't install.
- **The public page has its own loop.** [cloud_agent.py](../cloud_agent.py) runs the code in Anthropic's sandbox, so moving to LangGraph would mean rebuilding that loop too.

LangGraph would be worth it if the agent became a workflow with branches: several agents, a person approving steps before they run, runs that resume after a crash, or steps that run in parallel. It has built-in support for these: interrupts, checkpoints saved to a database, and parallel nodes. This agent is one loop.

## What LangGraph made easier

- **Seeing the control flow.** The loop is declared, START → model → route → tools → model, instead of living in a `for` loop. Adding a step, such as a check before tools run, is one more node and edge.
- **Memory and a step cap come built in.** The checkpointer keeps the conversation between questions, and `recursion_limit` stops a run after a set number of steps. Here that saved little: `ask` still copies the history into a list for the eval traces, rolling back a failed question needs its own code, and the limit counts graph steps, not model turns.

## What LangGraph made harder

- **Translating LangChain's objects.** LangChain keeps its own message objects, so the program needs 58 lines to turn them into what it uses: token counts (LangChain's input count includes cached tokens), tool input parsed the SDK's way, eval traces, and the stream events the terminal printer reads.
- **Behaviour hidden in the wrapper.** These turned up only when a recorded real stream, and 93 rare cases scripted from it, were replayed through both loops offline:
  - LangChain parses streamed tool input with its own parser, which disagrees with the SDK's both ways: it closes a string cut off mid-stream, so cut-off code would run, and it rejects some input the SDK accepts. `read_tool_input` now makes the SDK's own call (jiter, in partial mode).
  - LangGraph's usual way to stream tokens, `stream_mode="messages"`, runs each node in a worker thread. Ctrl+C then couldn't stop a model call: the loop kept reading, and paying for, the whole response, and the rollback raised an error. The graph now runs with `invoke` on the main thread, `TurnPrinter` prints through a LangChain callback, and a rollback resets the whole saved conversation.
  - A refused turn can still contain tool calls. The hand-written loop checks for a refusal first; the graph's router now checks too.
  - Smaller ones: no request timeout by default (the SDK's is 10 minutes), LangChain's own error classes in place of the SDK's, a string subclass for answer text, and text and tool calls out of order in the eval traces.
- **A code review found three more.** By default the checkpointer saves a full copy of the conversation after every step: a 12-question chat with a chart per question kept 18 MB of copies for 0.4 MB of conversation. It now saves once per question, which brings that chat to 2.5 MB, but each save is still a full copy. An error while printing was logged and ignored instead of stopping the turn. And a turn retried after a broken stream left its last printed line unfinished.
- **Graph details to get right.** `recursion_limit` counts graph steps; each step here runs one node, so 25 model turns need a limit of 50. `GraphRecursionError` is a kind of `RuntimeError`, so it must be caught first. Rolling back a question edits the saved state as if from START, so the router, which picks the next node, never runs on the edited list.
- **21 more packages** to install and keep up to date.

## Where the loops still differ

Most of these are on rare paths; the last two affect every run, though only memory matters, and only in a long chat. No eval run failed, and no hand-written run reported a model other than the one asked for.

- **Fallback models.** With the server-side fallback beta, the SDK's types say a fallback in the middle of a turn is marked with a `fallback` block. LangChain drops content blocks it doesn't know, so the LangGraph loop would record the wrong model for that turn and leave the block out of the next request, which the API may reject. Fixing it means overriding a private LangChain method.
- **Token counts.** LangChain reads them only from the `message_delta` event near the end of the stream. The API repeats the full counts there, so they match, but a stream that ends before that event counts zero.
- **Rejected tool input.** The SDK stops reading at the first piece of input it can't parse, and the hand-written loop runs the turn again. LangGraph reads the whole turn first, so it prints all of it and waits longer.
- **A response with no events.** LangChain raises a `ValueError`, which the LangGraph loop treats like unparseable tool input: it asks twice more, then reports an error. The hand-written loop raises an `AssertionError` out of `ask`.
- **Empty text.** LangChain leaves empty or whitespace-only text, and assistant turns left with nothing, out of the history it sends back. The hand-written loop sends them unchanged, which the API may reject on the next request.
- **Printing.** Text or a progress note that arrives in a block's start event is printed, and passed to `on_event`, by the LangGraph loop only.
- **Memory.** The LangGraph loop keeps a full copy of the conversation for every question asked: 2.5 MB for 0.4 MB of conversation after 12 questions with charts. The hand-written loop keeps one list.
- **Harmless:** `"is_error": false` on successful chart results (the API's default), two user messages in a row merged into one (after the step cap or a dropped connection), a different key order in the request JSON, LangChain's own User-Agent and `X-Stainless-*` headers, one connection pool shared by all LangGraph agents in a process, and a connect timeout of 600 seconds instead of 5, so a hung connection waits longer before a retry.

## Found in both loops (not fixed)

Fixing these means changing agent.py, which this comparison left alone:

- A stream that ends before its `message_delta` event, so with no stop reason, is taken as a complete turn: its text becomes the answer, or its tool calls run.
- A connection dropped mid-stream raises an error out of `ask` instead of returning one.
- A turn run again because of unparseable tool input is billed but not counted in the token totals.
- A fallback reported only in the usage details isn't noticed.

## How the predictions scored

Claude made the predictions, and Ak agreed with all of them, so there is one set to score.

| Prediction | Result | Right? |
|---|---|---|
| Coffee: 22 of 22 on every run, for both loops | 22 of 22 for both | Yes |
| Retail: 15 of 15 on every run, for both loops | 15 of 15 for both | Yes |
| Model turns: no gap beyond noise (±0.37 coffee, ±0.29 retail) | −0.02 and −0.03 | Yes |
| Tool calls: no gap beyond noise (±0.41, ±0.29) | −0.07 and 0.00 | Yes |
| Agent cost: no gap beyond noise (±18%, ±24%) | +7.5% and −10.9% on all runs; −1.7% and +2.3% without the first batches | Yes. The all-runs gaps are real but come from run order |
| Time: no gap beyond noise (±1.5 s, ±2.2 s) | −0.4 s and +0.2 s | Yes |
| LangGraph's code is 10–30% longer | 2.3 times as long | No |
| The smoke test passes on the first try: about even odds | It passed; the bugs turned up later, in the offline replay of rare cases | Not scorable |

The miss was the code. The loop and the graph alone are 92 lines, 39% more than 66. The prediction named the translation helpers but not their size, and later fixes (`read_tool_input`, the callback printer) brought them to 58 lines.

## Predictions

Written on 2026-11-04, before any paid run.

The saved runs closest to today's setup are coffee [v2](../.claude/hillclimb/analyst-answers/v2/results.jsonl) and retail [baseline](../.claude/hillclimb/retail-answers/baseline/results.jsonl). Both used Sonnet 5.5, high effort, code run on this computer, and the hand-written loop. The predictions compare LangGraph with a re-run of the hand-written loop on the same day, not with these old runs.

| Per answer unless stated | Saved runs (Oct 10), coffee / retail | Noise: a smaller gap could be luck | Claude predicts | Ak predicts |
|---|---|---|---|---|
| Coffee questions passed on every run | 22 of 22 | 1 question | 22 of 22 for both loops | agrees with Claude |
| Retail questions passed on every run | 15 of 15 | 1 question | 15 of 15 for both loops | agrees with Claude |
| Model turns | 3.5 / 2.6 | ±0.37 / ±0.29 | no gap beyond noise | agrees with Claude |
| Tool calls | 2.6 / 1.6 | ±0.41 / ±0.29 | no gap beyond noise | agrees with Claude |
| Agent cost | $0.020 / $0.017 | ±18% / ±24% | no gap beyond noise | agrees with Claude |
| Time | 10 s / 9 s | ±1.5 s / ±2.2 s | no gap beyond noise | agrees with Claude |
| Lines of code only one loop needs | | | LangGraph's is 10–30% longer | agrees with Claude |
| The smoke test passes on the first try | | | about even odds; any fix most likely in streaming | agrees with Claude |

Why:

- **Behaviour should match.** The first request is identical, and a test checks it. The model can't tell which loop sent a request, so pass rate, turns and tool calls should differ only by chance. Cost follows the tokens, so it should match too.
- **What could still make LangGraph differ:**
  1. LangChain builds the later turns of the conversation itself, from its own message objects, including the thinking blocks and chart images. If those differ from what the hand-written loop sends, the model sees a slightly different conversation, and cache hits may drop.
  2. Each step adds a little time: converting messages and saving a checkpoint. That is milliseconds, far below the time noise, so this eval can't measure it.
- **The code should be longer.** Translating between LangChain's message format and the API's needs extra code that the hand-written loop doesn't (`api_usage`, `api_message`, `_print_stream`). The graph itself is only 7 lines.
- **The smoke test is the first real stream.** So far every LangGraph test has used made-up stream chunks.
- **How the noise bands were set:** each is twice the standard error of the gap between two means, using the spread in the saved runs, with 44 coffee or 30 retail runs per loop. The bands are cautious: both loops answer the same questions, so comparing question by question cancels out how hard each question is. The 1-question band for pass rates is a judgment call: one flaky question in one loop is luck, but two or more would point to a bug.

## Runs and spend cap

The expected costs come from the saved runs: $0.024 per coffee answer and $0.020 per retail answer, with grading included.

| Order | Run | Command | Runs | Expected cost |
|---|---|---|---|---|
| 1 | Smoke test | `uv run python graph_agent.py data/coffee_sales.csv -q "<a question that needs 2+ tool calls>"` | 1 | $0.03 |
| 2 | LangGraph, coffee, first run of each question | `uv run python evals/run_eval.py --loop langgraph --variant langgraph-v1 --reps 1` | 22 | $0.53 |
| 3 | LangGraph, coffee, second run of each question | the same command with `--reps 2` (it adds only the missing runs) | 22 | $0.53 |
| 4 | Hand-written, coffee | `uv run python evals/run_eval.py --variant v5` | 44 | $1.05 |
| 5 | LangGraph, retail | `uv run python evals/run_eval.py --suite retail --loop langgraph --variant langgraph-v1` | 30 | $0.60 |
| 6 | Hand-written, retail | `uv run python evals/run_eval.py --suite retail --variant v2` | 30 | $0.60 |
| | | | | **$3.34** |

**Why re-run the hand-written loop:**
- API speed changes from day to day, and the saved runs are 3.5 weeks old.
- agent.py has changed since those runs.
- The saved coffee runs ran each question 2, 3 or 6 times.

Running both loops on the same day, with the same number of runs per question, leaves the loop as the only difference. The runs alternate between the loops, so a slow hour hits both.

**Spend cap: $6 in total,** counting the agent, the grader and the smoke test. Rules:

1. **If the smoke test shows any problem, stop before any batch.** Problems are: output in the wrong order, no cache reads after the first turn, missing token counts, or an error. Fix it offline first.
2. **After run 2, stop and look if** the agent's coffee cost passes $0.03 per answer (1.5 times the saved runs), or under half the input tokens come from the cache.
3. **If the total passes $4.50, ask Ak before the next run.**
4. **Never start a run whose expected cost would take the total past $6.**

## What ran

Every run used Sonnet 5.5 at high effort, with code run on this computer. No run failed, so none had to be repeated.

**The smoke test (2026-11-04, $0.03)** passed on the first try. Its recorded stream was replayed into the hand-written loop. Both loops sent requests with the same content, gave the same answer and token counts, and printed the same output, apart from one field: LangChain sends `"is_error": false` on a successful chart result, which is the API's default.

**The batches (2026-11-05), in the order they ran:**

| Loop and suite | Results folder | Runs | Passed | Agent cost per answer | Input read from the cache | Spend, grading included |
|---|---|---|---|---|---|---|
| LangGraph, coffee, first run of each question | `analyst-answers/langgraph-v1` | 22 | 22 | $0.0217 | 62% | $0.55 |
| LangGraph, coffee, second run | `analyst-answers/langgraph-v1` | 22 | 22 | $0.0183 | 78% | $0.48 |
| Hand-written, coffee | `analyst-answers/v5` | 44 | 44 | $0.0186 | 80% | $0.97 |
| Hand-written, retail, first run | `retail-answers/v2` | 15 | 15 | $0.0172 | 51% | $0.31 |
| LangGraph, retail | `retail-answers/langgraph-v1` | 30 | 30 | $0.0136 | 81% | $0.51 |
| Hand-written, retail, second run | `retail-answers/v2` | 15 | 15 | $0.0133 | 82% | $0.25 |
| | | | | | **Total, smoke test included** | **$3.10** |

**The first batch on each suite paid to fill the prompt cache.**
- A question's first request has the same content in both loops, and the API keeps it cached for 5 minutes after each use. The first time each question was asked, its first turn wrote about 1,500 tokens to the cache (1,600 on retail); every later run of it, by either loop, read them back.
- LangGraph asked the coffee questions first: its first run read 62% of its input from the cache. Its second run read 78%, close to the 80% of the hand-written coffee runs.
- So the retail order changed from the plan: the hand-written loop's first run went before LangGraph's, not after, and each loop paid for the cache on one suite.
- Without those first batches, the loops cost the same: coffee $0.0183 LangGraph vs $0.0186 hand-written; retail $0.0136 vs $0.0133.

graph_agent.py changed after these runs in three ways, none of which could have changed these runs' requests, answers or token counts: an error while printing now stops the turn (the evals print nothing), a turn retried after a broken stream finishes its printed line first (printing only), and the conversation is saved once per question (storage only). After these changes the tests pass, and a fresh offline replay shows only the differences listed under [Where the loops still differ](#where-the-loops-still-differ).
