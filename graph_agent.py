"""The data analyst agent with its loop rebuilt in LangGraph, to compare with the hand-written one.

Everything except the loop comes from agent.py: the request settings, system prompt, tools, sandbox and
tool code. So the two can be compared on the same evals with the loop as the only difference.

    START -> model --(route)--> tools -> model -> ... -> END

Usage, the same as agent.py, with the langgraph dependency group:
    uv run --group langgraph python graph_agent.py data/coffee_sales.csv
    uv run --group langgraph python graph_agent.py data/coffee_sales.csv -q "Which store makes the most money?"
"""

from types import SimpleNamespace

import anthropic
import httpx2
from jiter import from_json
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

import agent
from agent import BETAS, DIM, MAX_STEPS, MODEL, RED, RESET, TOOLS, Agent, Answer, StreamPrinter


class GraphAgent(Agent):
    """An Agent whose ask() runs a LangGraph graph instead of the hand-written loop."""

    def __init__(self, csv_path, out_dir, model=MODEL, effort="high", quiet=False, on_event=None, max_retries=2):
        super().__init__(csv_path, out_dir, model, effort, quiet, on_event)
        # The same request as Agent._call_model, sent through LangChain's wrapper for the Messages API.
        self.llm = ChatAnthropic(
            model=model,
            max_tokens=64000,
            max_retries=max_retries,
            timeout=600,  # the SDK's default read timeout; LangChain's default is no timeout at all
            streaming=True,
            betas=BETAS,
            thinking={"type": "adaptive", "display": "updates"},
            output_config={"effort": effort},
            model_kwargs={"fallbacks": "default", "cache_control": {"type": "ephemeral"}},
        ).bind_tools(TOOLS)
        self.graph = build_graph(self)
        # One conversation (thread) per agent. A step is one node run, so MAX_STEPS model turns take 2 * MAX_STEPS.
        self.config = {"configurable": {"thread_id": "chat"}, "recursion_limit": 2 * MAX_STEPS}

    def call_model(self, state: MessagesState) -> dict:
        """The model node: one model turn, like Agent._call_model."""
        for _ in range(3):
            try:
                response = self.llm.invoke([SystemMessage(self.system), *state["messages"]])
                read_tool_input(response)
                if response.response_metadata.get("stop_reason") is None:  # no message_delta, as in _call_model
                    raise RuntimeError("The response ended before the message was complete.")
                return {"messages": [response]}
            except ValueError:  # from the stream or from tool input the SDK's parser rejects: run the turn again,
                pass            # as _call_model does
        raise RuntimeError("The model's tool input couldn't be parsed after 3 tries.")

    def run_tools(self, state: MessagesState) -> dict:
        """The tools node: run every tool call from the last turn, in order, with Agent's own tool code."""
        turn = state["messages"][-1]
        cut_off = turn.response_metadata.get("stop_reason") == "max_tokens"
        results = []
        for call in turn.tool_calls:
            block = SimpleNamespace(id=call["id"], name=call["name"], input=call["args"])
            if cut_off:  # a tool input cut off mid-stream still parses as a valid-looking object. Don't run it.
                result = self._error(block, "Your tool input was cut off at the output limit. Send a shorter one.")
            else:
                result = self._run_tool(block)
            status = "error" if result.get("is_error") else "success"
            results.append(ToolMessage(result["content"], tool_call_id=call["id"], status=status))
        return {"messages": results}

    def ask(self, question: str) -> Answer:
        """Answer one question: run the graph on it, then read what happened from the checkpointer."""
        before = self.graph.get_state(self.config).values.get("messages", [])
        answer = Answer()
        try:
            # invoke runs each node on this thread, so Ctrl+C stops a model call or tool at once, as in Agent.ask.
            # (stream_mode="messages" would run nodes in a worker thread that Ctrl+C can't stop.)
            # durability="exit" saves the conversation once per question. By default every step saves a full copy.
            self.graph.invoke({"messages": [HumanMessage(question)]},
                              {**self.config, "callbacks": [TurnPrinter(self.out, self.on_event)]}, durability="exit")
        except GraphRecursionError:  # before RuntimeError, which it subclasses
            answer.status = "max_steps"
        except KeyboardInterrupt:
            answer.status = "cancelled"
        except (anthropic.APIError, httpx2.TransportError, RuntimeError) as exc:  # the same errors as Agent.ask
            # LangChain re-raises SDK errors as its own subclasses; report the SDK's, as Agent.ask does.
            exc = exc.__cause__ if isinstance(exc.__cause__, anthropic.APIError) else exc
            answer.status, answer.error, message = "error", f"{type(exc).__name__}: {exc}", str(exc)

        new = self.graph.get_state(self.config).values.get("messages", [])[len(before):]  # this question's messages
        for m in new:
            if m.type == "ai":
                answer.usage.add(api_usage(m))
                # LangChain drops usage.iterations, so this misses a fallback named only there (docs/LANGGRAPH.md).
                answer.models.append(m.response_metadata.get("model_name"))
            elif m.type == "tool":
                answer.usage.tool_calls += 1

        stop_reason = new[-1].response_metadata.get("stop_reason") if answer.status == "ok" else None
        if stop_reason == "refusal":
            answer.status = "refused"
            self.out(f"\n{RED}The model declined this request.{RESET}")
        elif answer.status == "ok":
            answer.text = str(new[-1].text)  # .text is LangChain's str subclass; the hand-written loop returns a str
            if stop_reason == "max_tokens":
                answer.status = "truncated"
                self.out(f"{RED}[The answer was cut off at the output limit.]{RESET}")
            self.out(f"\n{DIM}({answer.usage}){RESET}")
        elif answer.status == "max_steps":
            self.out(f"\n{RED}Stopped after {MAX_STEPS} steps without a final answer.{RESET}")
        elif answer.status == "cancelled":
            self.out(f"\n{DIM}[cancelled]{RESET}")
        else:
            self.out(f"\n{RED}Error: {message}{RESET}")

        if answer.status in ("refused", "cancelled", "error"):
            self._roll_back(before)  # so later questions still work
        self.messages = [api_message(m) for m in self.graph.get_state(self.config).values.get("messages", [])]
        return answer

    def _roll_back(self, before: list) -> None:
        """Put the saved conversation back as it was before this question, like `del self.messages[start:]`."""
        # An update runs the edges out of the node it's credited to. START's only edge is the fixed one to the
        # model, so the router never sees the edited list (it would fail on an empty one).
        reset = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *before]
        self.graph.update_state(self.config, {"messages": reset}, as_node=START)


class TurnPrinter(BaseCallbackHandler):
    """Prints each model turn as it streams, with agent.py's StreamPrinter. LangChain hands every streamed chunk
    to this callback; the chunks are turned back into the SDK stream events the printer reads, with a new block
    wherever the block index changes."""

    raise_error = True  # an error while printing stops the turn, as it does in Agent._call_model

    def __init__(self, out, on_event):
        self.out, self.on_event = out, on_event

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:  # a model turn, or a retry of one
        self.printer, self.block = StreamPrinter(self.out, self.on_event), -1

    def on_llm_new_token(self, token, *, chunk=None, **kwargs) -> None:
        content = chunk.message.content if chunk is not None else None
        if not isinstance(content, list):
            return
        for part in content:
            if part.get("index") != self.block:
                if self.block != -1:
                    self.printer.handle(SimpleNamespace(type="content_block_stop"))
                start = SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type=part["type"]))
                self.printer.handle(start)
                self.block = part.get("index")
            if part["type"] == "text":
                self.printer.handle(SimpleNamespace(type="text", text=part["text"]))
            elif part["type"] == "thinking":
                self.printer.handle(SimpleNamespace(type="thinking", thinking=part.get("thinking", "")))

    def on_llm_end(self, response, **kwargs) -> None:  # the whole turn arrived; Agent._call_model finishes here too
        self.printer.finish()

    def on_llm_error(self, error, **kwargs) -> None:  # the turn failed or will run again: finish, as _call_model does
        self.printer.finish()


def route(state: MessagesState) -> str:
    """The conditional edge after the model node: return the name of the node to run next.
    "tools" if the last turn asked for tools, otherwise END (the answer is done). A refused turn ends the run
    even when it holds tool calls, as in Agent.ask, which checks for a refusal first."""
    last = state["messages"][-1]
    return "tools" if last.tool_calls and last.response_metadata.get("stop_reason") != "refusal" else END


def build_graph(agent: GraphAgent):
    """Wire the nodes into a graph: START -> model, model -> route, tools -> model.
    Compile it with a checkpointer, so the conversation is kept between questions."""
    graph = StateGraph(MessagesState)
    graph.add_node("model", agent.call_model)
    graph.add_node("tools", agent.run_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=InMemorySaver())


def read_tool_input(message) -> None:
    """Parse each tool call's streamed input with the call the SDK makes (jiter, partial mode), so both loops run
    the same code for the same stream. LangChain's own parser disagrees with it both ways: it closes a string cut
    off mid-stream (so cut-off code would run) and rejects some input the SDK accepts. Raises ValueError where the
    SDK does."""
    content = message.content if isinstance(message.content, list) else []
    blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
    for block in blocks:
        if block.get("partial_json"):
            block["input"] = from_json(block["partial_json"].encode(), partial_mode=True)
    message.tool_calls = [{"type": "tool_call", "id": b["id"], "name": b["name"], "args": b["input"]} for b in blocks]
    message.invalid_tool_calls = []


def api_usage(message) -> SimpleNamespace:
    """A turn's token counts in the API's shape, for agent.Usage.add."""
    usage = message.usage_metadata or {}
    details = usage.get("input_token_details", {})
    cache_read = details.get("cache_read") or 0
    # When the API splits cache writes by lifetime, LangChain reports them there and sets cache_creation to 0.
    split = (details.get("ephemeral_5m_input_tokens") or 0) + (details.get("ephemeral_1h_input_tokens") or 0)
    cache_write = split or details.get("cache_creation") or 0
    return SimpleNamespace(
        input_tokens=usage.get("input_tokens", 0) - cache_read - cache_write,  # LangChain's total includes the cache
        output_tokens=usage.get("output_tokens", 0),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def api_message(message) -> dict:
    """One LangChain message in the API's shape, for the traces evals/run_eval.py writes."""
    if message.type == "human":
        return {"role": "user", "content": message.content}
    if message.type == "tool":
        result = {"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content,
                  "is_error": message.status == "error"}
        return {"role": "user", "content": [result]}
    # Text and tool calls in the order the model sent them. Traces leave out thinking blocks anyway.
    blocks = message.content if isinstance(message.content, list) else [{"type": "text", "text": message.content}]
    content = [{"type": "text", "text": b["text"]} if b["type"] == "text" else
               {"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["input"]}
               for b in blocks if isinstance(b, dict) and b.get("type") in ("text", "tool_use")]
    return {"role": "assistant", "content": content}


if __name__ == "__main__":
    agent.Agent = GraphAgent  # agent.py's command line, with this loop in place of the hand-written one
    agent.main()
