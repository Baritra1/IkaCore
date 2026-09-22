"""Deterministic in-process LLM provider for benchmarking IkaCore.

The fake is installed underneath httpx (``HTTPTransport.handle_request`` and
``AsyncHTTPTransport.handle_async_request``), so every line of IkaCore and
httpx client code runs exactly as it would against a real provider: payload
building, JSON encoding, retries/status handling, response parsing, SSE
collection for Codex, and so on. Only the socket is removed.

Like a real model, the fake is *stateless*: the reply is a pure function of
the request it receives (which agent is asking, which tools are offered, and
which tool calls already appear in the conversation). That keeps scenarios
deterministic under async fan-out and parallel workflow instances, which
share API keys and interleave arbitrarily.

Time spent inside the fake (request parsing, policy, response construction)
is accumulated so the harness can report framework time net of the fake.
"""

from __future__ import annotations

import functools
import itertools
import json
import re
import sys
import threading
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

# --------------------------------------------------------------------------
# Scripted model output
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Turn:
    """One model response: optional text plus zero or more tool calls."""

    text: str = ""
    calls: tuple[Call, ...] = ()


def end(text: str) -> Turn:
    return Turn(calls=(Call("agent_end", {"input": text}),))


def call(name: str, **args: Any) -> Turn:
    return Turn(calls=(Call(name, args),))


@dataclass
class RequestView:
    """Provider-neutral view of an outgoing LLM request."""

    agent: str  # derived from the api key: "bench-<agent>"
    provider: str
    tool_names: frozenset[str]
    prior_calls: list[str]  # names of assistant tool calls already in the conversation
    raw: str  # request body text, for marker checks

    def count(self, *names: str) -> int:
        wanted = set(names)
        return sum(1 for name in self.prior_calls if name in wanted)


Policy = Callable[[RequestView], Turn]


def plan_policy(turns: list[Turn]) -> Policy:
    """Replay ``turns`` in order, locating progress from prior tool calls.

    Progress is the number of prior calls whose names appear anywhere in the
    plan, so each plan should use tool names distinct from other plans that
    can share a conversation (e.g. per-stage tools).
    """
    names = {c.name for t in turns for c in t.calls}
    cumulative: list[int] = []
    total = 0
    for t in turns:
        cumulative.append(total)
        total += len(t.calls)

    def policy(view: RequestView) -> Turn:
        done = view.count(*names)
        for idx in range(len(turns) - 1, -1, -1):
            if done >= cumulative[idx]:
                return turns[idx]
        return turns[0]

    return policy


SUMMARY_TEXT = "Summary: the work so far gathered the requested facts and is ready to be finalised."


# --------------------------------------------------------------------------
# Request parsing (provider -> RequestView)
# --------------------------------------------------------------------------


def _provider_for(url: httpx.URL) -> str:
    host, path = url.host, url.path
    if "/backend-api/codex/" in path:
        return "codex"
    if "anthropic" in host:
        return "anthropic"
    if "generativelanguage" in host:
        return "gemini"
    if "openrouter" in host:
        return "openrouter"
    if "deepseek" in host:
        return "deepseek"
    if path.endswith("/responses"):
        return "openai_responses"
    return "openai"


def _api_key(request: httpx.Request) -> str:
    headers = request.headers
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return headers.get("x-api-key") or headers.get("x-goog-api-key") or request.url.params.get("key", "")


def _tool_names(provider: str, body: dict[str, Any]) -> frozenset[str]:
    tools = body.get("tools") or []
    names: list[str] = []
    for tool in tools:
        if provider == "gemini":
            names.extend(fd.get("name", "") for fd in tool.get("functionDeclarations", []) or [])
        elif "function" in tool and isinstance(tool["function"], dict):
            names.append(tool["function"].get("name", ""))
        else:
            names.append(tool.get("name", ""))
    return frozenset(names)


@functools.lru_cache(maxsize=65536)
def _canonical_arg_string(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw or "{}"), sort_keys=True)
    except ValueError:
        return raw


def _canonical_args(raw: Any) -> str:
    # Memoized for string arguments: every request re-lists the whole history,
    # and re-parsing it would make the fake's own cost quadratic.
    if isinstance(raw, str):
        return _canonical_arg_string(raw)
    return json.dumps(raw, sort_keys=True)


def _raw_prior_calls(provider: str, body: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    if provider in ("openai_responses", "codex"):
        return [(i.get("name", ""), i.get("arguments")) for i in body.get("input", []) or [] if i.get("type") == "function_call"]
    if provider == "anthropic":
        return [
            (block.get("name", ""), block.get("input"))
            for msg in body.get("messages", []) or []
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list)
            for block in msg["content"]
            if block.get("type") == "tool_use"
        ]
    if provider == "gemini":
        return [
            (part["functionCall"].get("name", ""), part["functionCall"].get("args"))
            for content in body.get("contents", []) or []
            for part in content.get("parts", []) or []
            if "functionCall" in part
        ]
    return [
        (tc.get("function", {}).get("name", ""), tc.get("function", {}).get("arguments"))
        for msg in body.get("messages", []) or []
        if msg.get("role") == "assistant"
        for tc in msg.get("tool_calls", []) or []
    ]


def _prior_calls(provider: str, body: dict[str, Any]) -> list[str]:
    """Names of distinct assistant tool calls in the conversation, in first-seen order.

    Calls are de-duplicated by (name, arguments) so that a provider path which
    replays the same round twice (currently Anthropic and Gemini) still
    progresses through the same plan as every other provider; the duplication
    itself remains visible in the recorded payloads.
    """
    seen: set[tuple[str, str]] = set()
    names: list[str] = []
    for name, args in _raw_prior_calls(provider, body):
        key = (name, _canonical_args(args))
        if key not in seen:
            seen.add(key)
            names.append(name)
    return names


def build_view(request: Any) -> tuple[RequestView, dict[str, Any]]:
    raw = request.content.decode("utf-8")
    body = json.loads(raw) if raw else {}
    provider = _provider_for(request.url)
    key = _api_key(request)
    agent = key[len("bench-"):] if key.startswith("bench-") else key
    view = RequestView(
        agent=agent,
        provider=provider,
        tool_names=_tool_names(provider, body),
        prior_calls=_prior_calls(provider, body),
        raw=raw,
    )
    return view, body


# --------------------------------------------------------------------------
# Response encoding (Turn -> provider-native body)
# --------------------------------------------------------------------------

_USAGE_IN, _USAGE_OUT = 120, 30


def _call_id(offset: int, idx: int, c: Call) -> str:
    # Deterministic across processes and unique within a conversation.
    digest = zlib.crc32(f"{c.name}:{json.dumps(c.args, sort_keys=True)}".encode())
    return f"call_{offset + idx}_{digest:08x}"


def _encode_responses(model: str, turn: Turn, offset: int) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if turn.text:
        output.append({
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": turn.text, "annotations": []}],
        })
    for idx, c in enumerate(turn.calls):
        cid = _call_id(offset, idx, c)
        output.append({
            "type": "function_call", "id": f"fc_{cid}", "call_id": cid, "name": c.name,
            "arguments": json.dumps(c.args), "status": "completed",
        })
    return {
        "id": "resp_bench", "object": "response", "status": "completed", "model": model, "output": output,
        "usage": {"input_tokens": _USAGE_IN, "output_tokens": _USAGE_OUT, "total_tokens": _USAGE_IN + _USAGE_OUT},
    }


def _encode_chat(model: str, turn: Turn, offset: int) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": turn.text or None}
    if turn.calls:
        message["tool_calls"] = [
            {"id": _call_id(offset, idx, c), "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.args)}}
            for idx, c in enumerate(turn.calls)
        ]
    return {
        "id": "chatcmpl-bench", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if turn.calls else "stop"}],
        "usage": {"prompt_tokens": _USAGE_IN, "completion_tokens": _USAGE_OUT, "total_tokens": _USAGE_IN + _USAGE_OUT},
    }


def _encode_anthropic(model: str, turn: Turn, offset: int) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": turn.text}] if turn.text else []
    content.extend(
        {"type": "tool_use", "id": _call_id(offset, idx, c), "name": c.name, "input": c.args}
        for idx, c in enumerate(turn.calls)
    )
    return {
        "id": "msg_bench", "type": "message", "role": "assistant", "model": model, "content": content,
        "stop_reason": "tool_use" if turn.calls else "end_turn",
        "usage": {"input_tokens": _USAGE_IN, "output_tokens": _USAGE_OUT},
    }


def _encode_gemini(model: str, turn: Turn, offset: int) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": turn.text}] if turn.text else []
    parts.extend({"functionCall": {"name": c.name, "args": c.args}} for c in turn.calls)
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {
            "promptTokenCount": _USAGE_IN, "candidatesTokenCount": _USAGE_OUT, "totalTokenCount": _USAGE_IN + _USAGE_OUT,
        },
        "modelVersion": model,
    }


def _encode_codex_sse(model: str, turn: Turn, offset: int) -> bytes:
    body = _encode_responses(model, turn, offset)
    frames: list[str] = []

    def frame(event: str, data: dict[str, Any]) -> None:
        frames.append(f"event: {event}\ndata: {json.dumps(data)}\n\n")

    frame("response.created", {"type": "response.created", "response": {"id": body["id"], "model": model}})
    for idx, item in enumerate(body["output"]):
        frame("response.output_item.done", {"type": "response.output_item.done", "output_index": idx, "item": item})
    completed = dict(body, output=[])
    frame("response.completed", {"type": "response.completed", "response": completed})
    return "".join(frames).encode("utf-8")


_ENCODERS: dict[str, Callable[[str, Turn, int], dict[str, Any]]] = {
    "openai_responses": _encode_responses,
    "openai": _encode_chat,
    "deepseek": _encode_chat,
    "openrouter": _encode_chat,
    "anthropic": _encode_anthropic,
    "gemini": _encode_gemini,
}


def encode(provider: str, model: str, turn: Turn, offset: int) -> tuple[bytes, str]:
    if provider == "codex":
        return _encode_codex_sse(model, turn, offset), "text/event-stream"
    return json.dumps(_ENCODERS[provider](model, turn, offset)).encode("utf-8"), "application/json"


# --------------------------------------------------------------------------
# The fake server
# --------------------------------------------------------------------------

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\b[0-9a-f]{32}\b")


def normalize_text(text: str) -> str:
    """Strip values that legitimately differ between runs (uuids, hex ids)."""
    return _UUID_RE.sub("<id>", text)


@dataclass
class RecordedRequest:
    agent: str
    provider: str
    path: str
    body: str  # normalized, key-sorted JSON


@dataclass(frozen=True)
class Fault:
    """A scripted failure: an HTTP error response, or a raised httpx exception when ``exc`` is set."""

    status: int = 500
    body: Any = None  # dict -> JSON body, str -> text body
    headers: tuple[tuple[str, str], ...] = ()
    exc: Optional[str] = None  # httpx exception class name, e.g. "ReadTimeout", "ConnectError"


FaultPlan = Callable[[RequestView, int], Optional[Fault]]  # (request, attempt index for that agent) -> fault


@dataclass
class FakeLLM:
    """Routes each request to a per-agent policy and returns a canned response."""

    policies: dict[str, Policy] = field(default_factory=dict)
    default_policy: Optional[Policy] = None
    record: bool = False
    recorded: list[RecordedRequest] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    calls: int = 0
    toolless_calls: int = 0  # summarization / force-answer requests (sent without tools)
    bytes_sent: int = 0
    fake_seconds: float = 0.0
    # Transports (connection pools) that carried at least one request since the last reset.
    # Each is at least one TCP+TLS handshake on a real network. This is exact while every
    # client has at most one request in flight, which holds for IkaCore's per-agent/per-call clients.
    transports_used: set[int] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cache: dict[tuple[str, str, str, int], tuple[bytes, str]] = field(default_factory=dict)
    faults: Optional[FaultPlan] = None
    # Simulated model latency per request (seconds). Slept outside the lock so concurrent
    # requests overlap, and not counted in fake_seconds (it stands in for the model, not the fake).
    latency: Optional[Callable[[RequestView], float]] = None
    attempts: dict[str, int] = field(default_factory=dict)  # every request per agent, failed ones included
    faults_served: int = 0

    def reset_counters(self) -> None:
        self.calls = 0
        self.toolless_calls = 0
        self.bytes_sent = 0
        self.transports_used = set()
        self.fake_seconds = 0.0
        self.attempts = {}
        self.faults_served = 0
        self.recorded = []
        self.errors = []

    def _turn_for(self, view: RequestView) -> Turn:
        if not view.tool_names:
            # Summarization / force-answer requests are sent without tools.
            return Turn(text=SUMMARY_TEXT)
        policy = self.policies.get(view.agent, self.default_policy)
        if policy is None:
            self.errors.append(f"no policy for agent {view.agent!r}")
            return end("NO POLICY")
        return policy(view)

    @property
    def connections(self) -> int:
        return len(self.transports_used)

    def respond(self, request: Any, transport: Optional[object] = None, response_cls: Any = httpx.Response) -> Any:
        start = time.perf_counter()
        view, body = build_view(request)
        if self.faults is not None:
            with self._lock:
                index = self.attempts.get(view.agent, 0)
                self.attempts[view.agent] = index + 1
            fault = self.faults(view, index)
            if fault is not None:
                with self._lock:
                    self.faults_served += 1
                return _fault_response(fault, request, response_cls)
        turn = self._turn_for(view)
        model = str(body.get("model", "bench-model"))
        offset = len(view.prior_calls)
        key = (view.provider, model, repr(turn), offset)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._cache[key] = encode(view.provider, model, turn, offset)
        content, content_type = cached
        if self.record:
            recorded = RecordedRequest(
                agent=view.agent,
                provider=view.provider,
                path=request.url.path,
                body=normalize_text(json.dumps(body, sort_keys=True)),
            )
        response = response_cls(200, content=content, headers={"content-type": content_type}, request=request)
        delay = self.latency(view) if self.latency is not None else 0.0
        with self._lock:
            self.calls += 1
            self.bytes_sent += len(request.content)
            if not view.tool_names:
                self.toolless_calls += 1
            if transport is not None:
                self.transports_used.add(_transport_token(transport))
            if self.record:
                self.recorded.append(recorded)
            self.fake_seconds += time.perf_counter() - start
        if delay > 0:
            time.sleep(delay)
        return response


def _fault_response(fault: Fault, request: Any, response_cls: Any) -> Any:
    module = sys.modules[response_cls.__module__.split(".")[0]]
    if fault.exc:
        raise getattr(module, fault.exc)(f"injected {fault.exc}", request=request)
    headers = dict(fault.headers)
    if isinstance(fault.body, str):
        return response_cls(fault.status, text=fault.body, headers=headers, request=request)
    return response_cls(fault.status, json=fault.body or {"error": {"message": "injected"}}, headers=headers, request=request)


# --------------------------------------------------------------------------
# Transport installation
# --------------------------------------------------------------------------

_ACTIVE: Optional[FakeLLM] = None
_ORIGINALS: dict[str, Any] = {}
_TOKENS = itertools.count(1)
_TOKEN_ATTR = "_ikabench_transport_token"


def _transport_token(transport: object) -> int:
    # id() can be recycled after a short-lived client is garbage collected, so tag the object.
    token = getattr(transport, _TOKEN_ATTR, None)
    if token is None:
        token = next(_TOKENS)
        setattr(transport, _TOKEN_ATTR, token)
    return token


def _http_modules() -> list[Any]:
    """httpx plus its ``httpx2`` fork (used by openai>=3 / anthropic>=1), when installed."""
    modules: list[Any] = [httpx]
    try:
        import httpx2  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        modules.append(httpx2)
    return modules


def _handlers(module: Any) -> tuple[Any, Any]:
    response_cls = module.Response

    def sync_handle(self: Any, request: Any) -> Any:
        if _ACTIVE is None:
            raise RuntimeError("benchmark transport used without an active FakeLLM")
        request.read()
        return _ACTIVE.respond(request, self, response_cls)

    async def async_handle(self: Any, request: Any) -> Any:
        if _ACTIVE is None:
            raise RuntimeError("benchmark transport used without an active FakeLLM")
        await request.aread()
        return _ACTIVE.respond(request, self, response_cls)

    return sync_handle, async_handle


def install() -> None:
    """Patch every installed httpx flavour so no request can reach the network."""
    if _ORIGINALS:
        return
    for module in _http_modules():
        sync_handle, async_handle = _handlers(module)
        _ORIGINALS[module.__name__] = (module.HTTPTransport.handle_request, module.AsyncHTTPTransport.handle_async_request)
        module.HTTPTransport.handle_request = sync_handle
        module.AsyncHTTPTransport.handle_async_request = async_handle


def uninstall() -> None:
    for module in _http_modules():
        originals = _ORIGINALS.pop(module.__name__, None)
        if originals is not None:
            module.HTTPTransport.handle_request, module.AsyncHTTPTransport.handle_async_request = originals


def activate(fake: Optional[FakeLLM]) -> None:
    global _ACTIVE
    _ACTIVE = fake


def recorded_by_agent(recorded: Iterable[RecordedRequest]) -> dict[str, list[RecordedRequest]]:
    grouped: dict[str, list[RecordedRequest]] = {}
    for item in recorded:
        grouped.setdefault(item.agent, []).append(item)
    return grouped
