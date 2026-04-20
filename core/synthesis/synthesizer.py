"""
core/synthesis/synthesizer.py
──────────────────────────────
The Synthesis — LLM Integration

This is where Theory (Mirror) + Reality (Probe) → Understanding.

The synthesizer takes a HybridResult (code context from the Mirror)
and optionally a list of LiveSnapshots (runtime state from the Probe),
then constructs a prompt and calls an LLM to generate a coherent explanation
of what the kernel is doing and *why*.

The prompt engineering here is deliberately structured:
  1. System role: orient the LLM as a kernel expert
  2. Theory block: static code context (functions, structs, call chains)
  3. Reality block: live drgn snapshots (actual field values)
  4. Question: the user's natural language query
  5. Instruction: explain the synthesis — don't just repeat the code

Backend-agnostic: supports Ollama (local) and any OpenAI-compatible API.
Ollama is the default because it's the right choice for a privacy-first,
offline-capable system. No API keys. No data egress. Just the model.

Recommended models (via Ollama):
  deepseek-coder:6.7b  — Code-specialized, fast on consumer hardware
  codellama:13b        — Good reasoning about C code
  llama3:8b-instruct   — Strong general reasoning, understands systems
  qwen2.5-coder:7b     — Very strong at code, multilingual
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..mirror.store import HybridResult
from ..probe.drgn_bridge import LiveSnapshot


# ─── Synthesis Result ─────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    query: str
    answer: str
    sources: list[str]      # file_path:line_start references cited
    model: str
    tokens_used: int = 0


# ─── Prompt Construction ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Linux kernel expert with deep knowledge of C systems programming.
You have access to both the kernel's source code (static analysis) and live runtime state
captured via the drgn kernel debugger.

Your role is to bridge Theory (source code) with Reality (live memory) to explain EXACTLY
why the system is behaving the way it is — not just what the code says it should do.

When citing code, always reference the specific file and function (e.g., kernel/sched/core.c:schedule()).
When discussing live state, note the specific field values that are relevant.
Be precise, technical, and substantive. Avoid vague generalities."""


def _estimate_tokens(text: str) -> int:
    """
    Rough token count.  Uses tiktoken when available (fast, accurate for GPT
    tokenizers); falls back to the BPE heuristic 1 token ≈ 4 chars for code.
    For Ollama/local models this is an estimate — they use their own
    tokenizers, but the 4-chars-per-token heuristic is conservative and
    works well enough for budget enforcement.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except (ImportError, Exception):
        return max(1, len(text) // 4)


def build_prompt(
    query: str,
    hybrid_result: HybridResult,
    live_snapshots: list[LiveSnapshot] | None = None,
    max_prompt_tokens: int = 8192,
) -> tuple[str, int]:
    """
    Construct the full LLM prompt with enforced token budget.

    F-24: The old interface used fixed character limits (6000 + 2000 chars)
    that ignored the model's actual context window and the variable sizes
    of the system prompt, query, and instruction block.  This version:
      1. Measures the fixed overhead (system + query + instructions)
      2. Allocates the remaining token budget to code context (75%) and
         live state (25%), shrinking max_nodes dynamically
      3. Returns the estimated prompt token count alongside the prompt string

    Returns: (prompt_text, estimated_token_count)

    Structure:
      [THEORY]  → static code context from Mirror  (75% of variable budget)
      [REALITY] → live drgn snapshots              (25% of variable budget)
      [QUERY]   → user question
    """
    # ── Fixed overhead ────────────────────────────────────────────────────────
    INSTRUCTIONS = (
        "\n=== INSTRUCTIONS ===\n"
        "Synthesize the theory (source code) and reality (live state) to answer the question.\n"
        "- Identify the specific code paths and data structures involved\n"
        "- Explain the causal mechanism (why, not just what)\n"
        "- Cite specific file:function references\n"
        "- If live state is available, connect field values to code behavior\n"
        "- Highlight any discrepancies between theory and reality"
    )

    fixed_text = (
        SYSTEM_PROMPT
        + f"\n\n=== QUESTION ===\n{query}"
        + INSTRUCTIONS
    )
    fixed_tokens = _estimate_tokens(fixed_text)
    variable_budget_tokens = max(256, max_prompt_tokens - fixed_tokens - 200)  # 200 token margin

    # ── Allocate variable budget: 75% code, 25% live state ────────────────────
    code_token_budget = int(variable_budget_tokens * 0.75)
    live_token_budget = int(variable_budget_tokens * 0.25)

    # Chars ≈ tokens * 4 for code
    max_code_chars = code_token_budget * 4
    max_live_chars = live_token_budget * 4

    # ── Build blocks ──────────────────────────────────────────────────────────

    # Theory block — reduce max_nodes if even the first node blows the budget
    max_nodes = 12
    while max_nodes > 1:
        candidate = hybrid_result.to_context_text(max_nodes=max_nodes)
        if len(candidate) <= max_code_chars:
            break
        max_nodes -= 2

    code_context = hybrid_result.to_context_text(max_nodes=max_nodes)
    if len(code_context) > max_code_chars:
        code_context = code_context[:max_code_chars] + "\n... [truncated for token budget]"

    parts = []
    parts.append("=== THEORY: KERNEL SOURCE CODE ===")
    parts.append(code_context)

    # Reality block
    if live_snapshots:
        parts.append("\n=== REALITY: LIVE KERNEL STATE (captured via drgn) ===")
        live_text = "\n\n".join(s.to_text() for s in live_snapshots)
        if len(live_text) > max_live_chars:
            live_text = live_text[:max_live_chars] + "\n... [truncated for token budget]"
        parts.append(live_text)
    else:
        parts.append("\n[No live kernel state available — analysis based on static code only]")

    parts.append(f"\n=== QUESTION ===\n{query}")
    parts.append(INSTRUCTIONS)

    prompt = "\n".join(parts)
    estimated_tokens = fixed_tokens + _estimate_tokens(
        "\n".join(parts[:3])  # theory + reality only
    )
    return prompt, estimated_tokens


# ─── Synthesizer ──────────────────────────────────────────────────────────────

class KernelSynthesizer:
    """
    Wraps an LLM backend and produces structured synthesis results.

    Supports:
      - Ollama (local, default): model="ollama:deepseek-coder:6.7b"
      - OpenAI-compatible: model="openai:gpt-4o", requires OPENAI_API_KEY
      - Anthropic: model="anthropic:claude-3-5-sonnet-20241022", requires ANTHROPIC_API_KEY
    """

    def __init__(
        self,
        model: str = "ollama:deepseek-coder:6.7b",
        base_url: str | None = None,  # For OpenAI-compatible APIs
        api_key: str | None = None,
        temperature: float = 0.2,     # Low temp for factual kernel analysis
        max_tokens: int = 2048,
    ):
        # Parse "provider:model_name" format
        if ":" in model:
            self.provider, self.model_name = model.split(":", 1)
        else:
            self.provider = "ollama"
            self.model_name = model

        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def synthesize(
        self,
        query: str,
        hybrid_result: HybridResult,
        live_snapshots: list[LiveSnapshot] | None = None,
    ) -> SynthesisResult:
        """Run a full synthesis: build prompt → call LLM → parse response."""
        prompt, prompt_tokens = build_prompt(
            query, hybrid_result, live_snapshots,
            max_prompt_tokens=self.max_tokens,
        )
        sources = [
            f"{r.node.file_path}:{r.node.line_start}"
            for r in hybrid_result.primary
        ]

        answer = self._call_llm(prompt)

        return SynthesisResult(
            query=query,
            answer=answer,
            sources=sources,
            model=f"{self.provider}:{self.model_name}",
            tokens_used=prompt_tokens,
        )

    def synthesize_stream(
        self,
        query: str,
        hybrid_result: HybridResult,
        live_snapshots: list[LiveSnapshot] | None = None,
    ) -> Iterator[str]:
        """Streaming synthesis — yields answer tokens as they arrive."""
        prompt, _ = build_prompt(
            query, hybrid_result, live_snapshots,
            max_prompt_tokens=self.max_tokens,
        )
        yield from self._call_llm_stream(prompt)

    # ── LLM Backends ──────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._ollama(prompt)
        elif self.provider in ("openai", "openai-compatible"):
            return self._openai_compatible(prompt)
        elif self.provider == "anthropic":
            return self._anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use ollama, openai, or anthropic.")

    def _call_llm_stream(self, prompt: str) -> Iterator[str]:
        if self.provider == "ollama":
            yield from self._ollama_stream(prompt)
        elif self.provider in ("openai", "openai-compatible"):
            yield from self._openai_stream(prompt)
        else:
            # Fallback: non-streaming
            yield self._call_llm(prompt)

    def _ollama(self, prompt: str) -> str:
        """Call a local Ollama model."""
        try:
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            return response["message"]["content"]
        except ImportError:
            return self._ollama_http(prompt)

    def _ollama_http(self, prompt: str) -> str:
        """Fallback: call Ollama via raw HTTP (no ollama Python package needed)."""
        import json
        import urllib.request

        base = self.base_url or "http://localhost:11434"
        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{base}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["message"]["content"]

    def _ollama_stream(self, prompt: str) -> Iterator[str]:
        """Streaming from Ollama via HTTP."""
        import json
        import urllib.request

        base = self.base_url or "http://localhost:11434"
        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream": True,
            "options": {"temperature": self.temperature},
        }).encode()

        req = urllib.request.Request(
            f"{base}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if line.strip():
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    def _openai_compatible(self, prompt: str) -> str:
        """Call any OpenAI-compatible API (OpenAI, vLLM, LM Studio, etc.)."""
        import os
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        client = OpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=self.base_url,
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _openai_stream(self, prompt: str) -> Iterator[str]:
        """Streaming from OpenAI-compatible API."""
        import os
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=self.base_url,
        )
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        import os
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        client = anthropic.Anthropic(
            api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        message = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
