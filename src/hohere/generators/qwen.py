"""Qwen dialogue generator over an OpenAI-compatible API (vLLM, SGLang, DashScope, Ollama)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import ValidationError

from hohere.config.schema import DialogueConfig
from hohere.generators.base import DialogueGenerationError, normalize_raw_dialogue
from hohere.generators.prompts import JUDGE_PROMPT, build_system_prompt, build_user_prompt
from hohere.schemas.dialogue import ConversationSpec, Dialogue

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model response (tolerates think tags and fences)."""
    text = _THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise DialogueGenerationError("no JSON object in response") from None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise DialogueGenerationError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise DialogueGenerationError("response JSON is not an object")
    return obj


class QwenGenerator:
    name = "qwen"

    def __init__(self, cfg: DialogueConfig, language: str = "en") -> None:
        from openai import OpenAI

        self.cfg = cfg
        self.language = language
        base_url = cfg.base_url or os.environ.get("HOHERE_QWEN_BASE_URL")
        if not base_url:
            raise DialogueGenerationError(
                "no Qwen endpoint: set dialogue.base_url or $HOHERE_QWEN_BASE_URL"
            )
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(cfg.api_key_env, "EMPTY"),
            timeout=cfg.request_timeout_s,
            max_retries=2,
        )
        self._system = build_system_prompt(cfg.max_words_per_utterance)

    def _chat(self, messages: list[dict[str, str]], seed: int) -> tuple[str, str]:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
            seed=seed,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        return content, resp.model or self.cfg.model

    def generate(
        self, spec: ConversationSpec, attempt: int = 0, feedback: str | None = None
    ) -> Dialogue:
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": build_user_prompt(spec, feedback, self.cfg.user_personas)},
        ]
        try:
            content, served_model = self._chat(messages, seed=spec.seed + attempt)
        except Exception as e:  # network / server errors
            raise DialogueGenerationError(f"request failed: {e}") from e
        raw = extract_json(content)
        info = {"provider": "qwen", "model": served_model, "attempt": str(attempt)}
        try:
            return normalize_raw_dialogue(raw, spec, info, self.language)
        except (ValidationError, KeyError, TypeError) as e:
            raise DialogueGenerationError(f"schema mismatch: {e}") from e

    def judge(self, dialogue: Dialogue) -> list[str]:
        """Optional LLM-as-judge pass. Returns a list of issues; empty means accepted."""
        payload = {
            "topic": dialogue.topic,
            "skill": dialogue.skill.value,
            "turns": [t.model_dump(mode="json") for t in dialogue.turns],
        }
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        content, _ = self._chat(messages, seed=dialogue.seed)
        verdict = extract_json(content)
        issues = [str(i) for i in verdict.get("issues") or []]
        for key in ("coherent", "on_topic", "appropriate_responses"):
            if verdict.get(key) is False:
                issues.insert(0, f"judge: not {key}")
        return issues if any(i.startswith("judge:") for i in issues) else []
