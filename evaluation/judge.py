"""
evaluation/judge.py — Groq model as a DeepEval LLM judge.

DeepEval defaults to OpenAI; any other model is plugged in by subclassing
DeepEvalBaseLLM (load_model / generate / a_generate / get_model_name).
When a metric passes a pydantic `schema`, generate() must return an
instance of it — the reply is parsed with the same fence/preamble-tolerant
parser as answer_relevancy.py, with Groq's JSON mode on.

Default judge: openai/gpt-oss-120b — the strongest model on this Groq key
(llama-3.3-70b isn't available on it). For retrieval eval the judge grades
document chunks, not the teacher's answers, so using the teacher's model
family doesn't cause self-grading bias there.

    from evaluation.judge import GroqJudge
    judge = GroqJudge()
    judge.generate("Return JSON ...", schema=MySchema)
    print(judge.usage)   # calls / input_tokens / output_tokens so far
"""

import os
from typing import Optional

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from .answer_relevancy import _call_with_retry, _parse_json

JUDGE_MODEL = "openai/gpt-oss-120b"


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = JUDGE_MODEL, temperature: float = 0.0):
        self.model_name = model
        self.temperature = temperature
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        super().__init__(model)  # calls load_model()

    def load_model(self):
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GroqJudge needs GROQ_API_KEY in .env")
        from langchain_groq import ChatGroq
        return ChatGroq(model=self.model_name, temperature=self.temperature)

    def _invoke(self, prompt: str, json_mode: bool) -> str:
        client = self.model.bind(response_format={"type": "json_object"}) if json_mode else self.model
        msg = client.invoke(prompt)
        usage = getattr(msg, "usage_metadata", None) or {}
        self.usage["calls"] += 1
        self.usage["input_tokens"] += usage.get("input_tokens", 0)
        self.usage["output_tokens"] += usage.get("output_tokens", 0)
        return msg.content

    def generate(self, prompt: str, schema: Optional[type] = None):
        raw = _call_with_retry(lambda p: self._invoke(p, json_mode=schema is not None), prompt)
        if schema is None:
            return raw
        data = _parse_json(raw)
        if issubclass(schema, BaseModel):
            return schema.model_validate(data)
        return schema(**data)

    async def a_generate(self, prompt: str, schema: Optional[type] = None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"groq/{self.model_name}"
