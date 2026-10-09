"""
evaluation/ — RAG evaluation metrics (LLM-as-judge). See answer_relevancy.py
(answer relevancy) and retrieval_metrics.py (retriever recall/precision).
"""

import os

# DeepEval sends anonymous usage telemetry unless opted out — set before any
# evaluation module imports deepeval. An explicit value in the environment wins.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from .answer_relevancy import AnswerRelevancyMetric, LLMTestCase, Verdict

__all__ = ["AnswerRelevancyMetric", "LLMTestCase", "Verdict"]
