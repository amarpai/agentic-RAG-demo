"""Shared configuration: paths, taxonomy and model defaults."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
INDEX_DIR = DATA_DIR / "vector_store"
EVALS_DIR = ROOT / "evals"
GOLDEN_PATH = EVALS_DIR / "golden_set.jsonl"
THRESHOLDS_PATH = EVALS_DIR / "thresholds.json"
REPORTS_DIR = ROOT / "reports"

# Intent taxonomy for the payments research agent. Every question is routed to
# exactly one of these, or to OUT_OF_SCOPE when the knowledge base can't answer it.
CATEGORIES: list[str] = [
    "fraud_detection",
    "chargebacks_disputes",
    "pci_compliance",
    "authorization_settlement",
    "interchange_pricing",
    "cross_border_fx",
    "merchant_onboarding_kyc",
    "real_time_payments",
]
OUT_OF_SCOPE = "out_of_scope"
ALL_LABELS: list[str] = CATEGORIES + [OUT_OF_SCOPE]

CLAUDE_MODEL = os.environ.get("PAYRAG_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.environ.get("PAYRAG_EFFORT", "medium")
SEED = 7
