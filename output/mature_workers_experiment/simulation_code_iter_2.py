#!/usr/bin/env python3
"""
simulate.py

End-to-end multi-agent persona simulation program:

- Loads persona feature rows from employment_selected_features.csv.
- Builds older-adult personas (age >= 50) with normalized employment status.
- Loads instruments from YAML when present:
  - lifesats_items.yaml (item_id coerced to str; response_scale.labels used in prompts)
  - writing_tasks.yaml (empty treated as missing; fallback if allow_missing_instruments)
- Administers Life Satisfaction psychometric test + reflective writing tasks using configurable
  administration mode (per_item / per_test / all_tests) and optional shuffling.
- Supports two LLM providers:
  - MockLLM (offline deterministic simulator)
  - OpenAI Responses API via call_gpt5_with_responses_api (online)
- Builds a simple similarity-based influence network + exogenous signal.
- Calibrates MockLLM parameters on training split (temporal holdout when possible).
- Rolls out on validation and computes evaluation metrics:
  Pearson/Spearman correlation, MAE, Cronbach’s alpha (keyed), split-half, optional test-retest,
  optional shuffle-check.
- Saves outputs (CSV + JSON/JSONL) merged with originating persona rows.

Path handling follows the required pattern via PROJECT_ROOT and DATA_PATH.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------
# Required OpenAI integration (hard requirement)
# ---------------------------------------------------------------------
def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000):
    api_key = get_openai_api_key()
    if OpenAI is None:
        raise ImportError(
            "openai package is not installed or could not be imported. "
            "Install it (pip install openai) to use --llm-provider openai."
        )
    client = OpenAI(api_key=api_key)

    responses_kwargs = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": max_output_tokens,
    }

    resp = client.responses.create(**responses_kwargs)

    def extract_response(resp_obj):
        if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
            return resp_obj.output_text
        try:
            output = getattr(resp_obj, "output", None)
            if output and isinstance(output, list):
                content = output[0].get("content") if isinstance(output[0], dict) else None
                if content and isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text")
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        return str(resp_obj)

    return extract_response(resp)


# ---------------------------------------------------------------------
# Global determinism
# ---------------------------------------------------------------------
GLOBAL_SEED = 12345


def set_global_seed(seed: int) -> np.random.Generator:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    np.random.seed(seed)
    return np.random.default_rng(seed)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------
# Required path handling pattern (derived at use-time; avoids stale globals)
# ---------------------------------------------------------------------
def get_data_dir() -> str:
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.environ.get("DATA_PATH")
    if project_root is None or data_path is None:
        raise EnvironmentError(
            "Missing required environment variables. Please set PROJECT_ROOT and DATA_PATH. "
            "Example:\n"
            "  export PROJECT_ROOT=/abs/path/to/project\n"
            "  export DATA_PATH=data"
        )
    data_dir = os.path.join(project_root, data_path)
    if not os.path.isdir(data_dir):
        raise EnvironmentError(
            f"DATA_DIR does not exist or is not a directory: {data_dir}\n"
            "Check PROJECT_ROOT and DATA_PATH."
        )
    return data_dir


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    cols_lower = [(c.lower(), c) for c in df.columns]
    for cand in candidates:
        cand_l = cand.lower()
        for cl, orig in cols_lower:
            if cand_l in cl:
                return orig
    return None


def safe_float(x: Any, field: str) -> float:
    try:
        if pd.isna(x):
            raise ValueError
        return float(x)
    except Exception as e:
        raise ValueError(f"Field '{field}' must be numeric and non-missing; got {x!r}.") from e


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError("pearson_corr requires arrays of same length >= 2.")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(np.asarray(x, dtype=float))
    ry = rankdata(np.asarray(y, dtype=float))
    return pearson_corr(rx, ry)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("mae requires arrays of same length.")
    return float(np.mean(np.abs(x - y)))


def cronbach_alpha(items: np.ndarray) -> float:
    items = np.asarray(items, dtype=float)
    if items.ndim != 2 or items.shape[1] < 2:
        raise ValueError("cronbach_alpha requires a 2D array with at least 2 items.")
    item_vars = np.var(items, axis=0, ddof=1)
    total = np.sum(items, axis=1)
    total_var = np.var(total, ddof=1)
    k = items.shape[1]
    if total_var <= 0:
        return 0.0
    alpha = (k / (k - 1.0)) * (1.0 - float(np.sum(item_vars) / total_var))
    return float(clamp(alpha, -1.0, 1.0))


def extract_json_object(text: str, *, context: str = "") -> Any:
    """
    Extract the first valid JSON object/array from a string by scanning for the first
    '{' or '[' and attempting JSONDecoder.raw_decode.

    Raises ValueError with contextual details if parsing fails.
    """
    if not isinstance(text, str):
        raise ValueError(f"extract_json_object expected str; got {type(text).__name__} ({context})")
    s = text.strip()
    if not s:
        raise ValueError(f"Empty model output; cannot parse JSON ({context})")

    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(s) if ch in "{["]
    last_err: Optional[Exception] = None
    for i in starts:
        try:
            obj, end = decoder.raw_decode(s[i:])
            return obj
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Failed to extract JSON from model output ({context}). Last error: {last_err}. Output head: {s[:400]!r}")


def json_safe_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
    return str(v)


# ---------------------------------------------------------------------
# Instruments (YAML-driven with fallbacks)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class TestItem:
    item_id: str
    text: str
    reverse_keyed: bool = False


@dataclass(frozen=True)
class ResponseScale:
    scale_min: int
    scale_max: int
    labels: Dict[int, str]


@dataclass(frozen=True)
class PsychometricTest:
    test_id: str
    name: str
    items: Tuple[TestItem, ...]
    response_scale: ResponseScale

    def validate(self) -> None:
        if self.response_scale.scale_min >= self.response_scale.scale_max:
            raise ValueError("scale_min must be < scale_max.")
        if len(self.items) < 2:
            raise ValueError("A test must contain at least 2 items.")
        ids = [it.item_id for it in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate item_id found in test items.")


@dataclass(frozen=True)
class WritingTask:
    task_id: str
    prompt: str


DEFAULT_LIFE_SAT_ITEMS = (
    TestItem("LS1", "I look forward to each day.", reverse_keyed=False),
    TestItem("LS2", "I feel satisfied with how my life is going.", reverse_keyed=False),
    TestItem("LS3", "I feel content and at ease.", reverse_keyed=False),
    TestItem("LS4", "I feel a sense of purpose and meaning.", reverse_keyed=False),
    TestItem("LS5", "Most days feel empty or repetitive.", reverse_keyed=True),
    TestItem("LS6", "I feel emotionally drained or unmotivated.", reverse_keyed=True),
)

DEFAULT_RESPONSE_SCALE = ResponseScale(
    scale_min=1,
    scale_max=4,
    labels={1: "Never", 2: "Not often", 3: "Sometimes", 4: "Often"},
)

DEFAULT_WRITING_TASKS: Tuple[WritingTask, ...] = (
    WritingTask(
        task_id="WT1",
        prompt=(
            "Write a short reflective paragraph (80–160 words) about how your current employment situation "
            "influences your daily routine, sense of purpose, stress, and overall life satisfaction. "
            "Be concrete and consistent with your persona."
        ),
    ),
)


def parse_lifesats_items_yaml(path: str) -> Tuple[Tuple[TestItem, ...], ResponseScale, str]:
    if yaml is None:
        raise ImportError("PyYAML is required to parse instrument YAML files (pip install pyyaml).")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None or raw == "" or raw == {}:
        raise ValueError(f"{os.path.basename(path)} exists but is empty.")

    # Tolerate a few common shapes:
    # - {test_id, name, items: [...], response_scale: {...}}
    # - {items: [...], response_scale: {...}}
    # - [...] (items list)
    test_name = "Life Satisfaction (6-item)"
    if isinstance(raw, dict):
        test_name = str(raw.get("name") or test_name)
        items_raw = raw.get("items")
        if items_raw is None and "lifesats_items" in raw:
            items_raw = raw.get("lifesats_items")
        rs_raw = raw.get("response_scale") or raw.get("scale") or {}
    elif isinstance(raw, list):
        items_raw = raw
        rs_raw = {}
    else:
        raise ValueError(f"Unrecognized YAML structure in {path}")

    if not isinstance(items_raw, list) or not items_raw:
        raise ValueError(f"{path} must contain a non-empty 'items' list.")

    items: List[TestItem] = []
    for i, it in enumerate(items_raw):
        if isinstance(it, dict):
            item_id = str(it.get("item_id") if "item_id" in it else it.get("id") if "id" in it else i + 1)
            text = it.get("text") or it.get("item_text") or it.get("prompt")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Item {item_id} missing valid 'text' in {path}")
            reverse_keyed = bool(it.get("reverse_keyed") or it.get("reverse") or False)
            items.append(TestItem(item_id=item_id, text=text.strip(), reverse_keyed=reverse_keyed))
        else:
            items.append(TestItem(item_id=str(i + 1), text=str(it), reverse_keyed=False))

    scale_min = int(rs_raw.get("min", rs_raw.get("scale_min", DEFAULT_RESPONSE_SCALE.scale_min)) or DEFAULT_RESPONSE_SCALE.scale_min)
    scale_max = int(rs_raw.get("max", rs_raw.get("scale_max", DEFAULT_RESPONSE_SCALE.scale_max)) or DEFAULT_RESPONSE_SCALE.scale_max)

    labels_raw = rs_raw.get("labels")
    labels: Dict[int, str] = {}
    if isinstance(labels_raw, dict) and labels_raw:
        for k, v in labels_raw.items():
            try:
                ki = int(k)
            except Exception:
                continue
            if isinstance(v, str) and v.strip():
                labels[ki] = v.strip()
    if not labels:
        labels = dict(DEFAULT_RESPONSE_SCALE.labels)

    return tuple(items), ResponseScale(scale_min=scale_min, scale_max=scale_max, labels=labels), test_name


def parse_writing_tasks_yaml(path: str, *, allow_missing_instruments: bool) -> Tuple[WritingTask, ...]:
    if yaml is None:
        raise ImportError("PyYAML is required to parse instrument YAML files (pip install pyyaml).")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Treat empty YAML file as missing per requirement.
    if raw is None or raw == "" or raw == {} or raw == []:
        if allow_missing_instruments:
            return DEFAULT_WRITING_TASKS
        raise ValueError(f"{os.path.basename(path)} exists but is empty.")

    tasks: List[WritingTask] = []
    if isinstance(raw, list):
        for i, t in enumerate(raw):
            if isinstance(t, dict):
                tid = str(t.get("task_id") or t.get("id") or f"WT{i+1}")
                prompt = t.get("prompt") or t.get("text")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Writing task {tid} missing valid prompt in {path}")
                tasks.append(WritingTask(task_id=tid, prompt=prompt.strip()))
            else:
                tasks.append(WritingTask(task_id=f"WT{i+1}", prompt=str(t)))
    elif isinstance(raw, dict):
        # Allow {tasks:[...]}
        tlist = raw.get("tasks") or raw.get("writing_tasks")
        if isinstance(tlist, list):
            for i, t in enumerate(tlist):
                if isinstance(t, dict):
                    tid = str(t.get("task_id") or t.get("id") or f"WT{i+1}")
                    prompt = t.get("prompt") or t.get("text")
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(f"Writing task {tid} missing valid prompt in {path}")
                    tasks.append(WritingTask(task_id=tid, prompt=prompt.strip()))
                else:
                    tasks.append(WritingTask(task_id=f"WT{i+1}", prompt=str(t)))
        else:
            # If dict itself is one task
            prompt = raw.get("prompt") or raw.get("text")
            if isinstance(prompt, str) and prompt.strip():
                tasks.append(WritingTask(task_id=str(raw.get("task_id") or "WT1"), prompt=prompt.strip()))
            else:
                raise ValueError(f"Unrecognized writing_tasks YAML structure in {path}")
    else:
        raise ValueError(f"Unrecognized writing_tasks YAML structure in {path}")

    if not tasks:
        return DEFAULT_WRITING_TASKS if allow_missing_instruments else tuple()
    return tuple(tasks)


def load_instruments(*, allow_missing_instruments: bool) -> Tuple[PsychometricTest, Tuple[WritingTask, ...], Dict[str, Any]]:
    data_dir = get_data_dir()
    lifesats_path = os.path.join(data_dir, "lifesats_items.yaml")
    writing_path = os.path.join(data_dir, "writing_tasks.yaml")

    meta: Dict[str, Any] = {
        "lifesats_yaml_used": None,
        "writing_tasks_yaml_used": None,
        "allow_missing_instruments": allow_missing_instruments,
    }

    # Life satisfaction instrument
    if os.path.isfile(lifesats_path):
        try:
            items, rs, name = parse_lifesats_items_yaml(lifesats_path)
            test = PsychometricTest(
                test_id="LIFE_SATISFACTION",
                name=name,
                items=items,
                response_scale=rs,
            )
            test.validate()
            meta["lifesats_yaml_used"] = os.path.basename(lifesats_path)
        except Exception:
            if not allow_missing_instruments:
                raise
            test = PsychometricTest(
                test_id="LIFE_SATISFACTION",
                name="Life Satisfaction (6-item)",
                items=DEFAULT_LIFE_SAT_ITEMS,
                response_scale=DEFAULT_RESPONSE_SCALE,
            )
            meta["lifesats_yaml_used"] = "FALLBACK_DEFAULT"
    else:
        if not allow_missing_instruments:
            raise FileNotFoundError(f"Missing required instrument file: {lifesats_path}")
        test = PsychometricTest(
            test_id="LIFE_SATISFACTION",
            name="Life Satisfaction (6-item)",
            items=DEFAULT_LIFE_SAT_ITEMS,
            response_scale=DEFAULT_RESPONSE_SCALE,
        )
        meta["lifesats_yaml_used"] = "FALLBACK_DEFAULT"

    # Writing tasks
    if os.path.isfile(writing_path):
        try:
            tasks = parse_writing_tasks_yaml(writing_path, allow_missing_instruments=allow_missing_instruments)
            meta["writing_tasks_yaml_used"] = os.path.basename(writing_path)
        except Exception:
            if not allow_missing_instruments:
                raise
            tasks = DEFAULT_WRITING_TASKS
            meta["writing_tasks_yaml_used"] = "FALLBACK_DEFAULT"
    else:
        tasks = DEFAULT_WRITING_TASKS if allow_missing_instruments else tuple()
        meta["writing_tasks_yaml_used"] = "FALLBACK_DEFAULT" if allow_missing_instruments else None

    return test, tasks, meta


# ---------------------------------------------------------------------
# Persona specification & prompt construction
# ---------------------------------------------------------------------
HIGH_DESCRIPTORS = [
    "often looks forward to each day",
    "often feels satisfied with how life is going",
    "often feels content and at ease",
    "often feels a sense of purpose and meaning",
    "often notices and appreciates positive moments",
    "often feels hopeful about the near future",
    "often feels that daily activities are worthwhile",
    "often feels emotionally balanced and steady",
]

LOW_DESCRIPTORS = [
    "never looks forward to each day",
    "never feels satisfied with how life is going",
    "never feels content or at ease",
    "never feels a sense of purpose or meaning",
    "rarely notices positive moments",
    "never feels hopeful about the near future",
    "often feels days are empty or repetitive",
    "often feels emotionally drained or unmotivated",
]

PROMPT_TEMPLATE = (
    'For the following life satisfaction items, respond as a person described as: '
    '"I {qualified_life_satisfaction_description}."'
)

EMPLOYMENT_MAP = {
    "full-time": "full time",
    "full time": "full time",
    "fulltime": "full time",
    "part-time": "part-time/casual",
    "part time": "part-time/casual",
    "parttime": "part-time/casual",
    "casual": "part-time/casual",
    "part-time/casual": "part-time/casual",
    "unemployed": "unemployed",
    "not employed": "unemployed",
    "retired": "unemployed",
    # Execution-blocker fix: map "No" variants
    "no": "unemployed",
    "n": "unemployed",
    "false": "unemployed",
    "0": "unemployed",
    "none": "unemployed",
    "no paid job": "unemployed",
    "no job": "unemployed",
}

PSYCHOSOCIAL_COL_CANDIDATE_KEYWORDS = (
    "depress",
    "anx",
    "stress",
    "lonely",
    "loneliness",
    "social",
    "support",
    "control",
    "purpose",
    "meaning",
    "wellbeing",
    "well-being",
    "health",
    "sleep",
    "pain",
)


@dataclass
class Persona:
    persona_id: str
    row_index: int
    age: float
    employment_status: str
    target_life_satisfaction: float
    features: Dict[str, Any]
    prompt: str
    psychosocial_index: float
    psychosocial_profile: str


def normalize_employment_status(raw: Any, *, default_on_unknown: Optional[str] = None) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        if default_on_unknown is not None:
            return default_on_unknown
        raise ValueError(
            "Employment status is missing. Provide a column with values like "
            "'full-time', 'part-time', or 'unemployed'."
        )
    s = str(raw).strip().lower()
    s = s.replace("_", " ").replace("-", " ").strip()
    s = " ".join(s.split())
    mapped = EMPLOYMENT_MAP.get(s)
    if mapped is not None:
        return mapped
    if "full" in s and "time" in s:
        return "full time"
    if "part" in s or "casual" in s:
        return "part-time/casual"
    if "unemploy" in s or "not employ" in s or "retir" in s:
        return "unemployed"
    if default_on_unknown is not None:
        return default_on_unknown
    raise ValueError(
        f"Unrecognized employment status value: {raw!r}. Expected values mapping to "
        "'full time', 'part-time/casual', or 'unemployed'."
    )


def life_satisfaction_level(target_score_1_to_4: float) -> int:
    level = int(round(clamp(float(target_score_1_to_4), 1.0, 4.0)))
    return int(clamp(level, 1, 4))


def construct_life_satisfaction_description(level: int, rng: np.random.Generator) -> str:
    if level not in (1, 2, 3, 4):
        raise ValueError("level must be one of {1,2,3,4}.")
    n_phrases = int(rng.integers(1, 4))
    if level == 4:
        phrases = rng.choice(HIGH_DESCRIPTORS, size=n_phrases, replace=False).tolist()
        return ", ".join(phrases)
    if level == 1:
        phrases = rng.choice(LOW_DESCRIPTORS, size=n_phrases, replace=False).tolist()
        return ", ".join(phrases)
    if level == 2:
        phrases = rng.choice(LOW_DESCRIPTORS, size=n_phrases, replace=False).tolist()
        softened = []
        for p in phrases:
            p2 = p.replace("never ", "not often ").replace("often feels", "sometimes feels")
            p2 = p2.replace("never feels", "rarely feels")
            softened.append(p2)
        return ", ".join(softened)
    phrases = rng.choice(HIGH_DESCRIPTORS, size=n_phrases, replace=False).tolist()
    tempered = []
    for p in phrases:
        p2 = p.replace("often ", "sometimes ").replace("often feels", "fairly often feels")
        tempered.append(p2)
    return ", ".join(tempered)


def banded_summary(value_z: float, label: str) -> str:
    if value_z <= -0.8:
        return f"{label}: low"
    if value_z >= 0.8:
        return f"{label}: high"
    return f"{label}: moderate"


def build_psychosocial_index_and_profile(row: Mapping[str, Any], numeric_cols: Sequence[str]) -> Tuple[float, str]:
    """
    Create a lightweight psychosocial index using available numeric columns that look relevant.
    This is intentionally simple: z-score average across selected columns where possible.
    """
    # This function returns per-row value after normalization is computed elsewhere.
    # Here we just format a profile string from already-provided z-index.
    # (Caller provides computed z-index; this stub is kept for clarity.)
    raise NotImplementedError


def construct_persona_prompt(
    level: int,
    rng: np.random.Generator,
    response_scale: ResponseScale,
    psychosocial_profile: str,
    employment_status: str,
    age: float,
) -> str:
    desc = construct_life_satisfaction_description(level, rng)
    scale_lines = "\n".join([f"{k}: {response_scale.labels.get(k, str(k))}" for k in range(response_scale.scale_min, response_scale.scale_max + 1)])
    return (
        f"{PROMPT_TEMPLATE.format(qualified_life_satisfaction_description=desc)}\n\n"
        f"Persona details (keep consistent):\n"
        f"- Age: {int(round(age))}\n"
        f"- Employment: {employment_status}\n"
        f"- Psychosocial summary: {psychosocial_profile}\n\n"
        f"Response scale (choose one number per item):\n{scale_lines}\n"
    )


# ---------------------------------------------------------------------
# Administration configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class AdministrationConfig:
    mode: str  # per_item | per_test | all_tests
    shuffle_items_within_tests: bool
    shuffle_test_order: bool
    fixed_order_for_reproducibility: bool

    def validate(self) -> None:
        modes = {"per_item", "per_test", "all_tests"}
        if self.mode not in modes:
            raise ValueError(f"Unknown administration mode {self.mode!r}; must be one of {sorted(modes)}.")
        for field_name in (
            "shuffle_items_within_tests",
            "shuffle_test_order",
            "fixed_order_for_reproducibility",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, bool):
                raise ValueError(f"{field_name} must be a boolean; got {val!r}.")
        if self.fixed_order_for_reproducibility and (self.shuffle_items_within_tests or self.shuffle_test_order):
            raise ValueError(
                "fixed_order_for_reproducibility=True is incompatible with shuffle_items_within_tests "
                "or shuffle_test_order. Set shuffling flags to False or fixed_order_for_reproducibility to False."
            )


@dataclass(frozen=True)
class HoldoutConfig:
    validation_fraction: float = 0.2
    time_column_candidates: Tuple[str, ...] = ("date", "time", "timestamp", "wave", "year", "month")

    def validate(self) -> None:
        if not (0.05 <= self.validation_fraction <= 0.5):
            raise ValueError("validation_fraction must be between 0.05 and 0.5 for a meaningful holdout split.")


# ---------------------------------------------------------------------
# Simple graph for multi-agent interactions
# ---------------------------------------------------------------------
class SimpleGraph:
    def __init__(self, n: int) -> None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("Graph size n must be a positive integer.")
        self._adj: List[List[int]] = [[] for _ in range(n)]

    def add_edge(self, i: int, j: int) -> None:
        if i == j:
            return
        n = len(self._adj)
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError("Edge indices out of bounds.")
        if j not in self._adj[i]:
            self._adj[i].append(j)
        if i not in self._adj[j]:
            self._adj[j].append(i)

    def neighbors(self, i: int) -> List[int]:
        return list(self._adj[i])

    @property
    def n_nodes(self) -> int:
        return len(self._adj)


# ---------------------------------------------------------------------
# Memory/Planning placeholders (required prompt composition)
# ---------------------------------------------------------------------
class MemoryAgent:
    def build_user_context(self, persona: Persona) -> str:
        return (
            "MEMORY_AGENT_USER_CONTEXT:\n"
            f"- Age: {int(round(persona.age))}\n"
            f"- Employment status: {persona.employment_status}\n"
            f"- Psychosocial profile: {persona.psychosocial_profile}\n"
        )

    def build_item_context(self, test: PsychometricTest, items: Sequence[TestItem], writing_tasks: Sequence[WritingTask]) -> str:
        parts = ["MEMORY_AGENT_ITEM_CONTEXT:"]
        parts.append(f"- Test: {test.name} ({test.test_id})")
        parts.append("- Items:")
        for it in items:
            rk = " (reverse-keyed)" if it.reverse_keyed else ""
            parts.append(f"  - {it.item_id}{rk}: {it.text}")
        if writing_tasks:
            parts.append("- Writing tasks:")
            for wt in writing_tasks:
                parts.append(f"  - {wt.task_id}: {wt.prompt}")
        return "\n".join(parts) + "\n"


class PlanningAgent:
    def build_plan_steps(self, mode: str) -> str:
        return (
            "PLANNING_AGENT_STEPS:\n"
            "1) Read the persona context and keep it consistent.\n"
            "2) For each item, select one numeric response using the provided scale labels.\n"
            "3) If a writing task is present, write a short reflective paragraph consistent with persona.\n"
            "4) Output ONLY valid JSON in the required schema.\n"
            f"(Administration mode: {mode})\n"
        )


def build_llm_prompt(
    *,
    persona: Persona,
    test: PsychometricTest,
    items: Sequence[TestItem],
    writing_tasks: Sequence[WritingTask],
    admin_mode: str,
) -> str:
    mem = MemoryAgent()
    plan = PlanningAgent()
    user_ctx = mem.build_user_context(persona)
    item_ctx = mem.build_item_context(test, items, writing_tasks)
    steps = plan.build_plan_steps(admin_mode)

    schema = {
        "responses": {it.item_id: "integer in [scale_min, scale_max]" for it in items},
        "reflection": "string",
        "writing": {wt.task_id: "string" for wt in writing_tasks} if writing_tasks else {},
    }

    scale = test.response_scale
    return (
        f"{user_ctx}\n"
        f"{item_ctx}\n"
        f"{steps}\n"
        f"PERSONA_PROMPT:\n{persona.prompt}\n\n"
        f"RESPONSE_SCHEMA (example types; output must be JSON object):\n{json.dumps(schema, indent=2)}\n\n"
        f"IMPORTANT:\n"
        f"- Output ONLY JSON.\n"
        f"- Use integers {scale.scale_min}..{scale.scale_max} for each item response.\n"
        f"- Do not include extra keys.\n"
    )


# ---------------------------------------------------------------------
# LLM interfaces
# ---------------------------------------------------------------------
@dataclass
class CalibratedParams:
    base_bias: float
    noise_sigma: float
    employment_effect: float
    psychosocial_effect: float
    influence_weight: float
    exogenous_weight: float


class LLMBase:
    def respond(
        self,
        *,
        persona: Persona,
        test: PsychometricTest,
        items: Sequence[TestItem],
        writing_tasks: Sequence[WritingTask],
        latent_level_1_to_4: float,
        neighbor_latent_mean: Optional[float],
        exogenous_signal: float,
        admin_mode: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class MockLLM(LLMBase):
    def __init__(self, rng: np.random.Generator, params: CalibratedParams) -> None:
        self.rng = rng
        self.params = params
        self.call_count = 0

    def _employment_numeric(self, employment_status: str) -> float:
        if employment_status == "full time":
            return 1.0
        if employment_status == "part-time/casual":
            return 0.5
        if employment_status == "unemployed":
            return 0.0
        raise ValueError(f"Unexpected normalized employment status: {employment_status!r}")

    def respond(
        self,
        *,
        persona: Persona,
        test: PsychometricTest,
        items: Sequence[TestItem],
        writing_tasks: Sequence[WritingTask],
        latent_level_1_to_4: float,
        neighbor_latent_mean: Optional[float],
        exogenous_signal: float,
        admin_mode: str,
    ) -> Dict[str, Any]:
        self.call_count += 1
        scale = test.response_scale

        emp = self._employment_numeric(persona.employment_status)
        latent = (
            float(latent_level_1_to_4)
            + self.params.base_bias
            + self.params.employment_effect * (emp - 0.5)
            + self.params.psychosocial_effect * persona.psychosocial_index
        )

        if neighbor_latent_mean is not None:
            latent += self.params.influence_weight * (neighbor_latent_mean - latent)

        latent += self.params.exogenous_weight * exogenous_signal
        latent = clamp(latent, scale.scale_min, scale.scale_max)

        item_responses: List[Dict[str, Any]] = []
        for it in items:
            noise = float(self.rng.normal(0.0, self.params.noise_sigma))

            # Generate observed responses:
            # For reverse-keyed items (negative wording), higher latent should yield lower observed response.
            raw = latent + noise
            if it.reverse_keyed:
                raw = (scale.scale_min + scale.scale_max) - raw

            score = int(round(clamp(raw, scale.scale_min, scale.scale_max)))
            item_responses.append(
                {
                    "item_id": it.item_id,
                    "item_text": it.text,
                    "reverse_keyed": it.reverse_keyed,
                    "response": score,
                    "scale_min": scale.scale_min,
                    "scale_max": scale.scale_max,
                }
            )

        reflection = (
            f"As a {int(round(persona.age))}-year-old who is {persona.employment_status}, "
            f"I would describe my recent life satisfaction as shaped by my routines and circumstances. "
            f"(context: {persona.psychosocial_profile})"
        )

        writing_out: Dict[str, str] = {}
        for wt in writing_tasks:
            writing_out[wt.task_id] = (
                f"My employment situation ({persona.employment_status}) affects my day-to-day structure and motivation. "
                f"{persona.psychosocial_profile}. I interpret my routines and relationships through that lens, "
                f"which influences how satisfied I feel overall."
            )

        return {
            "persona_id": persona.persona_id,
            "row_index": persona.row_index,
            "test_id": test.test_id,
            "call_index": self.call_count,
            "prompt": persona.prompt,
            "latent_used": float(latent),
            "exogenous_signal": float(exogenous_signal),
            "neighbor_latent_mean": neighbor_latent_mean,
            "employment_status": persona.employment_status,
            "responses": item_responses,
            "reflection": reflection,
            "writing": writing_out,
            "provider": "mock",
        }


class OpenAILLM(LLMBase):
    def __init__(self, model: str, max_output_tokens: int) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.call_count = 0

    def respond(
        self,
        *,
        persona: Persona,
        test: PsychometricTest,
        items: Sequence[TestItem],
        writing_tasks: Sequence[WritingTask],
        latent_level_1_to_4: float,
        neighbor_latent_mean: Optional[float],
        exogenous_signal: float,
        admin_mode: str,
    ) -> Dict[str, Any]:
        self.call_count += 1

        prompt = build_llm_prompt(
            persona=persona,
            test=test,
            items=items,
            writing_tasks=writing_tasks,
            admin_mode=admin_mode,
        )

        raw_text = call_gpt5_with_responses_api(
            prompt=prompt,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
        )

        context = f"persona_id={persona.persona_id}, test_id={test.test_id}, mode={admin_mode}, call={self.call_count}"
        obj = extract_json_object(raw_text, context=context)
        if not isinstance(obj, dict):
            raise ValueError(f"Model did not return a JSON object ({context}). Got: {type(obj).__name__}")

        resp_map = obj.get("responses")
        if not isinstance(resp_map, dict):
            raise ValueError(f"Missing/invalid 'responses' in model JSON ({context}).")

        scale = test.response_scale
        item_responses: List[Dict[str, Any]] = []
        for it in items:
            if it.item_id not in resp_map:
                raise ValueError(f"Missing response for item_id {it.item_id} ({context}).")
            try:
                r = int(resp_map[it.item_id])
            except Exception as e:
                raise ValueError(f"Non-integer response for item_id {it.item_id}: {resp_map[it.item_id]!r} ({context}).") from e
            if not (scale.scale_min <= r <= scale.scale_max):
                raise ValueError(f"Out-of-range response for {it.item_id}: {r} ({context}).")
            item_responses.append(
                {
                    "item_id": it.item_id,
                    "item_text": it.text,
                    "reverse_keyed": it.reverse_keyed,
                    "response": r,
                    "scale_min": scale.scale_min,
                    "scale_max": scale.scale_max,
                }
            )

        reflection = obj.get("reflection")
        if not isinstance(reflection, str):
            reflection = str(reflection) if reflection is not None else ""

        writing_out: Dict[str, str] = {}
        writing_obj = obj.get("writing")
        if writing_tasks:
            if isinstance(writing_obj, dict):
                for wt in writing_tasks:
                    val = writing_obj.get(wt.task_id, "")
                    writing_out[wt.task_id] = val if isinstance(val, str) else str(val)
            else:
                # tolerate missing writing: keep empty strings
                for wt in writing_tasks:
                    writing_out[wt.task_id] = ""

        return {
            "persona_id": persona.persona_id,
            "row_index": persona.row_index,
            "test_id": test.test_id,
            "call_index": self.call_count,
            "prompt": persona.prompt,
            "latent_used": float(latent_level_1_to_4),
            "exogenous_signal": float(exogenous_signal),
            "neighbor_latent_mean": neighbor_latent_mean,
            "employment_status": persona.employment_status,
            "responses": item_responses,
            "reflection": reflection,
            "writing": writing_out,
            "provider": "openai",
            "openai_model": self.model,
        }


# ---------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------
class Simulator:
    def __init__(self, rng: np.random.Generator, llm: LLMBase, admin_cfg: AdministrationConfig) -> None:
        self.rng = rng
        self.llm = llm
        self.admin_cfg = admin_cfg
        self.admin_cfg.validate()

    def rollout(
        self,
        personas: Sequence[Persona],
        test: PsychometricTest,
        writing_tasks: Sequence[WritingTask],
        graph: SimpleGraph,
        exogenous_series: np.ndarray,
        rollout_steps: int = 3,
    ) -> List[Dict[str, Any]]:
        return self.administer(personas, test, writing_tasks, graph, exogenous_series, rollout_steps=rollout_steps)

    def administer(
        self,
        personas: Sequence[Persona],
        test: PsychometricTest,
        writing_tasks: Sequence[WritingTask],
        graph: SimpleGraph,
        exogenous_series: np.ndarray,
        rollout_steps: int,
    ) -> List[Dict[str, Any]]:
        if len(personas) == 0:
            raise ValueError("No personas provided for simulation.")
        if graph.n_nodes != len(personas):
            raise ValueError(f"Graph node count ({graph.n_nodes}) must equal number of personas ({len(personas)}).")
        if exogenous_series.shape[0] != len(personas):
            raise ValueError(f"exogenous_series length ({exogenous_series.shape[0]}) must equal number of personas ({len(personas)}).")
        if rollout_steps < 1 or rollout_steps > 20:
            raise ValueError("rollout_steps must be between 1 and 20.")
        test.validate()

        # Influence rollout over target scores (smoothing)
        targets = np.array([p.target_life_satisfaction for p in personas], dtype=float)
        latent = targets.copy()
        for _ in range(rollout_steps - 1):
            updated = latent.copy()
            for i in range(len(personas)):
                neigh = graph.neighbors(i)
                if neigh:
                    neigh_mean = float(np.mean(latent[neigh]))
                    updated[i] = 0.7 * latent[i] + 0.3 * neigh_mean
            latent = updated

        records: List[Dict[str, Any]] = []

        for i, persona in enumerate(personas):
            neighbor_idxs = graph.neighbors(i)
            neighbor_latent_mean = float(np.mean(latent[neighbor_idxs])) if neighbor_idxs else None
            exo = float(exogenous_series[i])

            if self.admin_cfg.mode == "all_tests":
                # One call for items + writing tasks
                items = list(test.items)
                if not self.admin_cfg.fixed_order_for_reproducibility and self.admin_cfg.shuffle_items_within_tests:
                    self.rng.shuffle(items)

                call = self.llm.respond(
                    persona=persona,
                    test=test,
                    items=items,
                    writing_tasks=list(writing_tasks),
                    latent_level_1_to_4=float(latent[i]),
                    neighbor_latent_mean=neighbor_latent_mean,
                    exogenous_signal=exo,
                    admin_mode=self.admin_cfg.mode,
                )
                rec = self._build_test_level_record(persona, test, call["responses"])
                rec["call_metadata"] = {"mode": self.admin_cfg.mode, "calls_used": 1, "call_index": call.get("call_index")}
                rec["reflection"] = call.get("reflection", "")
                rec["writing"] = call.get("writing", {})
                rec["prompt"] = call.get("prompt", "")
                rec["provider"] = call.get("provider")
                rec["openai_model"] = call.get("openai_model")
                records.append(rec)

            elif self.admin_cfg.mode == "per_test":
                items = list(test.items)
                if not self.admin_cfg.fixed_order_for_reproducibility and self.admin_cfg.shuffle_items_within_tests:
                    self.rng.shuffle(items)
                call = self.llm.respond(
                    persona=persona,
                    test=test,
                    items=items,
                    writing_tasks=list(writing_tasks),
                    latent_level_1_to_4=float(latent[i]),
                    neighbor_latent_mean=neighbor_latent_mean,
                    exogenous_signal=exo,
                    admin_mode=self.admin_cfg.mode,
                )
                rec = self._build_test_level_record(persona, test, call["responses"])
                rec["call_metadata"] = {"mode": self.admin_cfg.mode, "calls_used": 1, "call_index": call.get("call_index")}
                rec["reflection"] = call.get("reflection", "")
                rec["writing"] = call.get("writing", {})
                rec["prompt"] = call.get("prompt", "")
                rec["provider"] = call.get("provider")
                rec["openai_model"] = call.get("openai_model")
                records.append(rec)

            elif self.admin_cfg.mode == "per_item":
                items = list(test.items)
                if not self.admin_cfg.fixed_order_for_reproducibility and self.admin_cfg.shuffle_items_within_tests:
                    self.rng.shuffle(items)

                all_item_resps: List[Dict[str, Any]] = []
                call_indices: List[int] = []
                last_reflection = ""
                last_prompt = ""
                writing_out: Dict[str, str] = {}

                for it in items:
                    # Only include writing tasks on the final call for this persona to limit token use.
                    wts = list(writing_tasks) if it.item_id == items[-1].item_id else []
                    call = self.llm.respond(
                        persona=persona,
                        test=test,
                        items=[it],
                        writing_tasks=wts,
                        latent_level_1_to_4=float(latent[i]),
                        neighbor_latent_mean=neighbor_latent_mean,
                        exogenous_signal=exo,
                        admin_mode=self.admin_cfg.mode,
                    )
                    call_indices.append(int(call.get("call_index", len(call_indices) + 1)))
                    all_item_resps.extend(call["responses"])
                    last_reflection = call.get("reflection", last_reflection)
                    last_prompt = call.get("prompt", last_prompt)
                    if wts:
                        writing_out = call.get("writing", writing_out) or writing_out

                rec = self._build_test_level_record(persona, test, all_item_resps)
                rec["call_metadata"] = {"mode": self.admin_cfg.mode, "calls_used": len(items), "call_indices": call_indices}
                rec["reflection"] = last_reflection
                rec["writing"] = writing_out
                rec["prompt"] = last_prompt
                rec["provider"] = (call.get("provider") if "call" in locals() else None)
                rec["openai_model"] = (call.get("openai_model") if "call" in locals() else None)
                records.append(rec)
            else:
                raise RuntimeError(f"Unhandled administration mode: {self.admin_cfg.mode}")

        return records

    def _build_test_level_record(
        self,
        persona: Persona,
        test: PsychometricTest,
        item_responses: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        resp_map = {r["item_id"]: r for r in item_responses}
        missing = [it.item_id for it in test.items if it.item_id not in resp_map]
        if missing:
            raise RuntimeError(f"Missing item responses for test {test.test_id}: {missing}.")

        scale = test.response_scale
        keyed_scores: List[int] = []
        for it in test.items:
            r = int(resp_map[it.item_id]["response"])
            # Re-key for scoring: higher keyed = more satisfaction
            if it.reverse_keyed:
                keyed = (scale.scale_min + scale.scale_max) - r
            else:
                keyed = r
            keyed_scores.append(int(keyed))

        obtained = float(np.mean(keyed_scores))
        target = float(persona.target_life_satisfaction)

        return {
            "persona_id": persona.persona_id,
            "row_index": persona.row_index,
            "age": persona.age,
            "employment_status": persona.employment_status,
            "psychosocial_index": float(persona.psychosocial_index),
            "psychosocial_profile": persona.psychosocial_profile,
            "test_id": test.test_id,
            "test_name": test.name,
            "target_scores": {"LIFE_SATISFACTION": target},
            "obtained_scores": {"LIFE_SATISFACTION": obtained},
            "item_responses": list(item_responses),
            "features": persona.features,
        }


# ---------------------------------------------------------------------
# Calibration (fast objective; avoids building full records)
# ---------------------------------------------------------------------
class GridSearchCalibrator:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit(
        self,
        train_personas: Sequence[Persona],
        test: PsychometricTest,
        graph: SimpleGraph,
        exogenous_series: np.ndarray,
        admin_cfg: AdministrationConfig,
        rollout_steps: int,
    ) -> CalibratedParams:
        if len(train_personas) < 10:
            raise ValueError(
                f"Training set too small ({len(train_personas)}). Increase data size or reduce validation_fraction."
            )
        test.validate()
        admin_cfg.validate()

        bias_grid = [-0.20, -0.10, 0.0, 0.10, 0.20]
        sigma_grid = [0.15, 0.25, 0.35, 0.50]
        emp_grid = [-0.20, 0.0, 0.20, 0.35]
        psy_grid = [-0.15, 0.0, 0.15]
        infl_grid = [0.0, 0.10, 0.20]
        exo_grid = [0.0, 0.05, 0.10]

        best_params: Optional[CalibratedParams] = None
        best_obj = float("inf")

        for base_bias in bias_grid:
            for noise_sigma in sigma_grid:
                for employment_effect in emp_grid:
                    for psychosocial_effect in psy_grid:
                        for influence_weight in infl_grid:
                            for exogenous_weight in exo_grid:
                                params = CalibratedParams(
                                    base_bias=base_bias,
                                    noise_sigma=noise_sigma,
                                    employment_effect=employment_effect,
                                    psychosocial_effect=psychosocial_effect,
                                    influence_weight=influence_weight,
                                    exogenous_weight=exogenous_weight,
                                )
                                obj = self._objective(
                                    params=params,
                                    personas=train_personas,
                                    test=test,
                                    graph=graph,
                                    exogenous_series=exogenous_series,
                                    rollout_steps=rollout_steps,
                                )
                                if obj < best_obj:
                                    best_obj = obj
                                    best_params = params

        if best_params is None:
            raise RuntimeError("Calibration failed to find parameters (unexpected).")
        return best_params

    def _objective(
        self,
        *,
        params: CalibratedParams,
        personas: Sequence[Persona],
        test: PsychometricTest,
        graph: SimpleGraph,
        exogenous_series: np.ndarray,
        rollout_steps: int,
    ) -> float:
        rng = np.random.default_rng(self.seed)
        obtained = simulate_obtained_scores_fast(
            personas=personas,
            test=test,
            graph=graph,
            exogenous_series=exogenous_series,
            rollout_steps=rollout_steps,
            params=params,
            rng=rng,
        )
        target = np.asarray([p.target_life_satisfaction for p in personas], dtype=float)
        p = pearson_corr(target, obtained)
        s = spearman_corr(target, obtained)
        m = mae(target, obtained)
        return float(0.6 * m + 0.2 * (1.0 - p) + 0.2 * (1.0 - s))


def simulate_obtained_scores_fast(
    *,
    personas: Sequence[Persona],
    test: PsychometricTest,
    graph: SimpleGraph,
    exogenous_series: np.ndarray,
    rollout_steps: int,
    params: CalibratedParams,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(personas)
    if graph.n_nodes != n:
        raise ValueError("Graph size mismatch.")
    if exogenous_series.shape[0] != n:
        raise ValueError("Exogenous series length mismatch.")
    scale = test.response_scale

    targets = np.array([p.target_life_satisfaction for p in personas], dtype=float)
    latent = targets.copy()
    for _ in range(max(0, rollout_steps - 1)):
        updated = latent.copy()
        for i in range(n):
            neigh = graph.neighbors(i)
            if neigh:
                updated[i] = 0.7 * latent[i] + 0.3 * float(np.mean(latent[neigh]))
        latent = updated

    def emp_num(s: str) -> float:
        return {"full time": 1.0, "part-time/casual": 0.5, "unemployed": 0.0}[s]

    obtained = np.zeros(n, dtype=float)
    for i, p in enumerate(personas):
        neigh = graph.neighbors(i)
        neigh_mean = float(np.mean(latent[neigh])) if neigh else None
        exo = float(exogenous_series[i])

        base = (
            float(latent[i])
            + params.base_bias
            + params.employment_effect * (emp_num(p.employment_status) - 0.5)
            + params.psychosocial_effect * p.psychosocial_index
        )
        if neigh_mean is not None:
            base += params.influence_weight * (neigh_mean - base)
        base += params.exogenous_weight * exo
        base = clamp(base, scale.scale_min, scale.scale_max)

        keyed_scores = []
        for it in test.items:
            noise = float(rng.normal(0.0, params.noise_sigma))
            raw = base + noise
            if it.reverse_keyed:
                raw = (scale.scale_min + scale.scale_max) - raw
            resp = int(round(clamp(raw, scale.scale_min, scale.scale_max)))
            keyed = (scale.scale_min + scale.scale_max) - resp if it.reverse_keyed else resp
            keyed_scores.append(keyed)
        obtained[i] = float(np.mean(keyed_scores))
    return obtained


# ---------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------
class Evaluator:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def compute_metrics(
        self,
        records: Sequence[Dict[str, Any]],
        test: PsychometricTest,
        simulator_factory_for_retest: Optional[Callable[[], Simulator]] = None,
        retest_subset_fraction: float = 0.2,
        simulator_factory_for_shuffle_check: Optional[Callable[[], Simulator]] = None,
        shuffle_subset_fraction: float = 0.2,
    ) -> Dict[str, Any]:
        if len(records) < 2:
            raise ValueError("Need at least 2 records to compute correlations.")
        recs = [r for r in records if r.get("test_id") == test.test_id]
        if len(recs) < 2:
            raise ValueError(f"Need at least 2 records for test {test.test_id}.")

        target = np.array([r["target_scores"]["LIFE_SATISFACTION"] for r in recs], dtype=float)
        obtained = np.array([r["obtained_scores"]["LIFE_SATISFACTION"] for r in recs], dtype=float)

        item_ids = [it.item_id for it in test.items]
        scale = test.response_scale
        item_matrix_keyed = np.zeros((len(recs), len(item_ids)), dtype=float)
        for i, r in enumerate(recs):
            resp_map = {x["item_id"]: x for x in r["item_responses"]}
            for j, it in enumerate(test.items):
                resp = float(resp_map[it.item_id]["response"])
                keyed = (scale.scale_min + scale.scale_max) - resp if it.reverse_keyed else resp
                item_matrix_keyed[i, j] = keyed

        fidelity = {
            "pearson_r": pearson_corr(target, obtained),
            "spearman_r": spearman_corr(target, obtained),
            "mae": mae(target, obtained),
        }
        consistency = {
            "cronbach_alpha_keyed_items": cronbach_alpha(item_matrix_keyed),
            "split_half_reliability_pearson_keyed": self._split_half(item_matrix_keyed),
        }

        if simulator_factory_for_retest is not None:
            consistency["test_retest"] = self._re_administer_consistency(
                recs=recs,
                test=test,
                simulator_factory=simulator_factory_for_retest,
                subset_fraction=retest_subset_fraction,
                label="test_retest",
            )

        if simulator_factory_for_shuffle_check is not None:
            consistency["shuffle_check"] = self._re_administer_consistency(
                recs=recs,
                test=test,
                simulator_factory=simulator_factory_for_shuffle_check,
                subset_fraction=shuffle_subset_fraction,
                label="shuffle_check",
            )

        return {"persona_fidelity": fidelity, "response_consistency": consistency, "n_records": len(recs)}

    def _split_half(self, item_matrix: np.ndarray) -> float:
        if item_matrix.shape[1] < 2:
            return 0.0
        odd = np.sum(item_matrix[:, ::2], axis=1)
        even = np.sum(item_matrix[:, 1::2], axis=1)
        return pearson_corr(odd, even)

    def _re_administer_consistency(
        self,
        *,
        recs: Sequence[Dict[str, Any]],
        test: PsychometricTest,
        simulator_factory: Callable[[], Simulator],
        subset_fraction: float,
        label: str,
    ) -> Dict[str, Any]:
        if not (0.0 < subset_fraction <= 1.0):
            raise ValueError("subset_fraction must be in (0,1].")
        n = len(recs)
        k = max(2, int(round(n * subset_fraction)))
        idx = np.arange(n)
        self.rng.shuffle(idx)
        chosen = idx[:k]

        personas: List[Persona] = []
        for i in chosen:
            r = recs[int(i)]
            feat = dict(r.get("features") or {})
            personas.append(
                Persona(
                    persona_id=str(r["persona_id"]),
                    row_index=int(r["row_index"]),
                    age=float(r["age"]),
                    employment_status=str(r["employment_status"]),
                    target_life_satisfaction=float(r["target_scores"]["LIFE_SATISFACTION"]),
                    features=feat,
                    prompt=str(r.get("prompt") or ""),
                    psychosocial_index=float(r.get("psychosocial_index", 0.0)),
                    psychosocial_profile=str(r.get("psychosocial_profile") or ""),
                )
            )

        graph = SimpleGraph(n=len(personas))
        exo = np.zeros(len(personas), dtype=float)

        sim = simulator_factory()
        retest_records = sim.administer(personas, test, writing_tasks=tuple(), graph=graph, exogenous_series=exo, rollout_steps=1)

        baseline = np.array([recs[int(i)]["obtained_scores"]["LIFE_SATISFACTION"] for i in chosen], dtype=float)
        retest = np.array([r["obtained_scores"]["LIFE_SATISFACTION"] for r in retest_records], dtype=float)

        return {"label": label, "n": int(k), "pearson_r": pearson_corr(baseline, retest), "mae": mae(baseline, retest)}


# ---------------------------------------------------------------------
# Data loading, persona construction, network construction, splitting, saving
# ---------------------------------------------------------------------
def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-agent persona simulation for older adults (employment & life satisfaction).")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Global random seed (non-negative int).")
    p.add_argument("--validation-fraction", type=float, default=0.2, help="Holdout validation fraction (0.05..0.5).")
    p.add_argument("--time-column", type=str, default="", help="Optional explicit time column for temporal holdout split.")
    p.add_argument(
        "--administration-mode",
        type=str,
        default="per_test",
        choices=["per_item", "per_test", "all_tests"],
        help="API granularity mode.",
    )
    p.add_argument("--shuffle-items-within-tests", action="store_true", help="Shuffle item order within each test.")
    p.add_argument("--shuffle-test-order", action="store_true", help="Shuffle order of tests (unused for single test).")
    p.add_argument("--fixed-order-for-reproducibility", action="store_true", help="Disable all shuffling to enforce a fixed order.")
    p.add_argument("--rollout-steps", type=int, default=3, help="Influence rollout steps (1..20).")

    p.add_argument("--llm-provider", type=str, default="mock", choices=["mock", "openai"], help="Which LLM backend to use.")
    p.add_argument("--openai-model", type=str, default="gpt-5", help="OpenAI model name for Responses API.")
    p.add_argument("--openai-max-output-tokens", type=int, default=2000, help="max_output_tokens for OpenAI Responses API.")

    p.add_argument("--allow-missing-instruments", action="store_true", help="Allow missing/empty instrument YAML and use fallbacks.")
    p.add_argument("--employment-default-on-unknown", type=str, default="", help="If set, map unknown employment values to this normalized status.")

    p.add_argument("--enable-test-retest", action="store_true", help="Enable test-retest stability check (default off).")
    p.add_argument("--retest-fraction", type=float, default=0.2, help="Fraction of personas to retest when enabled.")
    p.add_argument("--enable-shuffle-check", action="store_true", help="Enable shuffle-order stability check (default off).")
    p.add_argument("--shuffle-check-fraction", type=float, default=0.2, help="Fraction of personas for shuffle check when enabled.")

    p.add_argument("--output-subdir", type=str, default="simulation_outputs", help="Subdirectory under DATA_DIR to save outputs.")
    return p.parse_args(argv)


def load_data() -> pd.DataFrame:
    data_dir = get_data_dir()
    employment_file = os.path.join(data_dir, "employment_selected_features.csv")
    if not os.path.isfile(employment_file):
        raise FileNotFoundError(
            f"Required input file not found: {employment_file}\n"
            "Place employment_selected_features.csv in DATA_DIR or set PROJECT_ROOT/DATA_PATH accordingly."
        )
    df = pd.read_csv(employment_file)
    if df.shape[0] == 0:
        raise ValueError(f"Input file {employment_file} is empty.")
    return df


def build_personas_and_network(
    df: pd.DataFrame,
    rng: np.random.Generator,
    test: PsychometricTest,
    *,
    employment_default_on_unknown: Optional[str],
    k_neighbors: int = 5,
) -> Tuple[List[Persona], SimpleGraph, np.ndarray, Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    age_col = find_column(df, ["age", "AGE_GROUP"])
    emp_col = find_column(df, ["employment_status", "employment", "job_status", "work_status", "EMPL_STATUS", "empl_status"])
    ls_col = find_column(df, ["LIFE_SATISFACTION", "life_satisfaction", "life satisfaction", "LIFE_SATISFACTION_TOTAL"])

    if emp_col is None:
        raise ValueError("Could not find an employment status column (e.g., employment_status).")
    if ls_col is None:
        raise ValueError("Could not find LIFE_SATISFACTION column (case-insensitive).")

    if age_col is None:
        ages = np.full(df.shape[0], 60.0, dtype=float)
    else:
        # Handle AGE_GROUP format (e.g., "50-59", "60-69") by extracting the lower bound
        if age_col.upper() == "AGE_GROUP":
            def parse_age_group(x):
                if pd.isna(x):
                    return 60.0
                s = str(x).strip()
                # Extract first number from range like "50-59" or single number
                if "-" in s:
                    try:
                        return float(s.split("-")[0])
                    except Exception:
                        return 60.0
                try:
                    return float(s)
                except Exception:
                    return 60.0
            ages = df[age_col].apply(parse_age_group).to_numpy(dtype=float)
        else:
            ages = df[age_col].apply(lambda x: safe_float(x, age_col)).to_numpy(dtype=float)

    # Filter to age >= 50 as required
    mask = ages >= 50.0
    df2 = df.loc[mask].copy()
    ages2 = ages[mask]
    if df2.shape[0] == 0:
        raise ValueError("No rows with age >= 50 found after filtering.")
    dropped_under_50 = int((~mask).sum())

    targets = df2[ls_col].apply(lambda x: safe_float(x, ls_col)).to_numpy(dtype=float)
    
    # Handle LIFE_SATISFACTION_TOTAL: scale from 6-item total (6-24) to per-item average (1-4)
    # If values are in the range 6-24, assume it's a 6-item total and convert to average
    if ls_col.upper() == "LIFE_SATISFACTION_TOTAL" or (np.nanmin(targets) >= 6.0 and np.nanmax(targets) <= 24.0):
        # Scale from 6-item total (6-24) to per-item average (1-4)
        # Formula: (total - 6) / (24 - 6) * (4 - 1) + 1 = (total - 6) / 18 * 3 + 1
        targets = (targets - 6.0) / 18.0 * 3.0 + 1.0
        targets = np.clip(targets, 1.0, 4.0)
    
    if np.nanmin(targets) < 1.0 - 1e-6 or np.nanmax(targets) > 4.0 + 1e-6:
        raise ValueError(
            f"LIFE_SATISFACTION values must be on a 1-4 scale for this simulator; "
            f"found min={float(np.nanmin(targets)):.3f}, max={float(np.nanmax(targets)):.3f} "
            f"(after scaling if needed)."
        )

    # Pick psychosocial-like numeric columns
    numeric_cols: List[str] = []
    for c in df2.columns:
        if c == ls_col or c == emp_col or c == age_col:
            continue
        cl = str(c).lower()
        if any(k in cl for k in PSYCHOSOCIAL_COL_CANDIDATE_KEYWORDS):
            # numeric-ish?
            series = pd.to_numeric(df2[c], errors="coerce")
            if series.notna().mean() >= 0.6:
                numeric_cols.append(c)

    # Compute z-score index over selected columns (mean of z per row)
    if numeric_cols:
        mat = np.vstack([pd.to_numeric(df2[c], errors="coerce").to_numpy(dtype=float) for c in numeric_cols]).T
        col_means = np.nanmean(mat, axis=0)
        col_stds = np.nanstd(mat, axis=0) + 1e-9
        mat_z = (np.where(np.isnan(mat), col_means, mat) - col_means) / col_stds
        psy_index = np.nanmean(mat_z, axis=1)
    else:
        psy_index = np.zeros(df2.shape[0], dtype=float)

    # Profile string from a few representative columns (if any)
    profiles: List[str] = []
    if numeric_cols:
        # choose up to 3 most complete columns
        completeness = [(c, pd.to_numeric(df2[c], errors="coerce").notna().mean()) for c in numeric_cols]
        completeness.sort(key=lambda x: x[1], reverse=True)
        chosen_cols = [c for c, _ in completeness[:3]]
        # compute z for chosen cols
        mat2 = np.vstack([pd.to_numeric(df2[c], errors="coerce").to_numpy(dtype=float) for c in chosen_cols]).T
        means2 = np.nanmean(mat2, axis=0)
        stds2 = np.nanstd(mat2, axis=0) + 1e-9
        mat2_z = (np.where(np.isnan(mat2), means2, mat2) - means2) / stds2
        for i in range(df2.shape[0]):
            parts = [banded_summary(float(mat2_z[i, j]), label=str(chosen_cols[j])) for j in range(len(chosen_cols))]
            profiles.append("; ".join(parts))
    else:
        profiles = ["psychosocial: unspecified"] * df2.shape[0]

    personas: List[Persona] = []
    scale = test.response_scale
    for idx, row in df2.reset_index(drop=False).iterrows():
        raw_emp = row[emp_col]
        emp = normalize_employment_status(raw_emp, default_on_unknown=employment_default_on_unknown)
        age = float(ages2[idx])
        target = float(targets[idx])

        persona_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        level = life_satisfaction_level(target)

        psycho_profile = profiles[idx] if idx < len(profiles) else "psychosocial: unspecified"
        prompt = construct_persona_prompt(
            level=level,
            rng=persona_rng,
            response_scale=scale,
            psychosocial_profile=psycho_profile,
            employment_status=emp,
            age=age,
        )

        features = {str(k): row[k] for k in df2.columns}

        personas.append(
            Persona(
                persona_id=f"P{idx:06d}",
                row_index=int(row["index"]) if "index" in row else int(idx),
                age=age,
                employment_status=emp,
                target_life_satisfaction=target,
                features=features,
                prompt=prompt,
                psychosocial_index=float(psy_index[idx]),
                psychosocial_profile=psycho_profile,
            )
        )

    graph = build_similarity_graph(personas, rng=rng, k=k_neighbors)

    # Exogenous signal: deterministic sinusoid + small noise
    t = np.linspace(0.0, 2.0 * math.pi, len(personas), endpoint=False)
    base = 0.5 * np.sin(t)
    noise = rng.normal(0.0, 0.05, size=len(personas))
    exogenous_series = np.clip(base + noise, -1.0, 1.0).astype(float)

    meta = {
        "age_column_used": age_col,
        "employment_column_used": emp_col,
        "life_satisfaction_column_used": ls_col,
        "psychosocial_columns_used": numeric_cols,
        "n_dropped_under_50": dropped_under_50,
    }
    return personas, graph, exogenous_series, meta


def build_similarity_graph(personas: Sequence[Persona], rng: np.random.Generator, k: int = 5) -> SimpleGraph:
    if k < 1:
        raise ValueError("k must be >= 1.")
    n = len(personas)
    graph = SimpleGraph(n=n)

    def emp_num(s: str) -> float:
        return {"full time": 1.0, "part-time/casual": 0.5, "unemployed": 0.0}[s]

    ages = np.array([p.age for p in personas], dtype=float)
    emps = np.array([emp_num(p.employment_status) for p in personas], dtype=float)

    ages_n = (ages - ages.mean()) / (ages.std() + 1e-9)
    emps_n = (emps - emps.mean()) / (emps.std() + 1e-9)
    feats = np.stack([ages_n, emps_n], axis=1)

    # Use argpartition instead of full sort to reduce cost
    for i in range(n):
        d = np.sum((feats - feats[i]) ** 2, axis=1)
        if n <= k + 1:
            neigh = [j for j in range(n) if j != i]
        else:
            cand = np.argpartition(d, kth=k + 1)[: k + 1]
            cand = [int(j) for j in cand if int(j) != i]
            cand.sort(key=lambda j: float(d[j]))
            neigh = cand[:k]
        for j in neigh:
            graph.add_edge(i, j)

    extra = max(1, n // 50)
    for _ in range(extra):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        graph.add_edge(i, j)
    return graph


def holdout_split(
    df: pd.DataFrame,
    personas: Sequence[Persona],
    graph: SimpleGraph,
    exogenous_series: np.ndarray,
    cfg: HoldoutConfig,
    rng: np.random.Generator,
    *,
    explicit_time_column: str = "",
) -> Dict[str, Any]:
    cfg.validate()
    n = len(personas)
    if graph.n_nodes != n or exogenous_series.shape[0] != n:
        raise ValueError("Graph/exogenous length mismatch.")

    val_n = max(1, int(round(n * cfg.validation_fraction)))

    time_col: Optional[str] = None
    if explicit_time_column.strip():
        if explicit_time_column not in df.columns:
            raise ValueError(f"--time-column={explicit_time_column!r} not found in CSV columns.")
        time_col = explicit_time_column
    else:
        # Prefer exact matches; only use contains match if parse success rate is good.
        exact = None
        lower_map = {c.lower(): c for c in df.columns}
        for cand in cfg.time_column_candidates:
            if cand.lower() in lower_map:
                exact = lower_map[cand.lower()]
                break
        time_col = exact

    order: np.ndarray
    parse_success = None
    if time_col is not None:
        series = df[time_col]
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        parse_success = float(parsed.notna().mean())
        if parse_success >= 0.7:
            order = np.argsort(parsed.fillna(pd.Timestamp.min).to_numpy())
        else:
            # Fallback to numeric if workable; else shuffle
            try:
                vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                if np.isfinite(vals).mean() >= 0.7:
                    order = np.argsort(np.nan_to_num(vals, nan=np.nanmin(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 0.0))
                else:
                    order = np.arange(n)
                    rng.shuffle(order)
                    time_col = None
            except Exception:
                order = np.arange(n)
                rng.shuffle(order)
                time_col = None
    else:
        order = np.arange(n)
        rng.shuffle(order)

    val_idx = set(int(i) for i in order[-val_n:])
    train_idx = [i for i in range(n) if i not in val_idx]
    val_idx_list = sorted(list(val_idx))

    train_personas = [personas[i] for i in train_idx]
    val_personas = [personas[i] for i in val_idx_list]

    train_graph = induced_subgraph(graph, train_idx)
    val_graph = induced_subgraph(graph, val_idx_list)

    train_exo = exogenous_series[train_idx]
    val_exo = exogenous_series[val_idx_list]

    return {
        "train_personas": train_personas,
        "val_personas": val_personas,
        "train_graph": train_graph,
        "val_graph": val_graph,
        "train_exogenous": train_exo,
        "val_exogenous": val_exo,
        "train_indices": train_idx,
        "val_indices": val_idx_list,
        "time_column_used": time_col,
        "time_parse_success_rate": parse_success,
    }


def induced_subgraph(graph: SimpleGraph, idx_list: Sequence[int]) -> SimpleGraph:
    idx_list = list(idx_list)
    m = len(idx_list)
    mapping = {old: new for new, old in enumerate(idx_list)}
    sub = SimpleGraph(n=m)
    mapping_set = set(mapping.keys())
    for old_i in idx_list:
        new_i = mapping[old_i]
        for old_j in graph.neighbors(old_i):
            if old_j in mapping_set:
                sub.add_edge(new_i, mapping[old_j])
    return sub


def save_results(
    output_dir: str,
    records_train: Sequence[Dict[str, Any]],
    records_val: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    calibrated_params: CalibratedParams,
    instrument_meta: Dict[str, Any],
    persona_meta: Dict[str, Any],
) -> None:
    if not output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    os.makedirs(output_dir, exist_ok=True)

    def _json_default(obj: Any) -> Any:
        return json_safe_value(obj)

    def dump_json(path: str, obj: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)

    def dump_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")

    dump_json(os.path.join(output_dir, "metrics.json"), metrics)
    dump_json(os.path.join(output_dir, "calibrated_params.json"), dataclasses.asdict(calibrated_params))
    dump_json(os.path.join(output_dir, "instrument_meta.json"), instrument_meta)
    dump_json(os.path.join(output_dir, "persona_meta.json"), persona_meta)
    dump_jsonl(os.path.join(output_dir, "train_records.jsonl"), list(records_train))
    dump_jsonl(os.path.join(output_dir, "val_records.jsonl"), list(records_val))

    rows: List[Dict[str, Any]] = []
    for r in records_val:
        base: Dict[str, Any] = {
            "persona_id": r.get("persona_id"),
            "row_index": r.get("row_index"),
            "age": r.get("age"),
            "employment_status": r.get("employment_status"),
            "psychosocial_index": r.get("psychosocial_index"),
            "psychosocial_profile": r.get("psychosocial_profile"),
            "test_id": r.get("test_id"),
            "target_LIFE_SATISFACTION": (r.get("target_scores") or {}).get("LIFE_SATISFACTION"),
            "obtained_LIFE_SATISFACTION": (r.get("obtained_scores") or {}).get("LIFE_SATISFACTION"),
            "mode": (r.get("call_metadata") or {}).get("mode"),
            "provider": r.get("provider"),
            "openai_model": r.get("openai_model"),
            "reflection": r.get("reflection", ""),
        }
        for ir in r.get("item_responses", []):
            base[f"item_{ir['item_id']}"] = ir.get("response")
        writing = r.get("writing") or {}
        if isinstance(writing, dict):
            for k, v in writing.items():
                base[f"writing_{k}"] = v

        feats = r.get("features") or {}
        if isinstance(feats, dict):
            for k, v in feats.items():
                base[f"feat_{k}"] = json_safe_value(v)
        rows.append(base)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(output_dir, "analysis_dataset.csv"), index=False, quoting=csv.QUOTE_MINIMAL)


# ---------------------------------------------------------------------
# Main orchestrator (required call order)
# ---------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_cli(argv)
    rng = set_global_seed(args.seed)

    df = load_data()

    allow_missing_instruments = bool(args.allow_missing_instruments)
    test, writing_tasks, instrument_meta = load_instruments(allow_missing_instruments=allow_missing_instruments)

    employment_default_on_unknown = args.employment_default_on_unknown.strip() or None
    if employment_default_on_unknown is not None:
        employment_default_on_unknown = normalize_employment_status(employment_default_on_unknown, default_on_unknown="unemployed")

    personas, graph, exogenous_series, persona_meta = build_personas_and_network(
        df=df,
        rng=rng,
        test=test,
        employment_default_on_unknown=employment_default_on_unknown,
        k_neighbors=5,
    )

    hold_cfg = HoldoutConfig(validation_fraction=float(args.validation_fraction))
    split = holdout_split(
        df=df.loc[df.index.isin([p.row_index for p in personas])] if "row_index" in df.columns else df,
        personas=personas,
        graph=graph,
        exogenous_series=exogenous_series,
        cfg=hold_cfg,
        rng=rng,
        explicit_time_column=str(args.time_column),
    )

    admin_cfg = AdministrationConfig(
        mode=str(args.administration_mode),
        shuffle_items_within_tests=bool(args.shuffle_items_within_tests),
        shuffle_test_order=bool(args.shuffle_test_order),
        fixed_order_for_reproducibility=bool(args.fixed_order_for_reproducibility),
    )
    admin_cfg.validate()

    # Calibration is always done with MockLLM (fast/offline) to avoid API cost.
    calibrator = GridSearchCalibrator(seed=args.seed)
    calibrated_params = calibrator.fit(
        train_personas=split["train_personas"],
        test=test,
        graph=split["train_graph"],
        exogenous_series=np.asarray(split["train_exogenous"], dtype=float),
        admin_cfg=admin_cfg,
        rollout_steps=int(args.rollout_steps),
    )

    # Choose LLM provider for rollout
    if args.llm_provider == "openai":
        llm_train: LLMBase = OpenAILLM(model=str(args.openai_model), max_output_tokens=int(args.openai_max_output_tokens))
        llm_val: LLMBase = OpenAILLM(model=str(args.openai_model), max_output_tokens=int(args.openai_max_output_tokens))
    else:
        llm_train = MockLLM(rng=np.random.default_rng(args.seed), params=calibrated_params)
        llm_val = MockLLM(rng=np.random.default_rng(args.seed + 1), params=calibrated_params)

    simulator_train = Simulator(rng=np.random.default_rng(args.seed), llm=llm_train, admin_cfg=admin_cfg)
    simulator_val = Simulator(rng=np.random.default_rng(args.seed + 1), llm=llm_val, admin_cfg=admin_cfg)

    records_train = simulator_train.rollout(
        personas=split["train_personas"],
        test=test,
        writing_tasks=writing_tasks,
        graph=split["train_graph"],
        exogenous_series=np.asarray(split["train_exogenous"], dtype=float),
        rollout_steps=int(args.rollout_steps),
    )
    records_val = simulator_val.rollout(
        personas=split["val_personas"],
        test=test,
        writing_tasks=writing_tasks,
        graph=split["val_graph"],
        exogenous_series=np.asarray(split["val_exogenous"], dtype=float),
        rollout_steps=int(args.rollout_steps),
    )

    evaluator = Evaluator(rng=np.random.default_rng(args.seed + 2))

    simulator_factory_for_retest: Optional[Callable[[], Simulator]] = None
    if bool(args.enable_test_retest):
        def _retest_factory() -> Simulator:
            retest_admin = AdministrationConfig(
                mode=admin_cfg.mode,
                shuffle_items_within_tests=True,
                shuffle_test_order=False,
                fixed_order_for_reproducibility=False,
            )
            if args.llm_provider == "openai":
                retest_llm: LLMBase = OpenAILLM(model=str(args.openai_model), max_output_tokens=int(args.openai_max_output_tokens))
            else:
                retest_llm = MockLLM(rng=np.random.default_rng(args.seed + 999), params=calibrated_params)
            return Simulator(rng=np.random.default_rng(args.seed + 999), llm=retest_llm, admin_cfg=retest_admin)
        simulator_factory_for_retest = _retest_factory

    simulator_factory_for_shuffle_check: Optional[Callable[[], Simulator]] = None
    if bool(args.enable_shuffle_check):
        def _shuffle_factory() -> Simulator:
            shuffle_admin = AdministrationConfig(
                mode=admin_cfg.mode,
                shuffle_items_within_tests=True,
                shuffle_test_order=False,
                fixed_order_for_reproducibility=False,
            )
            if args.llm_provider == "openai":
                shuffle_llm: LLMBase = OpenAILLM(model=str(args.openai_model), max_output_tokens=int(args.openai_max_output_tokens))
            else:
                shuffle_llm = MockLLM(rng=np.random.default_rng(args.seed + 777), params=calibrated_params)
            return Simulator(rng=np.random.default_rng(args.seed + 777), llm=shuffle_llm, admin_cfg=shuffle_admin)
        simulator_factory_for_shuffle_check = _shuffle_factory

    metrics_val = evaluator.compute_metrics(
        records=records_val,
        test=test,
        simulator_factory_for_retest=simulator_factory_for_retest,
        retest_subset_fraction=float(args.retest_fraction),
        simulator_factory_for_shuffle_check=simulator_factory_for_shuffle_check,
        shuffle_subset_fraction=float(args.shuffle_check_fraction),
    )

    metrics = {
        "validation": metrics_val,
        "metadata": {
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "time_column_used_for_holdout": split.get("time_column_used"),
            "time_parse_success_rate": split.get("time_parse_success_rate"),
            "administration_mode": admin_cfg.mode,
            "shuffle_items_within_tests": admin_cfg.shuffle_items_within_tests,
            "shuffle_test_order": admin_cfg.shuffle_test_order,
            "fixed_order_for_reproducibility": admin_cfg.fixed_order_for_reproducibility,
            "rollout_steps": args.rollout_steps,
            "n_train": len(split["train_personas"]),
            "n_val": len(split["val_personas"]),
            "llm_provider": args.llm_provider,
            "openai_model": args.openai_model if args.llm_provider == "openai" else None,
            "enable_test_retest": bool(args.enable_test_retest),
            "enable_shuffle_check": bool(args.enable_shuffle_check),
        },
    }

    data_dir = get_data_dir()
    output_dir = os.path.join(data_dir, str(args.output_subdir))
    save_results(
        output_dir=output_dir,
        records_train=records_train,
        records_val=records_val,
        metrics=metrics,
        calibrated_params=calibrated_params,
        instrument_meta=instrument_meta,
        persona_meta=persona_meta,
    )
    return 0


# Execute main for both direct execution and sandbox wrapper invocation (unconditional).
try:
    sys.exit(main())
except SystemExit:
    raise
except Exception as e:
    # Clear, non-zero exit behavior while keeping unconditional execution.
    print(f"[simulate.py] ERROR: {e}", file=sys.stderr)
    sys.exit(1)

# # Execute main for both direct execution and sandbox wrapper invocation
# main()