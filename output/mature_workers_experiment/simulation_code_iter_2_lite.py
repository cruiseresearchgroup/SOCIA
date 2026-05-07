import os
import math
import random
import csv
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Literal, Iterable, cast

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

"""
Lite-mode social simulation of older-adult personas and the psychosocial influence of paid work.

Outputs are written under:
  DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

Environment variables:
  PROJECT_ROOT: base directory (defaults to current working directory)
  DATA_PATH: subdirectory name (defaults to "data")
  N_PERSONAS: number of personas to simulate (defaults to 200)
  SEED: global random seed (defaults to 7)
  STREAM_TEXT: "1"/"true" to stream responses.csv incrementally (defaults to 1)
  STREAM_CHUNK_SIZE: chunk size for streamed writes (defaults to 250)

Artifacts produced:
  - agent_attributes.csv: static persona parameters
  - responses.csv: reflective writing responses (baseline/post)
  - psychometrics_long.csv: observed measures at baseline and post
  - change_scores.csv: within-person post-baseline changes
  - group_stats.csv: group summary stats by employment_status
  - smd.csv: standardized mean differences (paid_job - no_paid_job) for changes
  - results.csv: long-form combined results table
  - figure.png: summary bar chart of mean changes by employment status
"""


# -----------------------------
# Path handling (as specified)
# -----------------------------
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.getcwd()
DATA_PATH = os.environ.get("DATA_PATH") or "data"

DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
os.makedirs(DATA_DIR, exist_ok=True)

result_path = os.path.join(DATA_DIR, "results.csv")
picture_path = os.path.join(DATA_DIR, "figure.png")

agent_file = os.path.join(DATA_DIR, "agent_attributes.csv")
text_file = os.path.join(DATA_DIR, "responses.csv")
change_scores_file = os.path.join(DATA_DIR, "change_scores.csv")
psychometrics_long_file = os.path.join(DATA_DIR, "psychometrics_long.csv")
group_stats_file = os.path.join(DATA_DIR, "group_stats.csv")
smd_file = os.path.join(DATA_DIR, "smd.csv")


# -----------------------------
# Utility helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a numeric value into the inclusive range [lo, hi]."""
    return max(lo, min(hi, x))


def logistic(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def seeded_rng(seed: int) -> random.Random:
    """Create a deterministic Python RNG."""
    return random.Random(int(seed))


def safe_to_csv(df: pd.DataFrame, path: str, *, index: bool = False) -> None:
    """Write a DataFrame to CSV with clearer errors."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df.to_csv(path, index=index)
    except Exception as e:
        raise IOError(f"Failed to write CSV to '{path}': {e}") from e


def safe_savefig(fig: Figure, path: str, *, dpi: int = 160) -> None:
    """Save a matplotlib figure to disk and raise a clearer error on failure."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=dpi)
    except Exception as e:
        raise IOError(f"Failed to save figure to '{path}': {e}") from e


def safe_append_csv_rows(
    path: str,
    rows: Sequence[Dict[str, Any]],
    *,
    fieldnames: Sequence[str],
    write_header_if_missing: bool = True,
) -> None:
    """
    Append dict rows to a CSV using Python's csv module (efficient, one file-open).
    """
    if not rows:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        file_exists = os.path.exists(path)
        need_header = True
        if file_exists:
            try:
                need_header = os.path.getsize(path) == 0
            except OSError:
                need_header = True

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            if write_header_if_missing and need_header:
                writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        raise IOError(f"Failed to append CSV rows to '{path}': {e}") from e


def _parse_bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def _parse_int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return int(default)
    return int(v.strip())


# -----------------------------
# Types
# -----------------------------
Timepoint = Literal["baseline", "post"]
PromptType = Literal["meaning_and_work", "daily_life"]

Gender = Literal["woman", "man", "nonbinary"]
MaritalStatus = Literal["married_partnered", "widowed", "divorced_separated", "never_married"]
EmploymentStatus = Literal["paid_job", "no_paid_job"]


# -----------------------------
# Data models
# -----------------------------
@dataclass
class PersonaParams:
    """
    Parameters describing a simulated older-adult persona.
    """

    persona_id: int
    age: int
    gender: Gender
    education_years: int
    marital_status: MaritalStatus
    living_alone: int
    income_level: int
    employment_status: EmploymentStatus
    weekly_work_hours: int

    chronic_conditions: int
    adl_limitations: int
    health_score: float
    cognitive_score: float

    depression_score: float
    loneliness_score: float
    perceived_stress: float
    purpose_in_life: float
    social_support: float
    self_efficacy: float
    financial_strain: float


class PersonaState(TypedDict):
    health_score: float
    cognitive_score: float
    depression_score: float
    loneliness_score: float
    perceived_stress: float
    purpose_in_life: float
    social_support: float
    self_efficacy: float
    financial_strain: float


class EmploymentDeltas(TypedDict):
    health_score: float
    cognitive_score: float
    depression_score: float
    loneliness_score: float
    perceived_stress: float
    purpose_in_life: float
    social_support: float
    self_efficacy: float
    financial_strain: float


class PsychometricObservation(TypedDict):
    timepoint: Timepoint
    depression_score: float
    loneliness_score: float
    perceived_stress: float
    purpose_in_life: float
    social_support: float
    self_efficacy: float
    financial_strain: float
    health_score: float
    cognitive_score: float


class LLMPersona:
    """
    Lite-mode persona proxy (no external LLM calls).
    """

    def __init__(self, params: PersonaParams, rng: random.Random):
        self.params = params
        self.rng = rng
        self.state: PersonaState = {
            "health_score": float(params.health_score),
            "cognitive_score": float(params.cognitive_score),
            "depression_score": float(params.depression_score),
            "loneliness_score": float(params.loneliness_score),
            "perceived_stress": float(params.perceived_stress),
            "purpose_in_life": float(params.purpose_in_life),
            "social_support": float(params.social_support),
            "self_efficacy": float(params.self_efficacy),
            "financial_strain": float(params.financial_strain),
        }

    def apply_employment_influence(self) -> EmploymentDeltas:
        deltas: EmploymentDeltas = {
            "health_score": 0.0,
            "cognitive_score": 0.0,
            "depression_score": 0.0,
            "loneliness_score": 0.0,
            "perceived_stress": 0.0,
            "purpose_in_life": 0.0,
            "social_support": 0.0,
            "self_efficacy": 0.0,
            "financial_strain": 0.0,
        }

        employed = 1 if self.params.employment_status == "paid_job" else 0
        hours = float(self.params.weekly_work_hours)
        hours_factor = clamp(hours / 40.0, 0.0, 1.0)

        age = float(self.params.age)
        chronic = float(self.params.chronic_conditions)
        adl = float(self.params.adl_limitations)

        health_tax = 0.10 * (chronic / 4.0) + 0.10 * (adl / 3.0)
        age_tax = 0.05 * clamp((age - 65.0) / 25.0, 0.0, 1.0)

        def eps(s: float) -> float:
            return self.rng.gauss(0.0, s)

        if employed:
            deltas["financial_strain"] += -0.35 * (0.5 + 0.8 * hours_factor) + eps(0.05)

            deltas["purpose_in_life"] += (0.25 * (0.6 + 0.7 * hours_factor)) * (
                1.0 - 0.7 * health_tax - 0.4 * age_tax
            ) + eps(0.06)
            deltas["self_efficacy"] += (0.18 * (0.6 + 0.7 * hours_factor)) * (
                1.0 - 0.5 * health_tax - 0.3 * age_tax
            ) + eps(0.05)

            social_gain = 0.14 * (0.5 + 0.7 * hours_factor)
            if self.params.living_alone:
                social_gain *= 0.65
            deltas["social_support"] += social_gain + eps(0.05)

            deltas["loneliness_score"] += -0.55 * social_gain * 2.0 + -0.12 * deltas["purpose_in_life"] + eps(0.08)

            deltas["perceived_stress"] += (2.2 * hours_factor) * (0.7 + 1.0 * health_tax + 0.6 * age_tax) + eps(0.6)
        else:
            deltas["financial_strain"] += 0.22 + eps(0.06)
            deltas["purpose_in_life"] += -0.12 + eps(0.06)
            deltas["self_efficacy"] += -0.10 + eps(0.05)

            social_loss = -0.08 * (1.15 if self.params.living_alone else 0.75)
            deltas["social_support"] += social_loss + eps(0.05)

            deltas["loneliness_score"] += 0.35 * (1.0 if self.params.living_alone else 0.7) + (-0.3 * social_loss) + eps(
                0.10
            )

            deltas["perceived_stress"] += -0.7 + 0.9 * (self.state["financial_strain"] - 4.0) + eps(0.8)

        dep_delta = (
            0.08 * deltas["perceived_stress"]
            + 0.35 * deltas["loneliness_score"]
            - 0.55 * deltas["purpose_in_life"]
            + 0.25 * deltas["financial_strain"]
            + eps(0.25)
        )
        deltas["depression_score"] += dep_delta

        if employed:
            deltas["cognitive_score"] += 0.45 * (0.4 + 0.7 * hours_factor) * (1.0 - 0.6 * health_tax) + eps(0.25)
            deltas["health_score"] += -0.25 * deltas["perceived_stress"] + 0.15 * deltas["purpose_in_life"] + eps(0.35)
        else:
            deltas["cognitive_score"] += -0.20 + eps(0.25)
            deltas["health_score"] += -0.12 * chronic + 0.08 * (-deltas["perceived_stress"]) + eps(0.40)

        self.state["health_score"] = clamp(self.state["health_score"] + deltas["health_score"], 0.0, 100.0)
        self.state["cognitive_score"] = clamp(self.state["cognitive_score"] + deltas["cognitive_score"], 0.0, 100.0)
        self.state["depression_score"] = clamp(self.state["depression_score"] + deltas["depression_score"], 0.0, 27.0)
        self.state["loneliness_score"] = clamp(self.state["loneliness_score"] + deltas["loneliness_score"], 0.0, 12.0)
        self.state["perceived_stress"] = clamp(self.state["perceived_stress"] + deltas["perceived_stress"], 0.0, 40.0)
        self.state["purpose_in_life"] = clamp(self.state["purpose_in_life"] + deltas["purpose_in_life"], 1.0, 7.0)
        self.state["social_support"] = clamp(self.state["social_support"] + deltas["social_support"], 1.0, 7.0)
        self.state["self_efficacy"] = clamp(self.state["self_efficacy"] + deltas["self_efficacy"], 1.0, 7.0)
        self.state["financial_strain"] = clamp(self.state["financial_strain"] + deltas["financial_strain"], 1.0, 7.0)
        return deltas

    def administer_psychometric_tests(self, timepoint: Timepoint) -> PsychometricObservation:
        def noise(s: float) -> float:
            return self.rng.gauss(0.0, s)

        return {
            "timepoint": timepoint,
            "depression_score": float(clamp(self.state["depression_score"] + noise(0.9), 0.0, 27.0)),
            "loneliness_score": float(clamp(self.state["loneliness_score"] + noise(0.7), 0.0, 12.0)),
            "perceived_stress": float(clamp(self.state["perceived_stress"] + noise(1.3), 0.0, 40.0)),
            "purpose_in_life": float(clamp(self.state["purpose_in_life"] + noise(0.25), 1.0, 7.0)),
            "social_support": float(clamp(self.state["social_support"] + noise(0.25), 1.0, 7.0)),
            "self_efficacy": float(clamp(self.state["self_efficacy"] + noise(0.25), 1.0, 7.0)),
            "financial_strain": float(clamp(self.state["financial_strain"] + noise(0.25), 1.0, 7.0)),
            "health_score": float(clamp(self.state["health_score"] + noise(1.2), 0.0, 100.0)),
            "cognitive_score": float(clamp(self.state["cognitive_score"] + noise(1.0), 0.0, 100.0)),
        }

    def reflective_writing_task(self, prompt_type: PromptType) -> str:
        employed = self.params.employment_status == "paid_job"

        dep = self.state["depression_score"]
        lonely = self.state["loneliness_score"]
        stress = self.state["perceived_stress"]
        purpose = self.state["purpose_in_life"]
        strain = self.state["financial_strain"]
        support = self.state["social_support"]

        tone = "steady"
        if dep >= 15 or lonely >= 8:
            tone = "downcast"
        elif purpose >= 5.5 and support >= 5.0 and dep <= 7:
            tone = "hopeful"
        elif stress >= 22:
            tone = "strained"

        if employed:
            work_phrase = f"I still do paid work about {self.params.weekly_work_hours} hours a week. "
        else:
            work_phrase = "I am not in paid work right now. "

        if prompt_type == "meaning_and_work":
            base = (
                f"{work_phrase}"
                "At this stage of my life, meaning comes from small routines and the people around me. "
            )
        else:
            base = f"{work_phrase}" "My days are shaped by my energy and what needs doing at home. "

        if tone == "downcast":
            add = (
                "Some mornings I feel heavy and it takes effort to get started. "
                "When I spend too much time alone, my thoughts spiral and I worry about the future. "
            )
        elif tone == "strained":
            add = (
                "Lately I feel pulled in different directions and I notice my patience runs thin. "
                "Even when things go well, my body holds onto tension and I have trouble winding down. "
            )
        elif tone == "hopeful":
            add = (
                "I feel useful, and that helps me stay grounded. "
                "I’ve learned to ask for help and to keep in touch with friends so I don’t drift. "
            )
        else:
            add = (
                "I try to keep balance by doing what I can and letting go of what I can’t control. "
                "I’m paying attention to my health and staying connected where possible. "
            )

        specifics: List[str] = []
        if strain >= 5.2:
            specifics.append("Money has been on my mind more than I’d like.")
        if stress >= 22:
            specifics.append("I notice stress shows up in my sleep and in my shoulders.")
        if purpose <= 3.2:
            specifics.append("Sometimes I wonder what I’m working toward.")
        if support >= 5.5:
            specifics.append("I’m grateful for the people who check in on me.")
        if lonely >= 8:
            specifics.append("I miss having someone to talk to in the evenings.")

        add2 = (" " + " ".join(specifics) + " ") if specifics else " "
        closing = "Overall, I’m trying to be honest about where I am and take the next small step."
        return (base + add + add2 + closing).strip()


class PersonaFactory:
    """Factory to sample internally-consistent persona parameters."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def _sample_categorical(self, items: Sequence[str], probs: Sequence[float]) -> str:
        if len(items) == 0 or len(items) != len(probs):
            raise ValueError("items and probs must be non-empty and of the same length.")
        total = float(sum(float(p) for p in probs))
        if not math.isfinite(total) or total <= 0:
            raise ValueError("probs must sum to a positive finite value.")
        r = self.rng.random() * total
        acc = 0.0
        for it, p in zip(items, probs):
            acc += float(p)
            if r <= acc:
                return str(it)
        return str(items[-1])

    def create_persona(self, persona_id: int) -> PersonaParams:
        age = int(clamp(round(self.rng.gauss(72, 6)), 60, 90))
        gender = cast(Gender, self._sample_categorical(["woman", "man", "nonbinary"], [0.52, 0.46, 0.02]))
        education_years = int(clamp(round(self.rng.gauss(13, 3)), 6, 20))
        marital_status = cast(
            MaritalStatus,
            self._sample_categorical(
                ["married_partnered", "widowed", "divorced_separated", "never_married"],
                [0.52, 0.20, 0.18, 0.10],
            ),
        )
        living_alone = (
            1
            if marital_status in ("widowed", "divorced_separated", "never_married") and self.rng.random() < 0.62
            else 0
        )
        income_level = int(clamp(round(self.rng.gauss(3.1, 1.0)), 1, 5))

        chronic_conditions = int(clamp(round(self.rng.gauss(1.6, 1.2)), 0, 6))
        adl_limitations = int(clamp(round(max(0.0, self.rng.gauss(0.7, 0.9))), 0, 4))

        base_health = 78 - 6.0 * chronic_conditions - 7.5 * adl_limitations + self.rng.gauss(0, 6)
        health_score = float(clamp(base_health, 15.0, 98.0))

        base_cog = 82 - 0.25 * (age - 65) - 1.2 * chronic_conditions + self.rng.gauss(0, 5)
        cognitive_score = float(clamp(base_cog, 35.0, 99.0))

        employ_logit = (
            0.9
            - 0.09 * (age - 65)
            + 0.20 * (income_level - 3)
            + 0.08 * (education_years - 12)
            + 0.02 * (health_score - 70)
            - 0.50 * adl_limitations
            - 0.20 * chronic_conditions
        )
        employed = 1 if self.rng.random() < logistic(employ_logit) else 0
        employment_status: EmploymentStatus = "paid_job" if employed else "no_paid_job"
        weekly_work_hours = int(clamp(round(self.rng.gauss(18, 10)), 1, 45)) if employed else 0

        financial_strain = float(
            clamp(
                4.6 - 0.65 * (income_level - 3) + 0.10 * chronic_conditions + self.rng.gauss(0, 0.6),
                1.0,
                7.0,
            )
        )
        social_support = float(
            clamp(
                4.8
                + (0.7 if marital_status == "married_partnered" else -0.2)
                + (-0.7 if living_alone else 0.0)
                + self.rng.gauss(0, 0.7),
                1.0,
                7.0,
            )
        )
        loneliness_score = float(
            clamp(
                4.6
                + (2.2 if living_alone else 0.0)
                - 0.35 * (social_support - 4.0)
                + 0.15 * chronic_conditions
                + self.rng.gauss(0, 1.2),
                0.0,
                12.0,
            )
        )
        perceived_stress = float(
            clamp(
                15.0
                + 1.0 * chronic_conditions
                + 1.4 * adl_limitations
                + 1.0 * (financial_strain - 4.0)
                + 0.4 * (loneliness_score - 4.0)
                + self.rng.gauss(0, 4.0),
                0.0,
                40.0,
            )
        )
        purpose_in_life = float(
            clamp(
                4.5
                + 0.25 * (social_support - 4.0)
                - 0.15 * (financial_strain - 4.0)
                + 0.06 * (health_score - 70) / 10.0
                + self.rng.gauss(0, 0.7),
                1.0,
                7.0,
            )
        )
        self_efficacy = float(
            clamp(
                4.6
                + 0.10 * (education_years - 12) / 2.0
                + 0.12 * (purpose_in_life - 4.0)
                + 0.08 * (health_score - 70) / 10.0
                + self.rng.gauss(0, 0.7),
                1.0,
                7.0,
            )
        )
        depression_score = float(
            clamp(
                6.5
                + 0.28 * (perceived_stress - 14.0)
                + 0.55 * (loneliness_score - 4.0)
                - 0.95 * (purpose_in_life - 4.0)
                + 0.35 * (financial_strain - 4.0)
                + self.rng.gauss(0, 3.0),
                0.0,
                27.0,
            )
        )

        return PersonaParams(
            persona_id=persona_id,
            age=age,
            gender=gender,
            education_years=education_years,
            marital_status=marital_status,
            living_alone=living_alone,
            income_level=income_level,
            employment_status=employment_status,
            weekly_work_hours=weekly_work_hours,
            chronic_conditions=chronic_conditions,
            adl_limitations=adl_limitations,
            health_score=health_score,
            cognitive_score=cognitive_score,
            depression_score=depression_score,
            loneliness_score=loneliness_score,
            perceived_stress=perceived_stress,
            purpose_in_life=purpose_in_life,
            social_support=social_support,
            self_efficacy=self_efficacy,
            financial_strain=financial_strain,
        )


class AnalysisTables(TypedDict):
    change_df: pd.DataFrame
    group_stats: pd.DataFrame
    smd_df: pd.DataFrame
    results_long: pd.DataFrame


class SocialSimulationLite:
    OUTCOMES: Sequence[str] = (
        "depression_score",
        "loneliness_score",
        "perceived_stress",
        "purpose_in_life",
        "social_support",
        "self_efficacy",
        "financial_strain",
        "health_score",
        "cognitive_score",
    )

    TEXT_COLUMNS: Sequence[str] = ("persona_id", "employment_status", "timepoint", "prompt_type", "text")

    def __init__(
        self,
        n_personas: int = 200,
        seed: int = 7,
        *,
        stream_text_to_disk: bool = True,
        stream_chunk_size: int = 250,
    ):
        self.n_personas = int(n_personas)
        self.seed = int(seed)
        self.stream_text_to_disk = bool(stream_text_to_disk)
        self.stream_chunk_size = int(stream_chunk_size)

        self.rng = seeded_rng(self.seed)
        self.factory = PersonaFactory(self.rng)

        self.personas: List[LLMPersona] = []
        self.agent_df: Optional[pd.DataFrame] = None
        self.text_responses_df: Optional[pd.DataFrame] = None
        self.psychometrics_long_df: Optional[pd.DataFrame] = None
        self.results_tables: Optional[AnalysisTables] = None

    def _persona_rng(self, persona_id: int) -> random.Random:
        mixed = (self.seed * 1000003 + int(persona_id) * 9176 + 0x9E3779B9) & 0xFFFFFFFF
        return seeded_rng(mixed)

    def initialize(self) -> None:
        agents: List[Dict[str, Any]] = []
        self.personas = []
        for i in range(self.n_personas):
            params = self.factory.create_persona(i)
            self.personas.append(LLMPersona(params=params, rng=self._persona_rng(i)))
            agents.append(asdict(params))

        self.agent_df = pd.DataFrame(agents)
        safe_to_csv(self.agent_df, agent_file, index=False)

    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, context: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{context}: missing required columns: {missing}")

    @staticmethod
    def _coerce_numeric_or_raise(
        df: pd.DataFrame,
        cols: Sequence[str],
        *,
        context: str,
        allow_na: bool = False,
    ) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            if (not allow_na) and out[c].isna().any():
                bad = out.loc[out[c].isna(), c].head(5)
                raise TypeError(
                    f"{context}: Column '{c}' has non-numeric or missing values. Examples (up to 5): {bad.to_list()}"
                )
        return out

    @staticmethod
    def _validate_unique_persona_timepoint(df: pd.DataFrame) -> None:
        counts = df.groupby(["persona_id", "timepoint"], dropna=False).size()
        if not counts.empty and int(counts.max()) > 1:
            dup = counts[counts > 1].reset_index(name="n")
            sample = dup.head(10).to_dict(orient="records")
            raise ValueError(
                "Duplicate rows detected for (persona_id, timepoint); cannot compute changes safely. "
                f"Examples: {sample}"
            )

    def run_protocol(self) -> None:
        psych_rows: List[Dict[str, Any]] = []
        stream = self.stream_text_to_disk

        text_rows_mem: List[Dict[str, Any]] = []
        text_buffer: List[Dict[str, Any]] = []

        if stream:
            safe_to_csv(pd.DataFrame(columns=list(self.TEXT_COLUMNS)), text_file, index=False)

        def flush_buffer() -> None:
            nonlocal text_buffer
            if not text_buffer:
                return
            safe_append_csv_rows(text_file, text_buffer, fieldnames=self.TEXT_COLUMNS)
            text_buffer = []

        for persona in self.personas:
            pid = persona.params.persona_id

            baseline = persona.administer_psychometric_tests("baseline")
            psych_rows.append(
                {
                    "persona_id": pid,
                    "employment_status": persona.params.employment_status,
                    "weekly_work_hours": persona.params.weekly_work_hours,
                    **baseline,
                }
            )

            baseline_text_row = {
                "persona_id": pid,
                "employment_status": persona.params.employment_status,
                "timepoint": "baseline",
                "prompt_type": "meaning_and_work",
                "text": persona.reflective_writing_task("meaning_and_work"),
            }
            if stream:
                text_buffer.append(baseline_text_row)
                if len(text_buffer) >= self.stream_chunk_size:
                    flush_buffer()
            else:
                text_rows_mem.append(baseline_text_row)

            deltas = persona.apply_employment_influence()

            post = persona.administer_psychometric_tests("post")
            psych_rows.append(
                {
                    "persona_id": pid,
                    "employment_status": persona.params.employment_status,
                    "weekly_work_hours": persona.params.weekly_work_hours,
                    **post,
                    "delta_health_score_applied": deltas["health_score"],
                    "delta_cognitive_score_applied": deltas["cognitive_score"],
                    "delta_depression_score_applied": deltas["depression_score"],
                    "delta_loneliness_score_applied": deltas["loneliness_score"],
                    "delta_perceived_stress_applied": deltas["perceived_stress"],
                    "delta_purpose_in_life_applied": deltas["purpose_in_life"],
                    "delta_social_support_applied": deltas["social_support"],
                    "delta_self_efficacy_applied": deltas["self_efficacy"],
                    "delta_financial_strain_applied": deltas["financial_strain"],
                }
            )

            post_text_row = {
                "persona_id": pid,
                "employment_status": persona.params.employment_status,
                "timepoint": "post",
                "prompt_type": "daily_life",
                "text": persona.reflective_writing_task("daily_life"),
            }
            if stream:
                text_buffer.append(post_text_row)
                if len(text_buffer) >= self.stream_chunk_size:
                    flush_buffer()
            else:
                text_rows_mem.append(post_text_row)

        if stream:
            flush_buffer()

        self.psychometrics_long_df = pd.DataFrame(psych_rows)

        if stream:
            self.text_responses_df = None
        else:
            self.text_responses_df = pd.DataFrame(text_rows_mem)
            safe_to_csv(self.text_responses_df, text_file, index=False)

    def analyze(self) -> None:
        if self.psychometrics_long_df is None:
            raise RuntimeError("No psychometric data available. Call run_protocol() before analyze().")

        df = self.psychometrics_long_df.copy()

        required = ["persona_id", "timepoint", "employment_status", "weekly_work_hours", *self.OUTCOMES]
        self._require_columns(df, required, context="analyze(psychometrics_long_df)")

        df["employment_status"] = df["employment_status"].astype(str)
        df["timepoint"] = df["timepoint"].astype(str)

        df = self._coerce_numeric_or_raise(
            df,
            ["persona_id", "weekly_work_hours", *self.OUTCOMES],
            context="analyze(psychometrics_long_df)",
            allow_na=True,
        )

        if df["persona_id"].isna().any():
            raise TypeError("analyze(psychometrics_long_df): persona_id contains missing/non-numeric values after coercion.")
        df["persona_id"] = df["persona_id"].astype(int)

        if df["weekly_work_hours"].isna().any():
            raise TypeError(
                "analyze(psychometrics_long_df): weekly_work_hours contains missing/non-numeric values after coercion."
            )

        self._validate_unique_persona_timepoint(df)

        baseline = df[df["timepoint"] == "baseline"].set_index("persona_id")
        post = df[df["timepoint"] == "post"].set_index("persona_id")

        if baseline.empty or post.empty:
            raise RuntimeError("Missing baseline or post timepoint data; cannot compute changes.")

        common_ids = baseline.index.intersection(post.index)
        if common_ids.empty:
            raise RuntimeError("No overlapping persona_id between baseline and post.")

        baseline = baseline.loc[common_ids]
        post = post.loc[common_ids]

        outcomes = list(self.OUTCOMES)

        change_cols: Dict[str, pd.Series] = {}
        for c in outcomes:
            change_cols[f"change_{c}"] = post[c] - baseline[c]

        meta = baseline[["employment_status", "weekly_work_hours"]].copy()
        meta["employment_status"] = meta["employment_status"].astype(str)

        change_df = pd.concat([meta, pd.DataFrame(change_cols, index=common_ids)], axis=1).reset_index()

        agg_spec = {col: ["mean", "std", "count"] for col in change_cols.keys()}
        group_stats = change_df.groupby("employment_status", dropna=False).agg(agg_spec)
        group_stats.columns = [f"{metric}_{stat}" for metric, stat in group_stats.columns]
        group_stats = group_stats.reset_index()

        paid = change_df[change_df["employment_status"] == "paid_job"]
        nopaid = change_df[change_df["employment_status"] == "no_paid_job"]

        smd_rows: List[Dict[str, Any]] = []
        for c in change_cols.keys():
            s_paid = paid[c].dropna()
            s_nopaid = nopaid[c].dropna()
            m1 = float(s_paid.mean()) if len(s_paid) else float("nan")
            m0 = float(s_nopaid.mean()) if len(s_nopaid) else float("nan")
            s1 = float(s_paid.std(ddof=1)) if len(s_paid) > 1 else float("nan")
            s0 = float(s_nopaid.std(ddof=1)) if len(s_nopaid) > 1 else float("nan")
            n1 = int(len(s_paid))
            n0 = int(len(s_nopaid))

            pooled = float("nan")
            if n1 > 1 and n0 > 1 and not (math.isnan(s1) or math.isnan(s0)):
                denom = (n1 + n0 - 2)
                if denom > 0:
                    pooled = math.sqrt(((n1 - 1) * (s1**2) + (n0 - 1) * (s0**2)) / denom)

            smd = (m1 - m0) / pooled if (math.isfinite(pooled) and pooled > 1e-12) else float("nan")
            smd_rows.append(
                {
                    "metric": c,
                    "mean_paid_job": m1,
                    "mean_no_paid_job": m0,
                    "smd_paid_minus_nopaid": smd,
                    "n_paid_job": n1,
                    "n_no_paid_job": n0,
                }
            )

        smd_df = pd.DataFrame(smd_rows)

        gs_long = (
            group_stats.set_index("employment_status")
            .stack()
            .reset_index()
            .rename(columns={"employment_status": "group", "level_1": "metric_stat", 0: "value"})
        )
        gs_long[["metric", "stat"]] = gs_long["metric_stat"].str.rsplit("_", n=1, expand=True)
        gs_long = gs_long.drop(columns=["metric_stat"])
        gs_long.insert(0, "table", "group_stats")

        smd_long = (
            smd_df.melt(
                id_vars=["metric"],
                value_vars=[
                    "mean_paid_job",
                    "mean_no_paid_job",
                    "smd_paid_minus_nopaid",
                    "n_paid_job",
                    "n_no_paid_job",
                ],
                var_name="stat",
                value_name="value",
            )
            .assign(table="smd", group=None)
            .loc[:, ["table", "group", "metric", "stat", "value"]]
        )

        results_long = pd.concat(
            [gs_long.loc[:, ["table", "group", "metric", "stat", "value"]], smd_long],
            ignore_index=True,
        )

        self.results_tables = {
            "change_df": change_df,
            "group_stats": group_stats,
            "smd_df": smd_df,
            "results_long": results_long,
        }

    def visualize(self) -> None:
        if self.results_tables is None:
            raise RuntimeError("No analysis results available. Call analyze() before visualize().")

        change_df = self.results_tables["change_df"].copy()
        metrics = [
            "change_depression_score",
            "change_loneliness_score",
            "change_perceived_stress",
            "change_purpose_in_life",
            "change_financial_strain",
        ]
        for m in metrics:
            if m not in change_df.columns:
                raise KeyError(f"visualize(): missing required metric column '{m}' in change_df.")

        plot_df = change_df.groupby("employment_status")[metrics].mean(numeric_only=True)
        plot_df = plot_df.reindex(["no_paid_job", "paid_job"])

        fig, ax = plt.subplots(figsize=(11, 5.8))
        x = np.arange(len(metrics))
        width = 0.38

        no_vals = plot_df.loc["no_paid_job"].to_numpy() if "no_paid_job" in plot_df.index else np.full(len(metrics), np.nan)
        pa_vals = plot_df.loc["paid_job"].to_numpy() if "paid_job" in plot_df.index else np.full(len(metrics), np.nan)

        ax.bar(x - width / 2, no_vals, width, label="no_paid_job")
        ax.bar(x + width / 2, pa_vals, width, label="paid_job")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("change_", "").replace("_", " ") for m in metrics], rotation=20, ha="right")
        ax.set_ylabel("Mean post - baseline change")
        ax.set_title("Simulated psychosocial changes by employment status (lite-mode personas)")
        ax.legend()
        fig.tight_layout()
        safe_savefig(fig, picture_path, dpi=160)
        plt.close(fig)

    def run(self) -> pd.DataFrame:
        self.initialize()
        self.run_protocol()
        self.analyze()
        self.visualize()

        if self.results_tables is None:
            raise RuntimeError("run(): results_tables not set after analysis.")

        safe_to_csv(self.results_tables["change_df"], change_scores_file, index=False)
        safe_to_csv(self.results_tables["group_stats"], group_stats_file, index=False)
        safe_to_csv(self.results_tables["smd_df"], smd_file, index=False)

        if self.psychometrics_long_df is not None:
            safe_to_csv(self.psychometrics_long_df, psychometrics_long_file, index=False)

        results_long = self.results_tables["results_long"].copy()
        safe_to_csv(results_long, result_path, index=False)
        return results_long

    def save_results(self, path: str, df: pd.DataFrame) -> None:
        """
        Save a results DataFrame to disk.

        This is a convenience wrapper around safe_to_csv.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("save_results(): df must be a pandas DataFrame.")
        safe_to_csv(df, path, index=False)


def main() -> pd.DataFrame:
    n_personas = _parse_int_env("N_PERSONAS", 200)
    seed = _parse_int_env("SEED", 7)
    stream_text = _parse_bool_env("STREAM_TEXT", True)
    stream_chunk_size = _parse_int_env("STREAM_CHUNK_SIZE", 250)

    sim = SocialSimulationLite(
        n_personas=n_personas,
        seed=seed,
        stream_text_to_disk=stream_text,
        stream_chunk_size=stream_chunk_size,
    )
    results_long = sim.run()
    return results_long


main()