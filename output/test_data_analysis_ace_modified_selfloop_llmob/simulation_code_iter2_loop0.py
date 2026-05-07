from __future__ import annotations

PLAYBOOK_USAGE_JSON = {"used_bullets":[{"id":"calibration-on-training-not-validation","why":"Blueprint requires calibrating parameters by minimizing validation loss; current code calibrates on training."},{"id":"stop-count-distribution-mismatch-and-early-termination","why":"Validation stop-count errors are high; current hazard prematurely truncates days after sampling stop counts."},{"id":"poi-choice-too-random-high-temperature-and-large-candidate-set","why":"Low top-K POI recall and calibrated near-max temperature indicate overly random destination choice and diluted candidate sets."},{"id":"distance-penalty-linear-in-km-dominates-and-miscalibrates-trips","why":"Distance term is linear in km and ignores agent mobility scale; contributes to trip-distance divergence."},{"id":"dwell-time-estimated-from-inter-event-gaps-without-travel-decomposition","why":"Dwell learned from raw gaps double-counts travel when simulator adds travel+dwell; speed calibration becomes unrealistic."},{"id":"transition-model-ignores-poi-level-personal-transitions","why":"Underperformance on transitions/POI recall suggests missing per-agent POI->POI transition priors."},{"id":"stop-count-distribution-smoothing-support-too-narrow","why":"Stop-count KL smoothing is not applied on a common support, inflating divergence artifacts."},{"id":"missing-event-driven-minute-rounding-seconds-retained","why":"Blueprint specifies seconds retained then rounded; parser currently drops seconds."},{"id":"no-colocation-layer-implemented","why":"Blueprint permits occupancy and co-location contacts; current_occupancy never changes and no contact stats exist."}]}
CHANGE_SUMMARY_JSON = {"touched_symbols":[{"symbol":"parse_hhmmss_to_minute_of_day","reason":"Parse optional seconds and round to minutes per blueprint."},{"symbol":"estimate_models_from_training","reason":"Fit dwell-time parameters from gaps with travel-time subtraction to avoid double-counting travel."},{"symbol":"MobilitySimulator._candidate_pois","reason":"Improve candidate generation: include agent anchor POIs and sample extras with popularity/affinity weighting to boost recall."},{"symbol":"MobilitySimulator._choose_poi","reason":"Add personal POI-transition prior, use mobility-radius-normalized log distance decay, and weight global transition by mixture to reduce randomness and improve distances/recall."},{"symbol":"MobilitySimulator.simulate_days","reason":"Gate end-day hazard to avoid shrinking sampled stop count; record visit windows for occupancy/contact computation."},{"symbol":"MobilitySimulator.rollout","reason":"Compute occupancy/contact stats per seed using visit windows without changing rollout return schema."},{"symbol":"Evaluator.compute_metrics","reason":"Apply stop-count smoothing over union support and include stable KL computation per blueprint."},{"symbol":"RandomSearchCalibrator._sample_params","reason":"Bias sampling away from degenerate high-temperature/huge-candidate regimes to improve POI recall."},{"symbol":"RandomSearchCalibrator.fit","reason":"Calibrate against validation dates/ground truth (optionally subsampled) instead of training, per blueprint."},{"symbol":"main","reason":"Route calibrator.fit to validation split while keeping training for model estimation and transitions; preserve orchestrator order."}],"applied_strategies":[{"id":"calibration-on-training-not-validation","applied":true,"note":"Calibration objective now computed on validation (optionally subsampled) while models are fit on training."},{"id":"stop-count-distribution-mismatch-and-early-termination","applied":true,"note":"End-day hazard is gated until most planned stops are completed and time is late, aligning realized stop counts with sampled targets."},{"id":"poi-choice-too-random-high-temperature-and-large-candidate-set","applied":true,"note":"Candidate sets include anchor POIs; extra candidates are weighted; calibration sampling is biased toward lower temperatures and smaller candidate_topk."},{"id":"distance-penalty-linear-in-km-dominates-and-miscalibrates-trips","applied":true,"note":"Distance penalty changed to mobility-radius-normalized log decay."},{"id":"dwell-time-estimated-from-inter-event-gaps-without-travel-decomposition","applied":true,"note":"Dwell fit subtracts estimated travel time where coordinates exist."},{"id":"transition-model-ignores-poi-level-personal-transitions","applied":true,"note":"Per-agent POI->POI transition priors added into POI choice utility."},{"id":"stop-count-distribution-smoothing-support-too-narrow","applied":true,"note":"Stop-count distributions are smoothed on a shared support in compute_metrics."},{"id":"missing-event-driven-minute-rounding-seconds-retained","applied":true,"note":"Seconds are parsed and rounded to minutes with clamping."},{"id":"no-colocation-layer-implemented","applied":true,"note":"Rollout computes per-seed occupancy/contact summary stats and resets POI occupancies; stored on simulator without breaking outputs."}]}

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12
MINUTES_PER_DAY = 24 * 60
DEFAULT_DWELL_FIT_SPEED_KMPH = 30.0

# -----------------------------
# OpenAI LLM utilities (required)
# -----------------------------

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


def get_openai_api_key():
    """Return the OpenAI API key from environment or raise with an actionable message."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000):
    """Call OpenAI Responses API and return extracted text output."""
    api_key = get_openai_api_key()
    if OpenAI is None:
        raise ImportError("OpenAI SDK not available. Install with `pip install openai` to use LLM features.")
    client = OpenAI(api_key=api_key)

    responses_kwargs = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        "max_output_tokens": max_output_tokens,
    }

    resp = client.responses.create(**responses_kwargs)

    def extract_response(resp_obj):
        if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
            return resp_obj.output_text
        try:
            output = getattr(resp_obj, "output", None)
            if output and isinstance(output, list):
                first = output[0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, list) and len(content) > 0:
                        c0 = content[0]
                        if isinstance(c0, dict):
                            text = c0.get("text")
                            if isinstance(text, str):
                                return text
                content = getattr(first, "content", None)
                if isinstance(content, list) and content:
                    c0 = content[0]
                    text = getattr(c0, "text", None)
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        return str(resp_obj)

    return extract_response(resp)


# -----------------------------
# Utilities
# -----------------------------

def set_global_seed(seed: int) -> None:
    """Set global seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)


def require_file(path: str) -> str:
    """
    Validate that a file exists and is readable.

    Returns a normalized absolute path.
    """
    if not path or not isinstance(path, str):
        raise ValueError(f"Expected a file path string, got: {path!r}")
    norm = os.path.abspath(path)
    if not os.path.exists(norm):
        raise FileNotFoundError(f"Required data file not found: {norm}")
    if not os.path.isfile(norm):
        raise FileNotFoundError(f"Required data path is not a file: {norm}")
    return norm


def safe_float(x: Any, ctx: str) -> float:
    """Parse a float with context for actionable error messages."""
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Failed to parse float for {ctx}: {x!r}") from e


def parse_hhmmss_to_minute_of_day(t: str) -> int:
    """
    Parse 'HH:MM' or 'HH:MM:SS' into minute-of-day, rounding seconds to minutes.

    Blueprint alignment:
      - Seconds are retained in parsing but rounded for simulation.
      - Rounding rule: add 1 minute when seconds >= 30.
      - Clamp to [0, 1439].

    The data may include trailing punctuation (e.g., '20:10:00.'); this function
    strips common trailing punctuation to be robust.
    """
    t = str(t).strip()
    t = t.strip(" \t\r\n.,;")  # tolerate '20:10:00.' and similar
    parts = t.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time token: {t!r}")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = 0
    if len(parts) >= 3 and parts[2] != "":
        # Strip any lingering punctuation from seconds part.
        ss_token = str(parts[2]).strip(" \t\r\n.,;")
        if ss_token:
            ss = int(ss_token)
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        raise ValueError(f"Time out of range: {t!r}")
    minute = hh * 60 + mm + (1 if ss >= 30 else 0)
    return int(max(0, min(MINUTES_PER_DAY - 1, minute)))


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(EPS, 1 - a)))
    return r * c


def softmax_sample(rng: np.random.Generator, utilities: np.ndarray, temperature: float) -> int:
    """Sample an index from softmax(utilities / temperature)."""
    if temperature <= 0:
        raise ValueError(f"softmax_temperature must be > 0, got {temperature}")
    u = utilities / temperature
    u = u - np.max(u)
    p = np.exp(u)
    p_sum = float(np.sum(p))
    if not math.isfinite(p_sum) or p_sum <= 0:
        p = np.ones_like(p, dtype=float) / float(len(p))
    else:
        p = p / p_sum
    return int(rng.choice(len(p), p=p))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for discrete distributions (arrays), with internal normalization."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(EPS, float(np.sum(p)))
    q = q / max(EPS, float(np.sum(q)))
    return float(np.sum(p * (np.log(p + EPS) - np.log(q + EPS))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence for discrete distributions (arrays)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(EPS, float(np.sum(p)))
    q = q / max(EPS, float(np.sum(q)))
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def wasserstein_1d(a: Sequence[float], b: Sequence[float]) -> float:
    """Approximate 1D Wasserstein-1 distance by comparing quantiles on a common grid."""
    if not a or not b:
        return float("nan")
    a_sorted = np.sort(np.asarray(a, dtype=float))
    b_sorted = np.sort(np.asarray(b, dtype=float))
    n = len(a_sorted)
    m = len(b_sorted)
    grid = np.linspace(0.0, 1.0, num=max(n, m), endpoint=True)
    qa = np.quantile(a_sorted, grid)
    qb = np.quantile(b_sorted, grid)
    return float(np.mean(np.abs(qa - qb)))


def day_type_from_date(d: dt.date) -> str:
    """Return 'weekday' or 'weekend' for a date."""
    return "weekend" if d.weekday() >= 5 else "weekday"


def ensure_nonempty_mapping(m: Mapping[str, Any], ctx: str) -> None:
    """Validate a JSON-like mapping is non-empty."""
    if not isinstance(m, Mapping) or len(m) == 0:
        raise ValueError(f"Expected non-empty mapping for {ctx}, got: {type(m)}")


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Event:
    """A visit start event at a POI, with mapped categories and minute-of-day timestamp."""
    poi_id: str
    place_type: str
    coarse_category: str
    time_minute: int


@dataclass(frozen=True)
class DayTrajectory:
    """A single-day ordered sequence of events for an agent."""
    agent_id: str
    date: dt.date
    events: Tuple[Event, ...]


@dataclass
class POI:
    """POI node with coordinates and dynamic occupancy state."""
    poi_id: str
    category: str
    coarse_category: str
    latitude: float
    longitude: float
    base_attractiveness: float = 0.0
    current_occupancy: int = 0


@dataclass
class AgentProfile:
    """Per-agent calibrated preferences derived from training trajectories."""
    agent_id: str
    home_poi_candidates: List[Tuple[str, float]]
    anchor_poi_affinity: Dict[str, float]
    category_preference: Dict[str, float]
    time_of_day_priors_by_category: Dict[str, np.ndarray]
    mobility_radius_preference_km: float


@dataclass
class GlobalModels:
    """Population-level models estimated from training trajectories."""
    stop_count_dist_by_day_type: Dict[str, Dict[int, float]]
    global_category_transition: Dict[str, Dict[str, Dict[str, float]]]
    global_start_time_hist_by_day_type: Dict[str, np.ndarray]
    global_time_of_day_hist_by_category: Dict[str, np.ndarray]
    poi_popularity: Dict[str, float]
    poi_transition_graph: Dict[str, List[Tuple[str, float]]]
    dwell_lognormal_params_by_category: Dict[str, Tuple[float, float]]


@dataclass
class SimulatorParams:
    """Calibratable simulator parameters (subset per blueprint, plus small utility weights)."""
    w_personal_transition: float
    w_distance_decay: float
    softmax_temperature: float
    travel_time_speed_kmph: float
    day_end_hazard_scale: float
    day_end_hazard_shift_minute: int
    candidate_set_topk_from_transition_graph: int
    w_personal_poi_affinity: float = 1.0
    w_global_poi_popularity: float = 1.0
    w_global_transition: float = 1.0
    dwell_mu_shift: float = 0.0
    dwell_sigma_scale: float = 1.0

    contact_time_bin_minutes: int = 10
    compute_contacts: bool = True

    def validate(self) -> None:
        """Validate parameter ranges for stability and blueprint-aligned bounds."""
        if not (0.0 <= self.w_personal_transition <= 1.0):
            raise ValueError("w_personal_transition must be in [0,1].")
        if not (0.0 <= self.w_distance_decay <= 10.0):
            raise ValueError("w_distance_decay must be in [0,10].")
        if not (0.05 <= self.softmax_temperature <= 5.0):
            raise ValueError("softmax_temperature must be in [0.05,5.0].")
        if not (5.0 <= self.travel_time_speed_kmph <= 80.0):
            raise ValueError("travel_time_speed_kmph must be in [5,80].")
        if self.day_end_hazard_scale < 0:
            raise ValueError("day_end_hazard_scale must be nonnegative.")
        if not (0 <= self.day_end_hazard_shift_minute <= 24 * 60):
            raise ValueError("day_end_hazard_shift_minute must be within a day.")
        if not (5 <= self.candidate_set_topk_from_transition_graph <= 200):
            raise ValueError("candidate_set_topk_from_transition_graph must be in [5,200].")
        if self.dwell_sigma_scale <= 0:
            raise ValueError("dwell_sigma_scale must be > 0.")
        if self.w_global_transition < 0:
            raise ValueError("w_global_transition must be nonnegative.")
        if self.contact_time_bin_minutes <= 0:
            raise ValueError("contact_time_bin_minutes must be positive.")


@dataclass(frozen=True)
class VisitWindow:
    """Internal visit interval used for occupancy/contact computations."""
    agent_id: str
    date: dt.date
    poi_id: str
    coarse_category: str
    start_minute: int
    end_minute: int


# -----------------------------
# Parsing and loading
# -----------------------------

class DataLoader:
    """Load required JSON inputs from file paths."""
    def __init__(self, agent_json_path: str, poi_json_path: str, catto_json_path: str):
        self.agent_json_path = require_file(agent_json_path)
        self.poi_json_path = require_file(poi_json_path)
        self.catto_json_path = require_file(catto_json_path)

    def load(self) -> Dict[str, Any]:
        """Load all JSONs and validate basic structure."""
        with open(self.agent_json_path, "r", encoding="utf-8") as f:
            agent_raw = json.load(f)
        with open(self.poi_json_path, "r", encoding="utf-8") as f:
            poi_raw = json.load(f)
        with open(self.catto_json_path, "r", encoding="utf-8") as f:
            catto_raw = json.load(f)

        ensure_nonempty_mapping(agent_raw, "1921Y.json")
        ensure_nonempty_mapping(poi_raw, "poi_category_192021_longitude_latitude.json")
        ensure_nonempty_mapping(catto_raw, "catto.json")

        return {"agent_raw": agent_raw, "poi_raw": poi_raw, "catto_raw": catto_raw}


class TrajectoryParser:
    """Parse per-agent daily activity strings into structured trajectories."""
    def __init__(self, place_type_to_coarse: Mapping[str, str], poi_catalog: Mapping[str, POI]):
        self.place_type_to_coarse = dict(place_type_to_coarse)
        self.poi_catalog = poi_catalog

    @staticmethod
    def _extract_date_prefix(s: str) -> Tuple[dt.date, str]:
        """
        Extract a date and the remainder of the daily string.

        Robust to strings formatted as:
          - 'Activities at YYYY-MM-DD: ...'
          - 'Activities at YYYY/MM/DD, ...'
        """
        if "Activities at " not in s:
            raise ValueError(f"Daily string missing 'Activities at ' prefix: {s!r}")
        after = s.split("Activities at ", 1)[1].strip()

        m = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", after)
        if not m:
            raise ValueError(f"Failed to locate date token in daily string: {s!r}")

        date_token = m.group(1).replace("/", "-").strip(" \t:;,-")
        try:
            date_obj = dt.date.fromisoformat(date_token)
        except Exception as e:
            raise ValueError(f"Failed to parse date from token {date_token!r} in string: {s!r}") from e

        rest = after[m.end():].lstrip(" ,:-\t")
        return date_obj, rest

    def _map_to_coarse_category(self, place_type: str, poi_id: str) -> str:
        """Map fine place_type / POI category to coarse category via mapping with fallbacks."""
        if place_type in self.place_type_to_coarse:
            return self.place_type_to_coarse[place_type]
        if "#" in poi_id:
            prefix = poi_id.split("#", 1)[0]
            if prefix in self.place_type_to_coarse:
                return self.place_type_to_coarse[prefix]
            return prefix
        if poi_id in self.poi_catalog:
            return self.poi_catalog[poi_id].coarse_category
        return place_type

    @staticmethod
    def _parse_event_token(token: str) -> Tuple[str, str, str]:
        """Parse a single token into (place_type, poi_id, time_str)."""
        token = token.strip()
        if not token:
            raise ValueError("Empty event token encountered.")

        time_str = None
        place_part = None

        if " at " in token:
            place_part, time_str = token.rsplit(" at ", 1)
        elif "@" in token and ":" in token.split("@")[-1]:
            place_part, time_str = token.rsplit("@", 1)
        else:
            parts = token.split()
            if len(parts) >= 2 and ":" in parts[-1]:
                time_str = parts[-1]
                place_part = " ".join(parts[:-1])
            else:
                raise ValueError(f"Cannot parse event token (no time found): {token!r}")

        place_part = place_part.strip(" ,")
        time_str = str(time_str).strip(" ,")

        if "#" not in place_part:
            place_type = place_part
            poi_id = place_part
        else:
            place_type = place_part.split("#", 1)[0].strip()
            poi_id = place_part.strip()

        if not place_type:
            place_type = poi_id.split("#", 1)[0] if "#" in poi_id else poi_id

        return place_type, poi_id, time_str

    def parse_daily_string(self, agent_id: str, daily_str: str) -> DayTrajectory:
        """Parse a daily activity string into a DayTrajectory with time-rounded minutes."""
        date_obj, rest = self._extract_date_prefix(daily_str)
        if not rest:
            return DayTrajectory(agent_id=agent_id, date=date_obj, events=tuple())

        raw_tokens = [t.strip() for t in rest.split(",") if t.strip()]
        events: List[Event] = []
        for tok in raw_tokens:
            place_type, poi_id, time_str = self._parse_event_token(tok)
            minute = parse_hhmmss_to_minute_of_day(time_str)
            coarse = self._map_to_coarse_category(place_type, poi_id)
            events.append(Event(poi_id=poi_id, place_type=place_type, coarse_category=coarse, time_minute=minute))

        events.sort(key=lambda e: e.time_minute)
        return DayTrajectory(agent_id=agent_id, date=date_obj, events=tuple(events))

    @staticmethod
    def extract_daily_strings_for_agent(agent_value: Any) -> List[str]:
        """Extract list of daily activity strings from various JSON schemas."""
        if isinstance(agent_value, list):
            if not all(isinstance(x, str) for x in agent_value):
                raise ValueError("Agent daily list contains non-string entries.")
            return list(agent_value)
        if isinstance(agent_value, str):
            return [agent_value]
        if isinstance(agent_value, dict):
            if "daily_activity_string" not in agent_value:
                strings = [v for v in agent_value.values() if isinstance(v, str)]
                if strings:
                    return strings
                raise ValueError("Agent dict missing 'daily_activity_string' and no string values found.")
            v = agent_value["daily_activity_string"]
            if isinstance(v, str):
                return [v]
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return list(v)
            raise ValueError("Unsupported type for 'daily_activity_string'.")
        raise ValueError(f"Unsupported agent value type: {type(agent_value)}")


def load_data() -> Dict[str, Any]:
    """
    Load required JSON inputs using environment variables.

    Environment variables:
      - PROJECT_ROOT
      - DATA_PATH

    The path setup lines are intentionally kept exactly as required by the harness.
    """
    import os
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")
    if not PROJECT_ROOT or not DATA_PATH:
        raise EnvironmentError(
            "Environment variables PROJECT_ROOT and DATA_PATH must be set. "
            "Example: export PROJECT_ROOT=/abs/path && export DATA_PATH=data"
        )
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

    try:
        _ = DATA_DIR  # appease linters; DATA_DIR is required by harness contract
    except Exception:
        pass

    data_dir = DATA_DIR
    agent_path = os.path.join(data_dir, "1921Y.json")
    poi_path = os.path.join(data_dir, "poi_category_192021_longitude_latitude.json")
    catto_path = os.path.join(data_dir, "catto.json")

    loader = DataLoader(agent_path, poi_path, catto_path)
    return loader.load()


# -----------------------------
# Build POIs, agents, networks
# -----------------------------

def _extract_place_type_to_coarse(catto_raw: Mapping[str, Any]) -> Dict[str, str]:
    """Extract place_type->coarse mapping from catto.json (supports two common schemas)."""
    if "place_type_to_coarse_category" in catto_raw and isinstance(catto_raw["place_type_to_coarse_category"], dict):
        mapping = catto_raw["place_type_to_coarse_category"]
    else:
        mapping = catto_raw
    ensure_nonempty_mapping(mapping, "place_type_to_coarse_category")
    return {str(k): str(v) for k, v in mapping.items()}


def _build_poi_catalog(
    poi_raw: Mapping[str, Any],
    place_type_to_coarse: Mapping[str, str],
) -> Dict[str, POI]:
    """Build POI catalog keyed by poi_id with coordinates and coarse categories."""
    catalog: Dict[str, POI] = {}
    for category, records in poi_raw.items():
        if not isinstance(records, list):
            raise ValueError(f"POI records for category {category!r} must be a list.")
        for rec in records:
            if not (isinstance(rec, (list, tuple)) and len(rec) >= 3):
                raise ValueError(f"Invalid POI record under category {category!r}: {rec!r}")
            lat = safe_float(rec[0], f"lat for {rec!r}")
            lon = safe_float(rec[1], f"lon for {rec!r}")
            poi_id = str(rec[2])
            coarse = place_type_to_coarse.get(str(category), str(category))
            catalog[poi_id] = POI(
                poi_id=poi_id,
                category=str(category),
                coarse_category=str(coarse),
                latitude=lat,
                longitude=lon,
            )
    if not catalog:
        raise ValueError("POI catalog is empty after parsing poi_category file.")
    return catalog


def build_network_and_agents(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse trajectories for all agents and construct POI catalog."""
    agent_raw = raw["agent_raw"]
    poi_raw = raw["poi_raw"]
    catto_raw = raw["catto_raw"]

    place_type_to_coarse = _extract_place_type_to_coarse(catto_raw)
    poi_catalog = _build_poi_catalog(poi_raw, place_type_to_coarse)

    parser = TrajectoryParser(place_type_to_coarse=place_type_to_coarse, poi_catalog=poi_catalog)

    trajectories_by_agent: Dict[str, List[DayTrajectory]] = {}
    for agent_id, agent_value in agent_raw.items():
        daily_strings = parser.extract_daily_strings_for_agent(agent_value)
        trajs = [parser.parse_daily_string(str(agent_id), s) for s in daily_strings]
        trajs.sort(key=lambda t: t.date)
        trajectories_by_agent[str(agent_id)] = trajs

    if not trajectories_by_agent:
        raise ValueError("No agent trajectories parsed from 1921Y.json")

    return {
        "poi_catalog": poi_catalog,
        "place_type_to_coarse": place_type_to_coarse,
        "trajectories_by_agent": trajectories_by_agent,
    }


# -----------------------------
# Holdout split
# -----------------------------

def holdout_split(
    trajectories_by_agent: Mapping[str, Sequence[DayTrajectory]],
    train_fraction: float = 0.8,
) -> Dict[str, Any]:
    """
    Per-agent temporal holdout split.

    Safeguard: if an agent has only 1 day, allocate it to validation (training empty)
    to ensure a minimum of 1 validation day when possible (per blueprint).
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0,1).")

    train: Dict[str, List[DayTrajectory]] = {}
    val: Dict[str, List[DayTrajectory]] = {}

    for agent_id, days in trajectories_by_agent.items():
        days_sorted = sorted(days, key=lambda d: d.date)
        n = len(days_sorted)
        if n == 0:
            train[agent_id] = []
            val[agent_id] = []
            continue
        if n == 1:
            train[agent_id] = []
            val[agent_id] = list(days_sorted)
            continue

        split = int(math.floor(train_fraction * n))
        split = max(1, split)
        split = min(n - 1, split)  # ensure at least 1 validation day
        train[agent_id] = list(days_sorted[:split])
        val[agent_id] = list(days_sorted[split:])
    return {"train": train, "validation": val}


# -----------------------------
# Model estimation
# -----------------------------

def _normalize_counter(c: Counter, min_prob: float = 0.0) -> Dict[Any, float]:
    """Normalize a Counter to probabilities with optional floor smoothing."""
    total = sum(c.values())
    if total <= 0:
        return {}
    probs = {k: float(v) / float(total) for k, v in c.items()}
    if min_prob > 0 and probs:
        probs = {kk: max(min_prob, p) for kk, p in probs.items()}
        s = sum(probs.values())
        probs = {kk: p / s for kk, p in probs.items()}
    return probs


def _fit_lognormal_params(samples: List[float]) -> Tuple[float, float]:
    """Fit lognormal mu/sigma (log-minutes) with clipping to blueprint-aligned ranges."""
    samples = [float(x) for x in samples if x is not None and math.isfinite(float(x)) and float(x) > 0]
    if len(samples) < 10:
        return (3.5, 0.7)
    logs = np.log(np.asarray(samples, dtype=float))
    mu = float(np.mean(logs))
    sigma = float(np.std(logs))
    sigma = max(0.1, min(2.0, sigma))
    mu = max(0.0, min(8.0, mu))
    return (mu, sigma)


def _hist_1440(minutes: Iterable[int], smoothing: float = 1.0) -> np.ndarray:
    """Build a 1440-bin minute-of-day histogram with additive smoothing."""
    h = np.full(MINUTES_PER_DAY, float(smoothing), dtype=float)
    for m in minutes:
        if 0 <= int(m) < MINUTES_PER_DAY:
            h[int(m)] += 1.0
    h = h / float(np.sum(h))
    return h


def estimate_models_from_training(
    train: Mapping[str, Sequence[DayTrajectory]],
    poi_catalog: Mapping[str, POI],
) -> Tuple[Dict[str, AgentProfile], GlobalModels]:
    """
    Estimate agent-specific profiles and global mobility models from training data.

    Key fix vs baseline:
      - Dwell-time parameters are fit from inter-event gaps with an estimated travel-time
        component subtracted when coordinates are available, to avoid double-counting travel.
    """
    stop_counts_by_day_type: Dict[str, Counter] = {"weekday": Counter(), "weekend": Counter()}
    start_time_by_day_type: Dict[str, List[int]] = {"weekday": [], "weekend": []}
    global_time_by_category: Dict[str, List[int]] = defaultdict(list)
    global_category_transitions_by_day_type: Dict[str, Dict[str, Counter]] = {
        "weekday": defaultdict(Counter),
        "weekend": defaultdict(Counter),
    }
    poi_visit_counts: Counter = Counter()
    poi_edge_counts: Dict[str, Counter] = defaultdict(Counter)
    dwell_gap_by_category: Dict[str, List[float]] = defaultdict(list)

    agent_poi_counts: Dict[str, Counter] = defaultdict(Counter)
    agent_cat_counts: Dict[str, Counter] = defaultdict(Counter)
    agent_time_by_cat: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    agent_night_poi_counts: Dict[str, Counter] = defaultdict(Counter)
    agent_trip_distances: Dict[str, List[float]] = defaultdict(list)

    def poi_coords(poi_id: str) -> Optional[Tuple[float, float]]:
        poi = poi_catalog.get(poi_id)
        if not poi:
            return None
        return poi.latitude, poi.longitude

    for agent_id, days in train.items():
        for day in days:
            dtyp = day_type_from_date(day.date)
            ev = list(day.events)
            stop_counts_by_day_type[dtyp][len(ev)] += 1
            if ev:
                start_time_by_day_type[dtyp].append(ev[0].time_minute)

            for i, e in enumerate(ev):
                poi_visit_counts[e.poi_id] += 1
                agent_poi_counts[agent_id][e.poi_id] += 1
                agent_cat_counts[agent_id][e.coarse_category] += 1
                agent_time_by_cat[agent_id][e.coarse_category].append(e.time_minute)
                global_time_by_category[e.coarse_category].append(e.time_minute)

                if e.time_minute >= 22 * 60 or e.time_minute <= 6 * 60:
                    agent_night_poi_counts[agent_id][e.poi_id] += 1

                if i + 1 < len(ev):
                    nxt = ev[i + 1]
                    global_category_transitions_by_day_type[dtyp][e.coarse_category][nxt.coarse_category] += 1
                    poi_edge_counts[e.poi_id][nxt.poi_id] += 1

                    gap = float(nxt.time_minute - e.time_minute)
                    if gap > 0:
                        dwell_sample = gap
                        c1 = poi_coords(e.poi_id)
                        c2 = poi_coords(nxt.poi_id)
                        if c1 and c2:
                            dist = haversine_km(c1[0], c1[1], c2[0], c2[1])
                            agent_trip_distances[agent_id].append(dist)
                            travel_min_est = (dist / max(EPS, DEFAULT_DWELL_FIT_SPEED_KMPH)) * 60.0
                            dwell_sample = gap - travel_min_est
                        if dwell_sample > 1.0:
                            dwell_gap_by_category[e.coarse_category].append(float(dwell_sample))

    stop_count_dist_by_day_type: Dict[str, Dict[int, float]] = {}
    for dtyp, c in stop_counts_by_day_type.items():
        stop_count_dist_by_day_type[dtyp] = _normalize_counter(c, min_prob=0.0) or {0: 1.0}

    global_start_time_hist_by_day_type = {
        dtyp: _hist_1440(start_time_by_day_type[dtyp], smoothing=1.0) for dtyp in ("weekday", "weekend")
    }

    global_time_of_day_hist_by_category: Dict[str, np.ndarray] = {}
    for cat, mins in global_time_by_category.items():
        global_time_of_day_hist_by_category[cat] = _hist_1440(mins, smoothing=1.0)

    global_category_transition: Dict[str, Dict[str, Dict[str, float]]] = {"weekday": {}, "weekend": {}}
    for dtyp in ("weekday", "weekend"):
        for from_cat, to_counter in global_category_transitions_by_day_type[dtyp].items():
            global_category_transition[dtyp][from_cat] = _normalize_counter(to_counter, min_prob=1e-6)

    pop_probs = _normalize_counter(poi_visit_counts, min_prob=1e-9)
    if not pop_probs:
        pop_probs = {pid: 1.0 / float(len(poi_catalog)) for pid in poi_catalog.keys()}

    poi_transition_graph: Dict[str, List[Tuple[str, float]]] = {}
    for from_poi, to_counter in poi_edge_counts.items():
        items = [(to_poi, float(cnt)) for to_poi, cnt in to_counter.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        poi_transition_graph[from_poi] = items

    dwell_params_by_category: Dict[str, Tuple[float, float]] = {}
    for cat, gaps in dwell_gap_by_category.items():
        dwell_params_by_category[cat] = _fit_lognormal_params(gaps)

    agent_profiles: Dict[str, AgentProfile] = {}
    for agent_id in train.keys():
        poi_counts = agent_poi_counts[agent_id]
        cat_counts = agent_cat_counts[agent_id]
        if sum(poi_counts.values()) <= 0:
            home_candidates = []
            affinity = {}
            cat_pref = {}
            tod_priors = {}
            radius = 2.0
        else:
            night_counts = agent_night_poi_counts[agent_id]
            candidates = night_counts if sum(night_counts.values()) > 0 else poi_counts
            top = candidates.most_common(3)
            total_top = sum(v for _, v in top) or 1
            home_candidates = [(pid, float(v) / float(total_top)) for pid, v in top]

            affinity = _normalize_counter(poi_counts, min_prob=1e-9)
            cat_pref = _normalize_counter(cat_counts, min_prob=1e-9)
            tod_priors = {}
            for cat, mins in agent_time_by_cat[agent_id].items():
                tod_priors[cat] = _hist_1440(mins, smoothing=1.0)

            dists = agent_trip_distances.get(agent_id, [])
            radius = float(np.median(dists)) if dists else 2.0

        agent_profiles[agent_id] = AgentProfile(
            agent_id=agent_id,
            home_poi_candidates=home_candidates,
            anchor_poi_affinity=affinity,
            category_preference=cat_pref,
            time_of_day_priors_by_category=tod_priors,
            mobility_radius_preference_km=radius,
        )

    global_models = GlobalModels(
        stop_count_dist_by_day_type=stop_count_dist_by_day_type,
        global_category_transition=global_category_transition,
        global_start_time_hist_by_day_type=global_start_time_hist_by_day_type,
        global_time_of_day_hist_by_category=global_time_of_day_hist_by_category,
        poi_popularity=pop_probs,
        poi_transition_graph=poi_transition_graph,
        dwell_lognormal_params_by_category=dwell_params_by_category,
    )

    return agent_profiles, global_models


# -----------------------------
# Simulator
# -----------------------------

class MobilitySimulator:
    """Event-driven simulator producing daily trajectories plus optional contact/occupancy summaries."""
    def __init__(
        self,
        poi_catalog: Mapping[str, POI],
        global_models: GlobalModels,
        agent_profiles: Mapping[str, AgentProfile],
    ):
        self.poi_catalog = poi_catalog
        self.global_models = global_models
        self.agent_profiles = agent_profiles
        self._pois_by_coarse_category: Dict[str, List[str]] = defaultdict(list)
        for pid, poi in poi_catalog.items():
            self._pois_by_coarse_category[poi.coarse_category].append(pid)

        self.last_contact_stats_by_seed: Dict[int, Dict[str, Any]] = {}
        self.last_occupancy_stats_by_seed: Dict[int, Dict[str, Any]] = {}

    def _sample_home_poi(self, rng: np.random.Generator, profile: AgentProfile) -> str:
        """Sample an initial/home POI for an agent."""
        if profile.home_poi_candidates:
            pids = [p for p, _ in profile.home_poi_candidates]
            w = np.asarray([w for _, w in profile.home_poi_candidates], dtype=float)
            w = w / max(EPS, float(np.sum(w)))
            return str(rng.choice(pids, p=w))
        if profile.anchor_poi_affinity:
            return max(profile.anchor_poi_affinity.items(), key=lambda kv: kv[1])[0]
        return str(rng.choice(list(self.poi_catalog.keys())))

    def _sample_stop_count(self, rng: np.random.Generator, day_type: str) -> int:
        """Sample number of stops for a day from the global distribution."""
        dist = self.global_models.stop_count_dist_by_day_type.get(day_type, None)
        if not dist:
            return int(rng.integers(1, 5))
        counts = sorted(dist.keys())
        p = np.asarray([dist[c] for c in counts], dtype=float)
        p = p / max(EPS, float(np.sum(p)))
        return int(rng.choice(counts, p=p))

    def _sample_start_time(self, rng: np.random.Generator, day_type: str) -> int:
        """Sample a day start time from the global histogram."""
        hist = self.global_models.global_start_time_hist_by_day_type.get(day_type, None)
        if hist is None or len(hist) != MINUTES_PER_DAY:
            return int(rng.integers(6 * 60, 10 * 60))
        return int(rng.choice(np.arange(MINUTES_PER_DAY), p=hist))

    def _end_day_hazard(self, t_minute: int, params: SimulatorParams) -> float:
        """Compute probability of ending the day at time t."""
        x = params.day_end_hazard_scale * ((t_minute - params.day_end_hazard_shift_minute) / 60.0)
        return float(sigmoid(x))

    def _get_poi_coords(self, poi_id: str) -> Optional[Tuple[float, float]]:
        poi = self.poi_catalog.get(poi_id)
        if not poi:
            return None
        return (poi.latitude, poi.longitude)

    def _distance_km(self, from_poi: str, to_poi: str) -> Optional[float]:
        c1 = self._get_poi_coords(from_poi)
        c2 = self._get_poi_coords(to_poi)
        if not c1 or not c2:
            return None
        return haversine_km(c1[0], c1[1], c2[0], c2[1])

    def _travel_time_minutes(self, distance_km: float, params: SimulatorParams) -> int:
        """Convert distance to travel time in minutes using a calibrated speed."""
        speed = params.travel_time_speed_kmph
        minutes = int(max(1.0, (distance_km / max(EPS, speed)) * 60.0))
        return min(minutes, 6 * 60)

    def _sample_dwell_minutes(self, rng: np.random.Generator, category: str, params: SimulatorParams) -> int:
        """Sample dwell time (minutes) from category-specific lognormal."""
        base = self.global_models.dwell_lognormal_params_by_category.get(category, (3.5, 0.7))
        mu = float(base[0] + params.dwell_mu_shift)
        sigma = float(base[1] * params.dwell_sigma_scale)
        sigma = max(0.1, min(2.0, sigma))
        mu = max(0.0, min(8.0, mu))
        dwell = float(rng.lognormal(mean=mu, sigma=sigma))
        dwell = max(5.0, min(8 * 60.0, dwell))
        return int(round(dwell))

    def _mix_category_transition(
        self,
        profile: AgentProfile,
        day_type: str,
        from_cat: Optional[str],
        t_minute: int,
        params: SimulatorParams,
        personal_transitions: Mapping[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Mixture of personal/global category transitions, modulated by time-of-day priors."""
        w_p = params.w_personal_transition
        w_g = 1.0 - w_p

        if from_cat is None:
            pers = profile.category_preference or {}
            glob = {}
        else:
            pers = personal_transitions.get(from_cat, {}) if personal_transitions else {}
            glob = self.global_models.global_category_transition.get(day_type, {}).get(from_cat, {}) or {}

        cats = set(pers.keys()) | set(glob.keys())
        if not cats:
            cats = set(self.global_models.global_time_of_day_hist_by_category.keys())
        if not cats:
            cats = set(p.coarse_category for p in self.poi_catalog.values())

        probs: Dict[str, float] = {}
        for c in cats:
            p_p = float(pers.get(c, 0.0))
            p_g = float(glob.get(c, 0.0))
            p = w_p * p_p + w_g * p_g

            tod_p = profile.time_of_day_priors_by_category.get(c)
            tod_g = self.global_models.global_time_of_day_hist_by_category.get(c)
            tod_score = 1.0
            if tod_p is not None:
                tod_score *= float(tod_p[t_minute])
            if tod_g is not None:
                tod_score *= float(tod_g[t_minute])

            probs[c] = max(EPS, p) * max(EPS, tod_score)

        s = sum(probs.values())
        if s <= 0:
            uni = 1.0 / float(len(probs)) if probs else 1.0
            return {c: uni for c in probs} or {"Unknown": 1.0}
        return {c: v / s for c, v in probs.items()}

    def _personal_category_transition_probs(
        self,
        agent_train_days: Sequence[DayTrajectory],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-agent category->category transition probabilities from training."""
        counts: Dict[str, Counter] = defaultdict(Counter)
        for day in agent_train_days:
            ev = list(day.events)
            for i in range(len(ev) - 1):
                counts[ev[i].coarse_category][ev[i + 1].coarse_category] += 1
        probs: Dict[str, Dict[str, float]] = {}
        for from_cat, ctr in counts.items():
            probs[from_cat] = _normalize_counter(ctr, min_prob=1e-6)
        return probs

    def _personal_poi_transition_probs(
        self,
        agent_train_days: Sequence[DayTrajectory],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-agent poi->poi transition probabilities from training."""
        counts: Dict[str, Counter] = defaultdict(Counter)
        for day in agent_train_days:
            ev = list(day.events)
            for i in range(len(ev) - 1):
                counts[ev[i].poi_id][ev[i + 1].poi_id] += 1
        probs: Dict[str, Dict[str, float]] = {}
        for from_poi, ctr in counts.items():
            probs[from_poi] = _normalize_counter(ctr, min_prob=1e-9)
        return probs

    def _candidate_pois(
        self,
        rng: np.random.Generator,
        profile: AgentProfile,
        current_poi: str,
        target_coarse_category: str,
        params: SimulatorParams,
    ) -> List[str]:
        """
        Build a candidate POI set.

        Improvements:
          - Always include agent's top affinity POIs for the target category (boost recall).
          - Include topK transition-neighbor POIs from global transition graph filtered by category.
          - Add a limited number of extra candidates sampled with weights based on popularity/affinity.
        """
        candidates: List[str] = []

        if profile.anchor_poi_affinity:
            top_anchor = sorted(profile.anchor_poi_affinity.items(), key=lambda kv: kv[1], reverse=True)[:15]
            for pid, _w in top_anchor:
                poi = self.poi_catalog.get(pid)
                if poi and poi.coarse_category == target_coarse_category:
                    candidates.append(pid)

        graph_list = self.global_models.poi_transition_graph.get(current_poi, [])
        if graph_list:
            topk = int(params.candidate_set_topk_from_transition_graph)
            for to_poi, _w in graph_list[:topk]:
                poi = self.poi_catalog.get(to_poi)
                if poi and poi.coarse_category == target_coarse_category:
                    candidates.append(to_poi)

        pool = self._pois_by_coarse_category.get(target_coarse_category, [])
        if not pool:
            pool = list(self.poi_catalog.keys())

        target_total = max(12, min(40, len(pool)))
        needed = target_total - len(candidates)

        if needed > 0 and pool:
            subset_size = min(250, len(pool))
            subset = list(rng.choice(pool, size=subset_size, replace=False))
            pop = self.global_models.poi_popularity
            affin = profile.anchor_poi_affinity
            w = np.asarray([max(EPS, pop.get(pid, EPS)) * max(EPS, affin.get(pid, 1.0)) for pid in subset], dtype=float)
            w = w / max(EPS, float(np.sum(w)))
            extra_n = min(needed, len(subset))
            extra = list(rng.choice(subset, size=extra_n, replace=False, p=w))
            candidates.extend(extra)

        seen = set()
        unique: List[str] = []
        for pid in candidates:
            if pid not in seen:
                seen.add(pid)
                unique.append(pid)
        return unique

    def _choose_poi(
        self,
        rng: np.random.Generator,
        profile: AgentProfile,
        current_poi: str,
        candidates: List[str],
        params: SimulatorParams,
        personal_poi_transitions_from_current: Optional[Mapping[str, float]] = None,
    ) -> str:
        """
        Choose a POI from candidates using a softmax utility.

        Utility terms:
          - personal POI affinity (log)
          - global POI popularity (log)
          - personal/global POI transition prior (log), mixed by w_personal_transition
          - distance decay using mobility-radius-normalized log distance
        """
        if not candidates:
            return current_poi

        affin = profile.anchor_poi_affinity
        pop = self.global_models.poi_popularity

        global_trans_probs: Dict[str, float] = {}
        graph_list = self.global_models.poi_transition_graph.get(current_poi, [])
        if graph_list:
            total_w = float(sum(max(0.0, w) for _pid, w in graph_list))
            if total_w > 0:
                global_trans_probs = {pid: float(max(0.0, w)) / total_w for pid, w in graph_list}

        w_p = float(params.w_personal_transition)
        w_g = float(1.0 - w_p)

        personal_tp = dict(personal_poi_transitions_from_current or {})

        utilities = np.zeros(len(candidates), dtype=float)
        radius = max(0.25, float(profile.mobility_radius_preference_km))
        for i, pid in enumerate(candidates):
            u = 0.0
            u += params.w_personal_poi_affinity * math.log(affin.get(pid, EPS) + EPS)
            u += params.w_global_poi_popularity * math.log(pop.get(pid, EPS) + EPS)

            if personal_tp:
                ptp = float(personal_tp.get(pid, 0.0))
                if ptp > 0:
                    u += (w_p * 1.0) * math.log(ptp + EPS)

            if global_trans_probs:
                gtp = float(global_trans_probs.get(pid, 0.0))
                if gtp > 0:
                    u += (w_g * params.w_global_transition) * math.log(gtp + EPS)

            dist = self._distance_km(current_poi, pid)
            if dist is not None and dist >= 0:
                u -= params.w_distance_decay * math.log1p(float(dist) / (radius + EPS))

            utilities[i] = u

        idx = softmax_sample(rng, utilities, temperature=params.softmax_temperature)
        return candidates[idx]

    def _simulate_days_with_windows(
        self,
        rng: np.random.Generator,
        agent_id: str,
        dates: Sequence[dt.date],
        agent_train_days_for_transition: Sequence[DayTrajectory],
        params: SimulatorParams,
    ) -> Tuple[List[DayTrajectory], List[VisitWindow]]:
        """Simulate days and also return visit windows for occupancy/contact computations."""
        params.validate()
        profile = self.agent_profiles.get(agent_id)
        if profile is None:
            raise KeyError(f"Missing AgentProfile for agent_id={agent_id!r}")

        personal_cat_transitions = self._personal_category_transition_probs(agent_train_days_for_transition)
        personal_poi_transitions = self._personal_poi_transition_probs(agent_train_days_for_transition)

        out: List[DayTrajectory] = []
        windows: List[VisitWindow] = []

        for date in dates:
            dtyp = day_type_from_date(date)
            n_stops = int(self._sample_stop_count(rng, dtyp))

            current_poi = self._sample_home_poi(rng, profile)
            t = self._sample_start_time(rng, dtyp)
            from_cat: Optional[str] = None

            events: List[Event] = []
            planned = max(0, n_stops)
            min_required = max(0, min(planned, int(round(0.7 * planned))))
            for k in range(planned):
                if t >= MINUTES_PER_DAY:
                    break

                if k >= min_required and t >= max(0, params.day_end_hazard_shift_minute - 180):
                    p_end = self._end_day_hazard(t, params)
                    if rng.random() < p_end:
                        break

                cat_dist = self._mix_category_transition(
                    profile=profile,
                    day_type=dtyp,
                    from_cat=from_cat,
                    t_minute=t,
                    params=params,
                    personal_transitions=personal_cat_transitions,
                )
                cats = list(cat_dist.keys())
                p = np.asarray([cat_dist[c] for c in cats], dtype=float)
                p = p / max(EPS, float(np.sum(p)))
                next_cat = str(rng.choice(cats, p=p))

                candidates = self._candidate_pois(
                    rng=rng,
                    profile=profile,
                    current_poi=current_poi,
                    target_coarse_category=next_cat,
                    params=params,
                )
                next_poi = self._choose_poi(
                    rng=rng,
                    profile=profile,
                    current_poi=current_poi,
                    candidates=candidates,
                    params=params,
                    personal_poi_transitions_from_current=personal_poi_transitions.get(current_poi, {}),
                )

                dist_km = self._distance_km(current_poi, next_poi)
                travel_min = (
                    self._travel_time_minutes(dist_km, params)
                    if dist_km is not None
                    else int(rng.integers(1, 15))
                )
                arrive_min = t + travel_min
                if arrive_min >= MINUTES_PER_DAY:
                    break

                poi_obj = self.poi_catalog.get(next_poi)
                coarse = poi_obj.coarse_category if poi_obj else next_cat
                place_type = poi_obj.category if poi_obj else next_cat

                events.append(
                    Event(poi_id=next_poi, place_type=place_type, coarse_category=coarse, time_minute=arrive_min)
                )

                dwell = self._sample_dwell_minutes(rng, coarse, params)
                depart_min = int(min(MINUTES_PER_DAY, arrive_min + dwell))
                windows.append(
                    VisitWindow(
                        agent_id=agent_id,
                        date=date,
                        poi_id=next_poi,
                        coarse_category=coarse,
                        start_minute=int(arrive_min),
                        end_minute=int(max(arrive_min + 1, depart_min)),
                    )
                )

                t = depart_min
                current_poi = next_poi
                from_cat = coarse

            out.append(DayTrajectory(agent_id=agent_id, date=date, events=tuple(events)))

        return out, windows

    def simulate_days(
        self,
        rng: np.random.Generator,
        agent_id: str,
        dates: Sequence[dt.date],
        agent_train_days_for_transition: Sequence[DayTrajectory],
        params: SimulatorParams,
    ) -> List[DayTrajectory]:
        """Public simulate_days API returning only trajectories (windows are internal)."""
        days, _windows = self._simulate_days_with_windows(
            rng=rng,
            agent_id=agent_id,
            dates=dates,
            agent_train_days_for_transition=agent_train_days_for_transition,
            params=params,
        )
        return days

    def _compute_contacts_and_occupancy_stats(
        self,
        windows: Sequence[VisitWindow],
        params: SimulatorParams,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute co-location contact and occupancy summary statistics.
        """
        if not params.compute_contacts or params.contact_time_bin_minutes <= 0:
            return {"enabled": False}, {"enabled": False}

        bin_size = int(params.contact_time_bin_minutes)
        by_day_poi: Dict[Tuple[dt.date, str], List[VisitWindow]] = defaultdict(list)
        for w in windows:
            if 0 <= w.start_minute < MINUTES_PER_DAY and 0 < w.end_minute <= MINUTES_PER_DAY and w.end_minute > w.start_minute:
                by_day_poi[(w.date, w.poi_id)].append(w)

        total_contact_events = 0
        contact_events_by_category: Counter = Counter()
        max_occupancy_by_poi: Dict[str, int] = defaultdict(int)
        max_occupancy_by_category: Dict[str, int] = defaultdict(int)

        for poi in self.poi_catalog.values():
            poi.current_occupancy = 0

        for (_date, poi_id), ws in by_day_poi.items():
            occ_by_bin: Dict[int, set] = defaultdict(set)
            coarse_cat = ws[0].coarse_category if ws else "Unknown"

            for w in ws:
                b0 = int(w.start_minute // bin_size)
                b1 = int((max(w.start_minute, w.end_minute - 1)) // bin_size)
                for b in range(b0, b1 + 1):
                    occ_by_bin[b].add(w.agent_id)

            for _b, agents in occ_by_bin.items():
                n = len(agents)
                if n <= 0:
                    continue
                poi = self.poi_catalog.get(poi_id)
                if poi:
                    poi.current_occupancy = n
                max_occupancy_by_poi[poi_id] = max(max_occupancy_by_poi[poi_id], n)
                max_occupancy_by_category[coarse_cat] = max(max_occupancy_by_category[coarse_cat], n)

                if n >= 2:
                    total_contact_events += (n * (n - 1)) // 2
                    contact_events_by_category[coarse_cat] += (n * (n - 1)) // 2

        for poi in self.poi_catalog.values():
            poi.current_occupancy = 0

        contact_stats = {
            "enabled": True,
            "time_bin_minutes": bin_size,
            "total_contact_events": int(total_contact_events),
            "contact_events_by_coarse_category": dict(contact_events_by_category),
        }
        occupancy_stats = {
            "enabled": True,
            "max_occupancy_by_poi": dict(max_occupancy_by_poi),
            "max_occupancy_by_coarse_category": dict(max_occupancy_by_category),
        }
        return contact_stats, occupancy_stats

    def rollout(
        self,
        params: SimulatorParams,
        dates_by_agent: Mapping[str, Sequence[dt.date]],
        train_days_by_agent_for_transition: Mapping[str, Sequence[DayTrajectory]],
        seeds: Sequence[int],
    ) -> Dict[int, Dict[str, List[DayTrajectory]]]:
        """
        Run stochastic rollouts for multiple seeds.
        """
        params.validate()
        results: Dict[int, Dict[str, List[DayTrajectory]]] = {}
        self.last_contact_stats_by_seed = {}
        self.last_occupancy_stats_by_seed = {}

        for seed in seeds:
            rng = np.random.default_rng(seed)
            sim_by_agent: Dict[str, List[DayTrajectory]] = {}
            all_windows: List[VisitWindow] = []
            for agent_id, dates in dates_by_agent.items():
                agent_train_days = train_days_by_agent_for_transition.get(agent_id, [])
                days, windows = self._simulate_days_with_windows(
                    rng=rng,
                    agent_id=agent_id,
                    dates=list(dates),
                    agent_train_days_for_transition=agent_train_days,
                    params=params,
                )
                sim_by_agent[agent_id] = days
                all_windows.extend(windows)

            contact_stats, occupancy_stats = self._compute_contacts_and_occupancy_stats(all_windows, params=params)
            self.last_contact_stats_by_seed[int(seed)] = contact_stats
            self.last_occupancy_stats_by_seed[int(seed)] = occupancy_stats

            results[int(seed)] = sim_by_agent
        return results


# -----------------------------
# Evaluation metrics
# -----------------------------

class Evaluator:
    """Compute blueprint-specified evaluation metrics for simulated vs ground-truth trajectories."""
    def __init__(self, poi_catalog: Mapping[str, POI]):
        self.poi_catalog = poi_catalog

    def _flatten_days(self, days_by_agent: Mapping[str, Sequence[DayTrajectory]]) -> List[DayTrajectory]:
        out: List[DayTrajectory] = []
        for _aid, days in days_by_agent.items():
            out.extend(list(days))
        return out

    def _stop_counts(self, days: Sequence[DayTrajectory]) -> List[int]:
        return [len(d.events) for d in days]

    def _category_time_hist(self, days: Sequence[DayTrajectory]) -> Dict[str, np.ndarray]:
        mins_by_cat: Dict[str, List[int]] = defaultdict(list)
        for d in days:
            for e in d.events:
                mins_by_cat[e.coarse_category].append(e.time_minute)
        return {cat: _hist_1440(mins, smoothing=1.0) for cat, mins in mins_by_cat.items()}

    def _category_share_per_day(self, days: Sequence[DayTrajectory]) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for d in days:
            ctr = Counter(e.coarse_category for e in d.events)
            total = sum(ctr.values())
            if total <= 0:
                out.append({})
            else:
                out.append({k: float(v) / float(total) for k, v in ctr.items()})
        return out

    def _transition_matrix(self, days: Sequence[DayTrajectory]) -> Dict[str, Dict[str, float]]:
        ctrs: Dict[str, Counter] = defaultdict(Counter)
        for d in days:
            ev = list(d.events)
            for i in range(len(ev) - 1):
                ctrs[ev[i].coarse_category][ev[i + 1].coarse_category] += 1
        mat: Dict[str, Dict[str, float]] = {}
        for from_cat, ctr in ctrs.items():
            mat[from_cat] = _normalize_counter(ctr, min_prob=1e-6)
        return mat

    def _transition_divergence_fro(self, a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]]) -> float:
        cats = set(a.keys()) | set(b.keys())
        cats_to = set()
        for m in (a, b):
            for _f, row in m.items():
                cats_to |= set(row.keys())
        all_to = sorted(cats_to)
        all_from = sorted(cats)
        if not all_from or not all_to:
            return float("nan")

        A = np.zeros((len(all_from), len(all_to)), dtype=float)
        B = np.zeros((len(all_from), len(all_to)), dtype=float)
        for i, f in enumerate(all_from):
            for j, t in enumerate(all_to):
                A[i, j] = float(a.get(f, {}).get(t, 0.0))
                B[i, j] = float(b.get(f, {}).get(t, 0.0))
        return float(np.linalg.norm(A - B))

    def _trip_distances(self, days: Sequence[DayTrajectory]) -> List[float]:
        dists: List[float] = []
        for d in days:
            ev = list(d.events)
            for i in range(len(ev) - 1):
                p1 = self.poi_catalog.get(ev[i].poi_id)
                p2 = self.poi_catalog.get(ev[i + 1].poi_id)
                if not p1 or not p2:
                    continue
                dist = haversine_km(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
                if math.isfinite(dist):
                    dists.append(float(dist))
        return dists

    def _topk_pois(self, days: Sequence[DayTrajectory], k: int) -> List[str]:
        ctr = Counter()
        for d in days:
            for e in d.events:
                ctr[e.poi_id] += 1
        return [pid for pid, _ in ctr.most_common(k)]

    def compute_metrics(
        self,
        ground_truth_by_agent: Mapping[str, Sequence[DayTrajectory]],
        simulated_by_agent: Mapping[str, Sequence[DayTrajectory]],
        topk: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute all required validation metrics.
        """
        gt_days_all = self._flatten_days(ground_truth_by_agent)
        sim_days_all = self._flatten_days(simulated_by_agent)

        gt_counts = self._stop_counts(gt_days_all)
        sim_counts = self._stop_counts(sim_days_all)

        gt_ctr = Counter(int(c) for c in gt_counts)
        sim_ctr = Counter(int(c) for c in sim_counts)

        support = sorted(set(gt_ctr.keys()) | set(sim_ctr.keys()))
        if not support:
            support = [0]
        alpha = 1.0
        p = np.asarray([gt_ctr.get(k, 0) + alpha for k in support], dtype=float)
        q = np.asarray([sim_ctr.get(k, 0) + alpha for k in support], dtype=float)
        p = p / float(np.sum(p))
        q = q / float(np.sum(q))

        stop_kl = kl_divergence(p, q)
        stop_mean_err = abs(
            (float(np.mean(sim_counts)) if sim_counts else 0.0)
            - (float(np.mean(gt_counts)) if gt_counts else 0.0)
        )

        gt_hist_cat = self._category_time_hist(gt_days_all)
        sim_hist_cat = self._category_time_hist(sim_days_all)
        cats = sorted(set(gt_hist_cat.keys()) | set(sim_hist_cat.keys()))
        jsds: Dict[str, float] = {}
        for c in cats:
            pg = gt_hist_cat.get(c, np.full(MINUTES_PER_DAY, 1.0 / MINUTES_PER_DAY))
            ps = sim_hist_cat.get(c, np.full(MINUTES_PER_DAY, 1.0 / MINUTES_PER_DAY))
            jsds[c] = js_divergence(pg, ps)
        tod_jsd_avg = float(np.mean(list(jsds.values()))) if jsds else float("nan")

        gt_shares = self._category_share_per_day(gt_days_all)
        sim_shares = self._category_share_per_day(sim_days_all)

        def avg_share(shares: List[Dict[str, float]]) -> Dict[str, float]:
            total = Counter()
            n = 0
            for sh in shares:
                for k, v in sh.items():
                    total[k] += float(v)
                n += 1
            if n <= 0:
                return {}
            return {k: v / float(n) for k, v in total.items()}

        gt_avg = avg_share(gt_shares)
        sim_avg = avg_share(sim_shares)
        share_cats = sorted(set(gt_avg.keys()) | set(sim_avg.keys()))
        share_maes = [abs(gt_avg.get(c, 0.0) - sim_avg.get(c, 0.0)) for c in share_cats] or [0.0]
        cat_share_mae = float(np.mean(share_maes))

        gt_mat = self._transition_matrix(gt_days_all)
        sim_mat = self._transition_matrix(sim_days_all)
        trans_div = self._transition_divergence_fro(gt_mat, sim_mat)

        gt_dists = self._trip_distances(gt_days_all)
        sim_dists = self._trip_distances(sim_days_all)
        w1 = wasserstein_1d(gt_dists, sim_dists)

        recalls = []
        for aid, gt_days in ground_truth_by_agent.items():
            sim_days = simulated_by_agent.get(aid, [])
            gt_top = set(self._topk_pois(gt_days, topk))
            sim_pois = set(e.poi_id for d in sim_days for e in d.events)
            if not gt_top:
                continue
            recalls.append(len(gt_top & sim_pois) / float(len(gt_top)))
        topk_recall = float(np.mean(recalls)) if recalls else float("nan")

        return {
            "daily_stop_count_distribution_kl": {"kl": stop_kl, "abs_mean_error": stop_mean_err},
            "time_of_day_histogram_jsd_by_category": {"jsd_by_category": jsds, "jsd_avg": tod_jsd_avg},
            "category_share_per_day_mae": cat_share_mae,
            "transition_matrix_divergence_category": trans_div,
            "trip_distance_distribution_wasserstein": w1,
            "topk_poi_recall": topk_recall,
        }


# -----------------------------
# Calibration
# -----------------------------

class CalibrationResult(Dict[str, Any]):
    """Typed alias for calibration result payload."""
    pass


class BaseCalibrator:
    """Abstract base calibrator."""
    def fit(self, *args: Any, **kwargs: Any) -> CalibrationResult:
        raise NotImplementedError


class RandomSearchCalibrator(BaseCalibrator):
    """Random-search calibrator that optimizes objective on validation data per blueprint."""
    def __init__(
        self,
        simulator: MobilitySimulator,
        evaluator: Evaluator,
        iters: int,
        base_seed: int,
        calib_seeds_per_eval: int = 1,
        topk_recall_k: int = 10,
        max_days_per_agent_for_calibration: int = 5,
        max_agents_for_calibration: int = 300,
    ):
        if iters <= 0:
            raise ValueError("iters must be positive.")
        if calib_seeds_per_eval <= 0:
            raise ValueError("calib_seeds_per_eval must be positive.")
        if max_days_per_agent_for_calibration <= 0:
            raise ValueError("max_days_per_agent_for_calibration must be positive.")
        if max_agents_for_calibration <= 0:
            raise ValueError("max_agents_for_calibration must be positive.")
        self.simulator = simulator
        self.evaluator = evaluator
        self.iters = iters
        self.base_seed = base_seed
        self.calib_seeds_per_eval = calib_seeds_per_eval
        self.topk_recall_k = topk_recall_k
        self.max_days_per_agent_for_calibration = max_days_per_agent_for_calibration
        self.max_agents_for_calibration = max_agents_for_calibration

    def _sample_params(self, rng: np.random.Generator) -> SimulatorParams:
        w_personal_transition = float(rng.uniform(0.0, 1.0))
        w_distance_decay = float(rng.uniform(0.0, 10.0))

        if rng.random() < 0.85:
            lo, hi = 0.05, 3.0
        else:
            lo, hi = 3.0, 5.0
        softmax_temperature = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))

        travel_time_speed_kmph = float(rng.uniform(5.0, 80.0))
        day_end_hazard_scale = float(rng.uniform(0.0, 5.0))
        shift_hour = float(rng.uniform(18.0, 24.0))
        day_end_hazard_shift_minute = int(round(shift_hour * 60.0))

        u = float(rng.uniform(0.0, 1.0))
        candidate_set_topk = int(round(5 + (200 - 5) * (u ** 2)))
        candidate_set_topk = int(max(5, min(200, candidate_set_topk)))

        dwell_mu_shift = float(rng.uniform(-0.5, 0.5))
        dwell_sigma_scale = float(rng.uniform(0.8, 1.2))

        return SimulatorParams(
            w_personal_transition=w_personal_transition,
            w_distance_decay=w_distance_decay,
            softmax_temperature=softmax_temperature,
            travel_time_speed_kmph=travel_time_speed_kmph,
            day_end_hazard_scale=day_end_hazard_scale,
            day_end_hazard_shift_minute=day_end_hazard_shift_minute,
            candidate_set_topk_from_transition_graph=candidate_set_topk,
            dwell_mu_shift=dwell_mu_shift,
            dwell_sigma_scale=dwell_sigma_scale,
        )

    @staticmethod
    def _objective_from_metrics(m: Mapping[str, Any]) -> float:
        stop_kl = float(m["daily_stop_count_distribution_kl"]["kl"])
        stop_abs_mean_err = float(m["daily_stop_count_distribution_kl"]["abs_mean_error"])
        tod_jsd = float(m["time_of_day_histogram_jsd_by_category"]["jsd_avg"])
        cat_share_mae = float(m["category_share_per_day_mae"])
        trans_div = float(m["transition_matrix_divergence_category"])
        w1 = float(m["trip_distance_distribution_wasserstein"])
        topk_rec = float(m["topk_poi_recall"])

        if not math.isfinite(w1):
            w1 = 0.0
        if not math.isfinite(topk_rec):
            topk_rec = 0.0
        if not math.isfinite(tod_jsd):
            tod_jsd = 0.0
        if not math.isfinite(trans_div):
            trans_div = 0.0

        return (
            1.0 * stop_kl
            + 0.05 * stop_abs_mean_err
            + 1.0 * tod_jsd
            + 1.0 * cat_share_mae
            + 1.0 * trans_div
            + 0.1 * w1
            - 0.7 * topk_rec
        )

    @staticmethod
    def _subsample_validation(
        validation_by_agent: Mapping[str, Sequence[DayTrajectory]],
        max_agents: int,
        max_days_per_agent: int,
    ) -> Dict[str, List[DayTrajectory]]:
        agent_ids = sorted(validation_by_agent.keys())
        if len(agent_ids) > max_agents:
            agent_ids = agent_ids[:max_agents]
        out: Dict[str, List[DayTrajectory]] = {}
        for aid in agent_ids:
            days = list(validation_by_agent.get(aid, []))
            days = sorted(days, key=lambda d: d.date)
            out[aid] = days[:max_days_per_agent]
        return out

    def fit(
        self,
        validation_ground_truth_by_agent: Mapping[str, Sequence[DayTrajectory]],
        validation_dates_by_agent: Mapping[str, Sequence[dt.date]],
        train_days_by_agent_for_transition: Mapping[str, Sequence[DayTrajectory]],
    ) -> CalibrationResult:
        rng = np.random.default_rng(self.base_seed + 1000)
        best_params: Optional[SimulatorParams] = None
        best_obj = float("inf")
        history: List[Dict[str, Any]] = []

        calib_seeds = [self.base_seed + 2000 + i for i in range(int(self.calib_seeds_per_eval))]

        val_sub = self._subsample_validation(
            validation_by_agent=validation_ground_truth_by_agent,
            max_agents=self.max_agents_for_calibration,
            max_days_per_agent=self.max_days_per_agent_for_calibration,
        )
        val_dates_sub = {aid: [d.date for d in days] for aid, days in val_sub.items()}

        for i in range(self.iters):
            params = self._sample_params(rng)
            sim_rollouts = self.simulator.rollout(
                params=params,
                dates_by_agent=val_dates_sub,
                train_days_by_agent_for_transition=train_days_by_agent_for_transition,
                seeds=calib_seeds,
            )
            objs = []
            for _seed, sim_by_agent in sim_rollouts.items():
                m = self.evaluator.compute_metrics(
                    ground_truth_by_agent=val_sub,
                    simulated_by_agent=sim_by_agent,
                    topk=self.topk_recall_k,
                )
                objs.append(self._objective_from_metrics(m))
            obj = float(np.mean(objs)) if objs else float("inf")

            history.append({"iter": i, "objective": obj, "params": dataclasses.asdict(params)})

            if obj < best_obj:
                best_obj = obj
                best_params = params

        if best_params is None:
            raise RuntimeError("Calibration failed to produce any parameter set.")

        dwell_params_by_cat = {}
        for cat, (mu, sigma) in self.simulator.global_models.dwell_lognormal_params_by_category.items():
            adj_mu = float(max(0.0, min(8.0, mu + best_params.dwell_mu_shift)))
            adj_sigma = float(max(0.1, min(2.0, sigma * best_params.dwell_sigma_scale)))
            dwell_params_by_cat[cat] = {"mu": adj_mu, "sigma": adj_sigma}

        return CalibrationResult(
            best_params=dataclasses.asdict(best_params),
            best_objective=best_obj,
            best_objective_split="validation",
            history=history,
            dwell_time_params_by_coarse_category=dwell_params_by_cat,
            calibration_subsample={"max_agents": self.max_agents_for_calibration, "max_days_per_agent": self.max_days_per_agent_for_calibration},
        )


# -----------------------------
# Results saving
# -----------------------------

def save_results(output_path: str, calibrated_parameters: Dict[str, Any], evaluation_results_on_validation: Dict[str, Any]) -> None:
    """Save calibration parameters and validation evaluation results as JSON."""
    if not output_path:
        raise ValueError("output_path must be a non-empty string.")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "calibrated_parameters": calibrated_parameters,
        "evaluation_results_on_validation": evaluation_results_on_validation,
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# -----------------------------
# CLI and orchestration
# -----------------------------

def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Daily mobility trajectory simulator with calibration and evaluation.")
    p.add_argument("--output", type=str, required=True, help="Output JSON file path for results.")
    p.add_argument("--seed", type=int, default=123, help="Global random seed.")
    p.add_argument("--calib-iters", type=int, default=25, help="Random-search calibration iterations.")
    p.add_argument("--calib-seeds", type=int, default=1, help="Number of RNG seeds per parameter evaluation during calibration.")
    p.add_argument("--eval-seeds", type=int, default=5, help="Number of stochastic rollouts for validation evaluation.")
    p.add_argument("--topk", type=int, default=10, help="K for topK POI recall metric.")
    return p.parse_args(argv)


def _dates_by_agent(days_by_agent: Mapping[str, Sequence[DayTrajectory]]) -> Dict[str, List[dt.date]]:
    """Extract list of dates per agent from trajectories."""
    return {aid: [d.date for d in days] for aid, days in days_by_agent.items()}


def _as_days_by_agent(days_by_agent: Mapping[str, Sequence[DayTrajectory]]) -> Dict[str, List[DayTrajectory]]:
    """Ensure mapping values are mutable lists."""
    return {aid: list(days) for aid, days in days_by_agent.items()}


def _compute_seed_summary(metrics_by_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed metrics into summary statistics."""
    scalars: Dict[str, List[float]] = defaultdict(list)

    def get_scalar(m: Mapping[str, Any], path: Tuple[str, ...]) -> float:
        cur: Any = m
        for k in path:
            if not isinstance(cur, Mapping) or k not in cur:
                return float("nan")
            cur = cur[k]
        try:
            return float(cur)
        except Exception:
            return float("nan")

    paths = {
        "stop_count_kl": ("daily_stop_count_distribution_kl", "kl"),
        "stop_count_abs_mean_error": ("daily_stop_count_distribution_kl", "abs_mean_error"),
        "tod_jsd_avg": ("time_of_day_histogram_jsd_by_category", "jsd_avg"),
        "category_share_mae": ("category_share_per_day_mae",),
        "transition_divergence": ("transition_matrix_divergence_category",),
        "trip_distance_wasserstein": ("trip_distance_distribution_wasserstein",),
        "topk_poi_recall": ("topk_poi_recall",),
    }

    for _seed, m in metrics_by_seed.items():
        for name, path in paths.items():
            val = get_scalar(m, path)
            if math.isfinite(val):
                scalars[name].append(val)

    summary: Dict[str, Any] = {"by_seed": metrics_by_seed, "summary": {}}
    for name, vals in scalars.items():
        if not vals:
            summary["summary"][name] = {"mean": None, "median": None, "p10": None, "p90": None}
            continue
        vals_sorted = sorted(vals)
        summary["summary"][name] = {
            "mean": float(np.mean(vals_sorted)),
            "median": float(np.median(vals_sorted)),
            "p10": float(np.quantile(vals_sorted, 0.10)),
            "p90": float(np.quantile(vals_sorted, 0.90)),
        }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the full pipeline end-to-end (load → split → fit → simulate → evaluate → save)."""
    args = parse_cli(argv)
    set_global_seed(int(args.seed))

    raw = load_data()

    built = build_network_and_agents(raw)
    poi_catalog: Dict[str, POI] = built["poi_catalog"]
    trajectories_by_agent: Dict[str, List[DayTrajectory]] = built["trajectories_by_agent"]

    split = holdout_split(trajectories_by_agent, train_fraction=0.8)
    train_by_agent: Dict[str, List[DayTrajectory]] = _as_days_by_agent(split["train"])
    val_by_agent: Dict[str, List[DayTrajectory]] = _as_days_by_agent(split["validation"])

    agent_profiles, global_models = estimate_models_from_training(train_by_agent, poi_catalog)

    pop = global_models.poi_popularity
    for pid, poi in poi_catalog.items():
        poi.base_attractiveness = float(math.log1p(pop.get(pid, 0.0)))

    simulator = MobilitySimulator(poi_catalog=poi_catalog, global_models=global_models, agent_profiles=agent_profiles)
    evaluator = Evaluator(poi_catalog=poi_catalog)

    val_dates = _dates_by_agent(val_by_agent)
    calibrator = RandomSearchCalibrator(
        simulator=simulator,
        evaluator=evaluator,
        iters=int(args.calib_iters),
        base_seed=int(args.seed),
        calib_seeds_per_eval=int(args.calib_seeds),
        topk_recall_k=int(args.topk),
    )
    calib_result = calibrator.fit(
        validation_ground_truth_by_agent=val_by_agent,
        validation_dates_by_agent=val_dates,
        train_days_by_agent_for_transition=train_by_agent,
    )

    best_params_dict = dict(calib_result["best_params"])
    best_params_dict["dwell_time_params_by_coarse_category"] = calib_result["dwell_time_params_by_coarse_category"]

    best_params = SimulatorParams(
        w_personal_transition=float(calib_result["best_params"]["w_personal_transition"]),
        w_distance_decay=float(calib_result["best_params"]["w_distance_decay"]),
        softmax_temperature=float(calib_result["best_params"]["softmax_temperature"]),
        travel_time_speed_kmph=float(calib_result["best_params"]["travel_time_speed_kmph"]),
        day_end_hazard_scale=float(calib_result["best_params"]["day_end_hazard_scale"]),
        day_end_hazard_shift_minute=int(calib_result["best_params"]["day_end_hazard_shift_minute"]),
        candidate_set_topk_from_transition_graph=int(calib_result["best_params"]["candidate_set_topk_from_transition_graph"]),
        w_personal_poi_affinity=float(calib_result["best_params"].get("w_personal_poi_affinity", 1.0)),
        w_global_poi_popularity=float(calib_result["best_params"].get("w_global_poi_popularity", 1.0)),
        w_global_transition=float(calib_result["best_params"].get("w_global_transition", 1.0)),
        dwell_mu_shift=float(calib_result["best_params"].get("dwell_mu_shift", 0.0)),
        dwell_sigma_scale=float(calib_result["best_params"].get("dwell_sigma_scale", 1.0)),
    )

    eval_seeds = [int(args.seed) + 5000 + i for i in range(int(args.eval_seeds))]
    sim_val_rollouts = simulator.rollout(
        params=best_params,
        dates_by_agent=val_dates,
        train_days_by_agent_for_transition=train_by_agent,
        seeds=eval_seeds,
    )

    metrics_by_seed: Dict[int, Dict[str, Any]] = {}
    for seed, sim_by_agent in sim_val_rollouts.items():
        metrics_by_seed[int(seed)] = evaluator.compute_metrics(
            ground_truth_by_agent=val_by_agent,
            simulated_by_agent=sim_by_agent,
            topk=int(args.topk),
        )

    evaluation_results_on_validation = _compute_seed_summary(metrics_by_seed)

    evaluation_results_on_validation["interaction_summaries"] = {
        "co_location_contact": simulator.last_contact_stats_by_seed,
        "place_occupancy": simulator.last_occupancy_stats_by_seed,
    }

    calibrated_parameters = {
        "best_params": best_params_dict,
        "best_objective": float(calib_result["best_objective"]),
        "best_objective_split": str(calib_result.get("best_objective_split", "validation")),
        "best_objective_on_validation": float(calib_result["best_objective"]),
        "best_objective_on_training": None,
        "calibration_history": calib_result["history"],
        "calibration_subsample": calib_result.get("calibration_subsample", {}),
    }
    save_results(args.output, calibrated_parameters, evaluation_results_on_validation)


main()