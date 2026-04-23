from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12
MINUTES_PER_DAY = 24 * 60


# -----------------------------
# OpenAI LLM utilities (required)
# -----------------------------

def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000):
    api_key = get_openai_api_key()
    try:
        from openai import OpenAI  # lazy import to avoid import-time failure if unused
    except Exception as e:
        raise ImportError(
            "Failed to import OpenAI SDK. Install with `pip install openai` to use LLM features."
        ) from e

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
                # SDK may return objects rather than dicts; try dict-like access first.
                first = output[0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, list) and content:
                        c0 = content[0]
                        if isinstance(c0, dict):
                            text = c0.get("text")
                            if isinstance(text, str):
                                return text
                # Fallback: try attribute access (best-effort)
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
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Failed to parse float for {ctx}: {x!r}") from e


def parse_hhmmss_to_minute_of_day(t: str) -> int:
    t = t.strip()
    parts = t.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time token: {t!r}")
    hh = int(parts[0])
    mm = int(parts[1])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Time out of range: {t!r}")
    return hh * 60 + mm


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(EPS, 1 - a)))
    return r * c


def softmax_sample(rng: np.random.Generator, utilities: np.ndarray, temperature: float) -> int:
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
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(EPS, float(np.sum(p)))
    q = q / max(EPS, float(np.sum(q)))
    return float(np.sum(p * (np.log(p + EPS) - np.log(q + EPS))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(EPS, float(np.sum(p)))
    q = q / max(EPS, float(np.sum(q)))
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def wasserstein_1d(a: Sequence[float], b: Sequence[float]) -> float:
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
    return "weekend" if d.weekday() >= 5 else "weekday"


def ensure_nonempty_mapping(m: Mapping[str, Any], ctx: str) -> None:
    if not isinstance(m, Mapping) or len(m) == 0:
        raise ValueError(f"Expected non-empty mapping for {ctx}, got: {type(m)}")


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Event:
    poi_id: str
    place_type: str
    coarse_category: str
    time_minute: int


@dataclass(frozen=True)
class DayTrajectory:
    agent_id: str
    date: dt.date
    events: Tuple[Event, ...]


@dataclass
class POI:
    poi_id: str
    category: str
    coarse_category: str
    latitude: float
    longitude: float
    base_attractiveness: float = 0.0
    current_occupancy: int = 0


@dataclass
class AgentProfile:
    agent_id: str
    home_poi_candidates: List[Tuple[str, float]]
    anchor_poi_affinity: Dict[str, float]
    category_preference: Dict[str, float]
    time_of_day_priors_by_category: Dict[str, np.ndarray]
    mobility_radius_preference_km: float


@dataclass
class GlobalModels:
    stop_count_dist_by_day_type: Dict[str, Dict[int, float]]
    global_category_transition: Dict[str, Dict[str, Dict[str, float]]]
    global_start_time_hist_by_day_type: Dict[str, np.ndarray]
    global_time_of_day_hist_by_category: Dict[str, np.ndarray]
    poi_popularity: Dict[str, float]
    poi_transition_graph: Dict[str, List[Tuple[str, float]]]
    dwell_lognormal_params_by_category: Dict[str, Tuple[float, float]]


@dataclass
class SimulatorParams:
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

    def validate(self) -> None:
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


# -----------------------------
# Parsing and loading
# -----------------------------

class DataLoader:
    def __init__(self, agent_json_path: str, poi_json_path: str, catto_json_path: str):
        self.agent_json_path = require_file(agent_json_path)
        self.poi_json_path = require_file(poi_json_path)
        self.catto_json_path = require_file(catto_json_path)

    def load(self) -> Dict[str, Any]:
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
    def __init__(self, place_type_to_coarse: Mapping[str, str], poi_catalog: Mapping[str, POI]):
        self.place_type_to_coarse = dict(place_type_to_coarse)
        self.poi_catalog = poi_catalog

    @staticmethod
    def _extract_date_prefix(s: str) -> Tuple[dt.date, str]:
        if "Activities at " not in s:
            raise ValueError(f"Daily string missing 'Activities at ' prefix: {s!r}")
        after = s.split("Activities at ", 1)[1].strip()
        date_token = after.split(",", 1)[0].strip()
        date_token = date_token.split()[0].strip()
        date_token = date_token.replace("/", "-")
        try:
            date_obj = dt.date.fromisoformat(date_token)
        except Exception as e:
            raise ValueError(f"Failed to parse date from token {date_token!r} in string: {s!r}") from e
        rest = after[len(date_token):].lstrip(" ,:-")
        return date_obj, rest

    def _map_to_coarse_category(self, place_type: str, poi_id: str) -> str:
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
        time_str = time_str.strip(" ,")

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


def load_data_from_env() -> Dict[str, Any]:
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.environ.get("DATA_PATH")
    if not project_root or not data_path:
        raise EnvironmentError(
            "Environment variables PROJECT_ROOT and DATA_PATH must be set. "
            "Example: export PROJECT_ROOT=/abs/path && export DATA_PATH=data"
        )

    data_dir = os.path.join(project_root, data_path)
    agent_path = os.path.join(data_dir, "1921Y.json")
    poi_path = os.path.join(data_dir, "poi_category_192021_longitude_latitude.json")
    catto_path = os.path.join(data_dir, "catto.json")

    loader = DataLoader(agent_path, poi_path, catto_path)
    return loader.load()


# -----------------------------
# Build POIs, agents, networks
# -----------------------------

def _extract_place_type_to_coarse(catto_raw: Mapping[str, Any]) -> Dict[str, str]:
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
        split = int(math.floor(train_fraction * n))
        split = max(1, split)
        if n >= 2:
            split = min(n - 1, split)
        else:
            split = n
        train[agent_id] = list(days_sorted[:split])
        val[agent_id] = list(days_sorted[split:])
    return {"train": train, "validation": val}


# -----------------------------
# Model estimation
# -----------------------------

def _normalize_counter(c: Counter, min_prob: float = 0.0) -> Dict[Any, float]:
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
        if poi_id in poi_catalog:
            p = poi_catalog[poi_id]
            return p.latitude, p.longitude
        return None

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
                        dwell_gap_by_category[e.coarse_category].append(gap)

                    c1 = poi_coords(e.poi_id)
                    c2 = poi_coords(nxt.poi_id)
                    if c1 and c2:
                        dist = haversine_km(c1[0], c1[1], c2[0], c2[1])
                        agent_trip_distances[agent_id].append(dist)

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

    def _sample_home_poi(self, rng: np.random.Generator, profile: AgentProfile) -> str:
        if profile.home_poi_candidates:
            pids = [p for p, _ in profile.home_poi_candidates]
            w = np.asarray([w for _, w in profile.home_poi_candidates], dtype=float)
            w = w / max(EPS, float(np.sum(w)))
            return str(rng.choice(pids, p=w))
        if profile.anchor_poi_affinity:
            return max(profile.anchor_poi_affinity.items(), key=lambda kv: kv[1])[0]
        return str(rng.choice(list(self.poi_catalog.keys())))

    def _sample_stop_count(self, rng: np.random.Generator, day_type: str) -> int:
        dist = self.global_models.stop_count_dist_by_day_type.get(day_type, None)
        if not dist:
            return int(rng.integers(1, 5))
        counts = sorted(dist.keys())
        p = np.asarray([dist[c] for c in counts], dtype=float)
        p = p / max(EPS, float(np.sum(p)))
        return int(rng.choice(counts, p=p))

    def _sample_start_time(self, rng: np.random.Generator, day_type: str) -> int:
        hist = self.global_models.global_start_time_hist_by_day_type.get(day_type, None)
        if hist is None or len(hist) != MINUTES_PER_DAY:
            return int(rng.integers(6 * 60, 10 * 60))
        return int(rng.choice(np.arange(MINUTES_PER_DAY), p=hist))

    def _end_day_hazard(self, t_minute: int, params: SimulatorParams) -> float:
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
        speed = params.travel_time_speed_kmph
        minutes = int(max(1.0, (distance_km / max(EPS, speed)) * 60.0))
        return min(minutes, 6 * 60)

    def _sample_dwell_minutes(self, rng: np.random.Generator, category: str, params: SimulatorParams) -> int:
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

    def _candidate_pois(
        self,
        rng: np.random.Generator,
        current_poi: str,
        target_coarse_category: str,
        params: SimulatorParams,
    ) -> List[str]:
        candidates: List[str] = []

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

        needed = max(10, min(50, len(pool))) - len(candidates)
        if needed > 0 and pool:
            extra = list(rng.choice(pool, size=min(needed, len(pool)), replace=False))
            candidates.extend(extra)

        seen = set()
        unique = []
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
    ) -> str:
        if not candidates:
            return current_poi

        affin = profile.anchor_poi_affinity
        pop = self.global_models.poi_popularity

        utilities = np.zeros(len(candidates), dtype=float)
        for i, pid in enumerate(candidates):
            u = 0.0
            u += params.w_personal_poi_affinity * math.log(affin.get(pid, EPS) + EPS)
            u += params.w_global_poi_popularity * math.log(pop.get(pid, EPS) + EPS)

            dist = self._distance_km(current_poi, pid)
            if dist is not None:
                u -= params.w_distance_decay * float(dist)
            utilities[i] = u

        idx = softmax_sample(rng, utilities, temperature=params.softmax_temperature)
        return candidates[idx]

    def _personal_category_transition_probs(
        self,
        agent_train_days: Sequence[DayTrajectory],
    ) -> Dict[str, Dict[str, float]]:
        counts: Dict[str, Counter] = defaultdict(Counter)
        for day in agent_train_days:
            ev = list(day.events)
            for i in range(len(ev) - 1):
                counts[ev[i].coarse_category][ev[i + 1].coarse_category] += 1
        probs: Dict[str, Dict[str, float]] = {}
        for from_cat, ctr in counts.items():
            probs[from_cat] = _normalize_counter(ctr, min_prob=1e-6)
        return probs

    def simulate_days(
        self,
        rng: np.random.Generator,
        agent_id: str,
        dates: Sequence[dt.date],
        agent_train_days_for_transition: Sequence[DayTrajectory],
        params: SimulatorParams,
    ) -> List[DayTrajectory]:
        params.validate()
        profile = self.agent_profiles.get(agent_id)
        if profile is None:
            raise KeyError(f"Missing AgentProfile for agent_id={agent_id!r}")

        personal_transitions = self._personal_category_transition_probs(agent_train_days_for_transition)

        out: List[DayTrajectory] = []
        for date in dates:
            dtyp = day_type_from_date(date)
            n_stops = self._sample_stop_count(rng, dtyp)

            current_poi = self._sample_home_poi(rng, profile)
            t = self._sample_start_time(rng, dtyp)
            from_cat: Optional[str] = None

            events: List[Event] = []
            for _k in range(max(0, int(n_stops))):
                if t >= MINUTES_PER_DAY:
                    break
                p_end = self._end_day_hazard(t, params)
                if rng.random() < p_end:
                    break

                cat_dist = self._mix_category_transition(
                    profile=profile,
                    day_type=dtyp,
                    from_cat=from_cat,
                    t_minute=t,
                    params=params,
                    personal_transitions=personal_transitions,
                )
                cats = list(cat_dist.keys())
                p = np.asarray([cat_dist[c] for c in cats], dtype=float)
                p = p / max(EPS, float(np.sum(p)))
                next_cat = str(rng.choice(cats, p=p))

                candidates = self._candidate_pois(
                    rng=rng,
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

                events.append(Event(poi_id=next_poi, place_type=place_type, coarse_category=coarse, time_minute=arrive_min))

                dwell = self._sample_dwell_minutes(rng, coarse, params)
                t = arrive_min + dwell
                current_poi = next_poi
                from_cat = coarse

            out.append(DayTrajectory(agent_id=agent_id, date=date, events=tuple(events)))
        return out

    def rollout(
        self,
        params: SimulatorParams,
        dates_by_agent: Mapping[str, Sequence[dt.date]],
        train_days_by_agent_for_transition: Mapping[str, Sequence[DayTrajectory]],
        seeds: Sequence[int],
    ) -> Dict[int, Dict[str, List[DayTrajectory]]]:
        params.validate()
        results: Dict[int, Dict[str, List[DayTrajectory]]] = {}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sim_by_agent: Dict[str, List[DayTrajectory]] = {}
            for agent_id, dates in dates_by_agent.items():
                agent_train_days = train_days_by_agent_for_transition.get(agent_id, [])
                sim_by_agent[agent_id] = self.simulate_days(
                    rng=rng,
                    agent_id=agent_id,
                    dates=list(dates),
                    agent_train_days_for_transition=agent_train_days,
                    params=params,
                )
            results[int(seed)] = sim_by_agent
        return results


# -----------------------------
# Evaluation metrics
# -----------------------------

class Evaluator:
    def __init__(self, poi_catalog: Mapping[str, POI]):
        self.poi_catalog = poi_catalog

    def _flatten_days(self, days_by_agent: Mapping[str, Sequence[DayTrajectory]]) -> List[DayTrajectory]:
        out: List[DayTrajectory] = []
        for _aid, days in days_by_agent.items():
            out.extend(list(days))
        return out

    def _stop_counts(self, days: Sequence[DayTrajectory]) -> List[int]:
        return [len(d.events) for d in days]

    def _stop_count_dist(self, counts: Sequence[int], smoothing: float = 1.0) -> Dict[int, float]:
        ctr = Counter(int(c) for c in counts)
        if not ctr:
            return {0: 1.0}
        support = sorted(ctr.keys())
        smoothed = {k: float(ctr.get(k, 0)) + smoothing for k in support}
        s = sum(smoothed.values())
        return {k: v / s for k, v in smoothed.items()}

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
        gt_days_all = self._flatten_days(ground_truth_by_agent)
        sim_days_all = self._flatten_days(simulated_by_agent)

        gt_counts = self._stop_counts(gt_days_all)
        sim_counts = self._stop_counts(sim_days_all)
        gt_dist = self._stop_count_dist(gt_counts, smoothing=1.0)
        sim_dist = self._stop_count_dist(sim_counts, smoothing=1.0)
        support = sorted(set(gt_dist.keys()) | set(sim_dist.keys()))
        p = np.asarray([gt_dist.get(x, EPS) for x in support], dtype=float) + 1e-9
        q = np.asarray([sim_dist.get(x, EPS) for x in support], dtype=float) + 1e-9
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
    pass


class BaseCalibrator:
    def fit(self, *args: Any, **kwargs: Any) -> CalibrationResult:
        raise NotImplementedError


class RandomSearchCalibrator(BaseCalibrator):
    def __init__(
        self,
        simulator: MobilitySimulator,
        evaluator: Evaluator,
        iters: int,
        base_seed: int,
        calib_seeds_per_eval: int = 1,
        topk_recall_k: int = 10,
    ):
        if iters <= 0:
            raise ValueError("iters must be positive.")
        self.simulator = simulator
        self.evaluator = evaluator
        self.iters = iters
        self.base_seed = base_seed
        self.calib_seeds_per_eval = calib_seeds_per_eval
        self.topk_recall_k = topk_recall_k

    def _sample_params(self, rng: np.random.Generator) -> SimulatorParams:
        w_personal_transition = float(rng.uniform(0.0, 1.0))
        w_distance_decay = float(rng.uniform(0.0, 10.0))
        softmax_temperature = float(rng.uniform(0.05, 5.0))
        travel_time_speed_kmph = float(rng.uniform(5.0, 80.0))
        day_end_hazard_scale = float(rng.uniform(0.0, 5.0))
        shift_hour = float(rng.uniform(18.0, 24.0))
        day_end_hazard_shift_minute = int(round(shift_hour * 60.0))
        candidate_set_topk = int(rng.integers(5, 201))

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
        trans_div = float(m["transition_matrix_divergence_category"])
        w1 = float(m["trip_distance_distribution_wasserstein"])
        topk_rec = float(m["topk_poi_recall"])

        if not math.isfinite(w1):
            w1 = 0.0
        if not math.isfinite(topk_rec):
            topk_rec = 0.0

        return (
            1.0 * stop_kl
            + 0.05 * stop_abs_mean_err
            + 1.0 * tod_jsd
            + 1.0 * trans_div
            + 0.1 * w1
            - 0.5 * topk_rec
        )

    def fit(
        self,
        train_ground_truth_by_agent: Mapping[str, Sequence[DayTrajectory]],
        train_dates_by_agent: Mapping[str, Sequence[dt.date]],
        train_days_by_agent_for_transition: Mapping[str, Sequence[DayTrajectory]],
    ) -> CalibrationResult:
        rng = np.random.default_rng(self.base_seed + 1000)
        best_params: Optional[SimulatorParams] = None
        best_obj = float("inf")
        history: List[Dict[str, Any]] = []

        calib_seeds = [self.base_seed + 2000 + i for i in range(int(self.calib_seeds_per_eval))]

        for i in range(self.iters):
            params = self._sample_params(rng)
            sim_rollouts = self.simulator.rollout(
                params=params,
                dates_by_agent=train_dates_by_agent,
                train_days_by_agent_for_transition=train_days_by_agent_for_transition,
                seeds=calib_seeds,
            )
            objs = []
            for _seed, sim_by_agent in sim_rollouts.items():
                m = self.evaluator.compute_metrics(
                    ground_truth_by_agent=train_ground_truth_by_agent,
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
            history=history,
            dwell_time_params_by_coarse_category=dwell_params_by_cat,
        )


# -----------------------------
# Results saving
# -----------------------------

def save_results(output_path: str, calibrated_parameters: Dict[str, Any], evaluation_results_on_validation: Dict[str, Any]) -> None:
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
    p = argparse.ArgumentParser(description="Daily mobility trajectory simulator with calibration and evaluation.")
    p.add_argument("--output", type=str, required=True, help="Output JSON file path for results.")
    p.add_argument("--seed", type=int, default=123, help="Global random seed.")
    p.add_argument("--calib-iters", type=int, default=25, help="Random-search calibration iterations.")
    p.add_argument("--calib-seeds", type=int, default=1, help="Number of RNG seeds per parameter evaluation during calibration.")
    p.add_argument("--eval-seeds", type=int, default=5, help="Number of stochastic rollouts for validation evaluation.")
    p.add_argument("--topk", type=int, default=10, help="K for topK POI recall metric.")
    return p.parse_args(argv)


def _dates_by_agent(days_by_agent: Mapping[str, Sequence[DayTrajectory]]) -> Dict[str, List[dt.date]]:
    return {aid: [d.date for d in days] for aid, days in days_by_agent.items()}


def _as_days_by_agent(days_by_agent: Mapping[str, Sequence[DayTrajectory]]) -> Dict[str, List[DayTrajectory]]:
    return {aid: list(days) for aid, days in days_by_agent.items()}


def _compute_seed_summary(metrics_by_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
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
    args = parse_cli(argv)
    set_global_seed(int(args.seed))

    raw = load_data_from_env()

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

    train_dates = _dates_by_agent(train_by_agent)
    calibrator = RandomSearchCalibrator(
        simulator=simulator,
        evaluator=evaluator,
        iters=int(args.calib_iters),
        base_seed=int(args.seed),
        calib_seeds_per_eval=int(args.calib_seeds),
        topk_recall_k=int(args.topk),
    )
    calib_result = calibrator.fit(
        train_ground_truth_by_agent=train_by_agent,
        train_dates_by_agent=train_dates,
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

    val_dates = _dates_by_agent(val_by_agent)
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

    calibrated_parameters = {
        "best_params": best_params_dict,
        "best_objective_on_training": float(calib_result["best_objective"]),
        "calibration_history": calib_result["history"],
    }
    save_results(args.output, calibrated_parameters, evaluation_results_on_validation)


main()