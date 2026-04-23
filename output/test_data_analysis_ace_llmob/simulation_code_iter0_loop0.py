import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Environment / Paths
# -----------------------------
def _resolve_data_dir() -> str:
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.environ.get("DATA_PATH")

    # Backward compatible with original behavior
    if project_root and data_path:
        return os.path.join(project_root, data_path)

    # Reasonable fallback: assume repo layout (useful for local runs/tests)
    here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
    fallback = os.path.join(here, "data_fitting", "llmob_data")
    if os.path.isdir(fallback):
        return fallback

    raise EnvironmentError(
        "Both environment variables PROJECT_ROOT and DATA_PATH must be set, "
        "or place data under ./data_fitting/llmob_data.\n"
        "Example: PROJECT_ROOT=/abs/path/to/project, DATA_PATH=data_fitting/llmob_data"
    )


DATA_DIR = _resolve_data_dir()


# -----------------------------
# Utilities
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            f"Check that PROJECT_ROOT and DATA_PATH are correct and the file exists."
        )


def safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(float(x), eps))


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return logits
    if temperature <= 0:
        raise ValueError(f"softmax_temperature must be > 0. Got: {temperature}")
    z = logits / float(temperature)
    z = z - np.max(z)
    exp_z = np.exp(z)
    s = float(exp_z.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(logits, dtype=float) / float(len(logits))
    return exp_z / s


def sample_from_probs(items: Sequence[Any], probs: np.ndarray, rng: np.random.Generator) -> Any:
    if len(items) != len(probs):
        raise ValueError("items and probs must have the same length.")
    if len(items) == 0:
        raise ValueError("Cannot sample from empty items.")
    p = np.array(probs, dtype=float)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        p = np.ones_like(p, dtype=float) / float(len(p))
    else:
        p = p / s
    idx = int(rng.choice(len(items), p=p))
    return items[idx]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / float(p.sum())
    q = q / float(q.sum())
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size == 0 and q.size == 0:
        return 0.0
    p = p + eps
    q = q + eps
    p = p / float(p.sum())
    q = q / float(q.sum())
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def wasserstein_1d(u: Sequence[float], v: Sequence[float]) -> float:
    u = np.asarray(list(u), dtype=float)
    v = np.asarray(list(v), dtype=float)
    u = u[np.isfinite(u)]
    v = v[np.isfinite(v)]
    if len(u) == 0 or len(v) == 0:
        return float("nan")
    u.sort()
    v.sort()
    n = max(len(u), len(v))
    qs = (np.arange(n) + 0.5) / n
    uq = np.quantile(u, qs, method="linear")
    vq = np.quantile(v, qs, method="linear")
    return float(np.mean(np.abs(uq - vq)))


def day_type_from_date(d: date) -> str:
    return "weekend" if d.weekday() >= 5 else "weekday"


_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s*$")


def minute_of_day_from_hms(hms: str) -> int:
    """
    Convert H:MM:SS or HH:MM:SS to minute-of-day (0..1439), rounding seconds to nearest minute.
    """
    m = _TIME_RE.match(hms)
    if not m:
        raise ValueError(f"Invalid time token: {hms!r}")
    hh, mm, ss = map(int, m.groups())
    if hh < 0 or hh > 23 or mm < 0 or mm > 59 or ss < 0 or ss > 59:
        raise ValueError(f"Out-of-range time: {hms!r}")
    minute = hh * 60 + mm + (1 if ss >= 30 else 0)
    return int(min(1439, max(0, minute)))


def hms_from_minute(minute_of_day: int, seconds: int = 0) -> str:
    m = int(np.clip(minute_of_day, 0, 1439))
    hh = m // 60
    mm = m % 60
    ss = int(np.clip(seconds, 0, 59))
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def parse_date_from_activity_string(s: str) -> date:
    m = re.search(r"Activities at\s+(\d{4}-\d{2}-\d{2})", s)
    if not m:
        raise ValueError(
            "Failed to parse date from daily_activity_string. Expected 'Activities at YYYY-MM-DD'. "
            f"Got: {s[:120]!r}"
        )
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def split_events_part(s: str) -> str:
    m = re.search(r"Activities at\s+\d{4}-\d{2}-\d{2}\s*[:\-]?\s*(.*)$", s)
    if not m:
        raise ValueError(
            "Failed to parse events from daily_activity_string after the date. "
            "Expected format like 'Activities at YYYY-MM-DD: ...'."
        )
    return m.group(1).strip()


_EVENT_RE = re.compile(
    r"^\s*(?P<poi>.+?)\s+at\s+(?P<time>\d{1,2}:\d{2}:\d{2})\s*[.]*\s*$"
)


def parse_event_token(token: str) -> Tuple[str, str, int]:
    """
    Parse a stop token like:
        'Convenience Store#2420 at 11:30:00'
        'small lodging establishment#793 at 0:00:00.'
    Returns (place_type, poi_id, minute_of_day).
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty event token encountered.")

    # Strip common trailing punctuation so the time parser isn't overly strict.
    token = token.rstrip().rstrip(";").rstrip()

    m = _EVENT_RE.match(token)
    if not m:
        # Fallback: try to locate a time at the end and ignore optional punctuation.
        m2 = re.search(r"(?P<time>\d{1,2}:\d{2}:\d{2})\s*[.]*\s*$", token)
        if not m2:
            raise ValueError(
                f"Failed to parse event token: {token!r}. Expected 'POI#id at H:MM:SS'."
            )
        hms = m2.group("time")
        minute = minute_of_day_from_hms(hms)
        poi_part = token[: m2.start("time")].strip()
        poi_part = re.sub(r"\bat\b\s*$", "", poi_part, flags=re.IGNORECASE).strip()
        poi_id = poi_part
    else:
        poi_id = m.group("poi").strip()
        minute = minute_of_day_from_hms(m.group("time"))

    if "#" not in poi_id:
        place_type = poi_id.strip()
    else:
        place_type = poi_id.split("#", 1)[0].strip()

    if not place_type:
        place_type = "Unknown"
    return place_type, poi_id, minute


def trajectory_to_activity_string(traj: "DayTrajectory") -> Optional[str]:
    if traj.events is None:
        return None
    if len(traj.events) == 0:
        # Dataset typically omits empty days; return None to skip.
        return None
    parts = [f"{e.poi_id} at {hms_from_minute(e.minute_of_day, seconds=0)}" for e in traj.events]
    return f"Activities at {traj.d.isoformat()}: " + ", ".join(parts) + "."


def validate_activity_string(s: str) -> Tuple[date, List[Tuple[str, int]]]:
    """
    Strict-ish validator for the 1921Y format.

    Returns: (date, [(poi_id, minute_of_day), ...]).
    """
    d = parse_date_from_activity_string(s)
    events_part = split_events_part(s).strip()
    if not events_part:
        return d, []

    # remove final trailing period for splitting
    events_part = events_part.strip()
    if events_part.endswith("."):
        events_part = events_part[:-1].strip()

    tokens = [t.strip() for t in events_part.split(",") if t.strip()]
    parsed: List[Tuple[str, int]] = []
    for tok in tokens:
        _pt, poi_id, minute = parse_event_token(tok)
        parsed.append((poi_id, minute))
    # enforce non-decreasing time
    for i in range(1, len(parsed)):
        if parsed[i][1] < parsed[i - 1][1]:
            raise ValueError("Times are not non-decreasing in generated trajectory string.")
    return d, parsed


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Event:
    place_type: str
    poi_id: str
    minute_of_day: int


@dataclass(frozen=True)
class DayTrajectory:
    agent_id: str
    d: date
    events: Tuple[Event, ...]


@dataclass
class POI:
    poi_id: str
    category: str
    coarse_category: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    base_attractiveness: float = 0.0
    current_occupancy: int = 0


@dataclass
class AgentProfile:
    agent_id: str
    home_poi_candidates: Dict[str, float] = field(default_factory=dict)
    anchor_poi_affinity: Dict[str, float] = field(default_factory=dict)
    category_preference: Dict[str, float] = field(default_factory=dict)
    time_of_day_priors_by_category: Dict[str, np.ndarray] = field(default_factory=dict)  # 1440 bins
    mobility_radius_km: float = 2.0
    personal_category_transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    personal_poi_transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    personal_stop_count_by_day_type: Dict[str, Dict[int, int]] = field(default_factory=dict)
    personal_first_time_by_day_type: Dict[str, np.ndarray] = field(default_factory=dict)  # 1440 bins


@dataclass
class GlobalModel:
    coarse_categories: List[str]
    poi_ids: List[str]
    poi_by_id: Dict[str, POI]
    pois_by_coarse_category: Dict[str, List[str]]

    global_poi_popularity: Dict[str, float] = field(default_factory=dict)
    global_category_transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    global_poi_transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    stop_count_by_day_type: Dict[str, Dict[int, int]] = field(default_factory=dict)
    first_time_by_day_type: Dict[str, np.ndarray] = field(default_factory=dict)
    dwell_lognormal_by_coarse_category: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    mobility_transition_graph: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)


@dataclass
class CalibratedParameters:
    w_personal_transition: float = 0.7
    w_distance_decay: float = 1.5
    softmax_temperature: float = 1.0
    travel_time_speed_kmph: float = 25.0
    day_end_hazard_scale: float = 2.0
    day_end_hazard_shift_minute: int = 21 * 60
    candidate_set_topk_from_transition_graph: int = 50
    dwell_mu_shift: float = 0.0
    dwell_sigma_mult: float = 1.0
    w_personal_poi_affinity: float = 1.0
    w_global_poi_popularity: float = 0.7


# -----------------------------
# Data loading
# -----------------------------
def load_json(path: str) -> Any:
    require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data() -> Dict[str, Any]:
    y_file = os.path.join(DATA_DIR, "1921Y.json")
    poi_file = os.path.join(DATA_DIR, "poi_category_192021_longitude_latitude.json")
    catto_file = os.path.join(DATA_DIR, "catto.json")
    return {
        "y": load_json(y_file),
        "poi_catalog": load_json(poi_file),
        "catto": load_json(catto_file),
        "paths": {"1921Y.json": y_file, "poi_catalog": poi_file, "catto": catto_file},
    }


def _extract_place_type_to_coarse(catto_obj: Any) -> Dict[str, str]:
    # catto.json is usually a flat dict {place_type: coarse_category}
    if isinstance(catto_obj, dict) and "place_type_to_coarse_category" in catto_obj:
        mapping = catto_obj["place_type_to_coarse_category"]
    else:
        mapping = catto_obj
    if not isinstance(mapping, dict):
        raise ValueError("catto.json must be a dict mapping place_type to coarse category.")
    out: Dict[str, str] = {}
    for k, v in mapping.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def parse_trajectories(y_obj: Any) -> List[DayTrajectory]:
    if not isinstance(y_obj, dict):
        raise ValueError("1921Y.json must be a JSON object mapping person_id to records.")
    trajs: List[DayTrajectory] = []
    for agent_id, record in y_obj.items():
        if not isinstance(agent_id, str) or not agent_id:
            continue
        if isinstance(record, str):
            strings = [record]
        elif isinstance(record, list) and all(isinstance(x, str) for x in record):
            strings = list(record)
        elif isinstance(record, dict) and "daily_activity_string" in record:
            das = record["daily_activity_string"]
            if isinstance(das, str):
                strings = [das]
            elif isinstance(das, list) and all(isinstance(x, str) for x in das):
                strings = list(das)
            else:
                raise ValueError(f"Unsupported daily_activity_string for agent_id {agent_id!r}.")
        else:
            raise ValueError(f"Unsupported record type for agent_id {agent_id!r}: {type(record)}")

        for s in strings:
            d = parse_date_from_activity_string(s)
            events_part = split_events_part(s)
            if not events_part:
                continue
            tokens = [t.strip() for t in events_part.split(",") if t.strip()]
            events: List[Event] = []
            for token in tokens:
                place_type, poi_id, minute = parse_event_token(token)
                events.append(Event(place_type=place_type, poi_id=poi_id, minute_of_day=minute))
            events.sort(key=lambda e: e.minute_of_day)
            trajs.append(DayTrajectory(agent_id=agent_id, d=d, events=tuple(events)))

    if not trajs:
        raise ValueError("No trajectories were parsed from 1921Y.json. Check the input format.")
    return trajs


def build_poi_catalog(
    poi_catalog_obj: Any,
    place_type_to_coarse: Dict[str, str],
) -> Tuple[Dict[str, POI], Dict[str, List[str]], List[str]]:
    """
    Supports two schemas:
      A) {category: [[lat, lon, poi_id], ...], ...}   (observed in this environment)
      B) {poi_id: [lat, lon], ...} or {poi_id: {"lat": .., "lon": ..}, ...}
    """
    poi_by_id: Dict[str, POI] = {}
    pois_by_coarse: Dict[str, List[str]] = {}

    if isinstance(poi_catalog_obj, dict):
        # Detect schema B quickly: if many keys contain '#' and values look like coordinates
        sample_items = list(poi_catalog_obj.items())[:50]
        b_like = 0
        for k, v in sample_items:
            if isinstance(k, str) and "#" in k:
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    b_like += 1
                elif isinstance(v, dict) and ("lat" in v and "lon" in v):
                    b_like += 1

        if b_like >= max(3, len(sample_items) // 3):
            # Schema B
            for poi_id_raw, v in poi_catalog_obj.items():
                if not isinstance(poi_id_raw, str) or not poi_id_raw.strip():
                    continue
                poi_id = poi_id_raw.strip()
                category = poi_id.split("#", 1)[0].strip() if "#" in poi_id else poi_id
                coarse = place_type_to_coarse.get(category, category)

                lat = None
                lon = None
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        lat = float(v[0])
                        lon = float(v[1])
                    except Exception:
                        lat, lon = None, None
                elif isinstance(v, dict):
                    try:
                        lat = float(v.get("lat"))
                        lon = float(v.get("lon"))
                    except Exception:
                        lat, lon = None, None

                poi_by_id[poi_id] = POI(
                    poi_id=poi_id,
                    category=category,
                    coarse_category=coarse,
                    latitude=lat,
                    longitude=lon,
                )
                pois_by_coarse.setdefault(coarse, []).append(poi_id)
        else:
            # Schema A
            for category_raw, records in poi_catalog_obj.items():
                if not isinstance(category_raw, str) or not category_raw.strip():
                    continue
                category = category_raw.strip()
                if not isinstance(records, list):
                    raise ValueError(f"POI catalog for category {category!r} must be a list.")
                for rec in records:
                    if not (isinstance(rec, list) and len(rec) >= 3):
                        raise ValueError(f"POI record under category {category!r} must be [lat, lon, poi_id]. Got: {rec!r}")
                    lat_raw, lon_raw, poi_id_raw = rec[0], rec[1], rec[2]
                    if not isinstance(poi_id_raw, str) or not poi_id_raw.strip():
                        continue
                    poi_id = poi_id_raw.strip()
                    lat = None
                    lon = None
                    try:
                        lat = float(lat_raw)
                        lon = float(lon_raw)
                    except Exception:
                        lat, lon = None, None

                    # Use category key as the place_type
                    coarse = place_type_to_coarse.get(category, category)
                    poi_by_id[poi_id] = POI(
                        poi_id=poi_id,
                        category=category,
                        coarse_category=coarse,
                        latitude=lat,
                        longitude=lon,
                    )
                    pois_by_coarse.setdefault(coarse, []).append(poi_id)
    else:
        raise ValueError("poi_category_192021_longitude_latitude.json must be a JSON object.")

    if not poi_by_id:
        raise ValueError("POI catalog is empty after parsing.")

    coarse_categories = sorted(pois_by_coarse.keys())
    return poi_by_id, pois_by_coarse, coarse_categories


def build_network_and_agents(data: Dict[str, Any]) -> Dict[str, Any]:
    place_type_to_coarse = _extract_place_type_to_coarse(data["catto"])
    trajectories = parse_trajectories(data["y"])
    poi_by_id, pois_by_coarse, coarse_categories = build_poi_catalog(data["poi_catalog"], place_type_to_coarse)
    agent_ids = sorted({t.agent_id for t in trajectories})
    return {
        "trajectories": trajectories,
        "poi_by_id": poi_by_id,
        "pois_by_coarse_category": pois_by_coarse,
        "coarse_categories": coarse_categories,
        "place_type_to_coarse": place_type_to_coarse,
        "agent_ids": agent_ids,
    }


def holdout_split(trajectories: List[DayTrajectory]) -> Dict[str, Dict[str, List[DayTrajectory]]]:
    by_agent: Dict[str, List[DayTrajectory]] = {}
    for t in trajectories:
        by_agent.setdefault(t.agent_id, []).append(t)
    split: Dict[str, Dict[str, List[DayTrajectory]]] = {}
    for agent_id, days in by_agent.items():
        days_sorted = sorted(days, key=lambda x: x.d)
        n = len(days_sorted)
        if n == 1:
            train_days = days_sorted
            val_days: List[DayTrajectory] = []
        else:
            train_n = int(math.floor(0.8 * n))
            train_n = max(1, min(n - 1, train_n))
            train_days = days_sorted[:train_n]
            val_days = days_sorted[train_n:]
        split[agent_id] = {"train": train_days, "validation": val_days}
    return split


def filter_by_year_range(trajs: List[DayTrajectory], year_min: int, year_max: int) -> List[DayTrajectory]:
    return [t for t in trajs if year_min <= t.d.year <= year_max]


def _split_train_validation_maps(
    split: Dict[str, Dict[str, List[DayTrajectory]]],
) -> Tuple[Dict[str, List[DayTrajectory]], Dict[str, List[DayTrajectory]], List[DayTrajectory], List[DayTrajectory]]:
    train: Dict[str, List[DayTrajectory]] = {}
    val: Dict[str, List[DayTrajectory]] = {}
    train_flat: List[DayTrajectory] = []
    val_flat: List[DayTrajectory] = []
    for aid, parts in split.items():
        tdays = parts.get("train", [])
        vdays = parts.get("validation", [])
        train[aid] = tdays
        val[aid] = vdays
        train_flat.extend(tdays)
        val_flat.extend(vdays)
    return train, val, train_flat, val_flat


# -----------------------------
# Model fitting
# -----------------------------
class ModelFitter:
    def __init__(
        self,
        poi_by_id: Dict[str, POI],
        pois_by_coarse_category: Dict[str, List[str]],
        coarse_categories: List[str],
        place_type_to_coarse: Dict[str, str],
    ) -> None:
        self.poi_by_id = poi_by_id
        self.pois_by_coarse_category = pois_by_coarse_category
        self.coarse_categories = coarse_categories
        self.place_type_to_coarse = place_type_to_coarse

    def _coarse_from_place_type_or_poi(self, place_type: str, poi_id: str) -> str:
        if place_type in self.place_type_to_coarse:
            return self.place_type_to_coarse[place_type]
        if "#" in poi_id:
            return poi_id.split("#", 1)[0].strip() or place_type
        return place_type

    def _poi_coords(self, poi_id: str) -> Optional[Tuple[float, float]]:
        p = self.poi_by_id.get(poi_id)
        if p is None or p.latitude is None or p.longitude is None:
            return None
        return (p.latitude, p.longitude)

    def fit(self, train_split: Dict[str, List[DayTrajectory]]) -> Tuple[Dict[str, AgentProfile], GlobalModel]:
        global_poi_visits: Dict[str, int] = {}
        global_cat_trans: Dict[Tuple[str, str], int] = {}
        global_poi_trans: Dict[Tuple[str, str], int] = {}
        stop_counts_by_type: Dict[str, Dict[int, int]] = {"weekday": {}, "weekend": {}}
        first_time_by_type: Dict[str, np.ndarray] = {"weekday": np.zeros(1440), "weekend": np.zeros(1440)}
        dwell_samples_by_cat: Dict[str, List[float]] = {}
        mobility_graph_counts: Dict[str, Dict[str, int]] = {}

        agent_profiles: Dict[str, AgentProfile] = {}

        for agent_id, days in train_split.items():
            prof = AgentProfile(agent_id=agent_id)

            poi_visit_counts: Dict[str, int] = {}
            cat_visit_counts: Dict[str, int] = {}
            tod_by_cat: Dict[str, np.ndarray] = {}
            personal_cat_trans: Dict[Tuple[str, str], int] = {}
            personal_poi_trans: Dict[Tuple[str, str], int] = {}
            personal_stop_counts: Dict[str, Dict[int, int]] = {"weekday": {}, "weekend": {}}
            personal_first_time: Dict[str, np.ndarray] = {"weekday": np.zeros(1440), "weekend": np.zeros(1440)}
            distances: List[float] = []
            home_score: Dict[str, float] = {}

            for day in days:
                dt = day_type_from_date(day.d)
                n_events = len(day.events)
                personal_stop_counts[dt][n_events] = personal_stop_counts[dt].get(n_events, 0) + 1
                stop_counts_by_type[dt][n_events] = stop_counts_by_type[dt].get(n_events, 0) + 1
                if n_events > 0:
                    ft = int(day.events[0].minute_of_day)
                    personal_first_time[dt][ft] += 1.0
                    first_time_by_type[dt][ft] += 1.0

                for e in day.events:
                    poi_visit_counts[e.poi_id] = poi_visit_counts.get(e.poi_id, 0) + 1
                    global_poi_visits[e.poi_id] = global_poi_visits.get(e.poi_id, 0) + 1

                    coarse = self._coarse_from_place_type_or_poi(e.place_type, e.poi_id)
                    cat_visit_counts[coarse] = cat_visit_counts.get(coarse, 0) + 1
                    tod_by_cat.setdefault(coarse, np.zeros(1440))
                    tod_by_cat[coarse][e.minute_of_day] += 1.0

                    if e.minute_of_day <= 5 * 60 or e.minute_of_day >= 22 * 60:
                        home_score[e.poi_id] = home_score.get(e.poi_id, 0.0) + 2.0
                    if e.place_type.lower().strip() == "home" or e.poi_id.lower().startswith("home#"):
                        home_score[e.poi_id] = home_score.get(e.poi_id, 0.0) + 5.0

                evs = list(day.events)
                for i in range(len(evs) - 1):
                    a = evs[i]
                    b = evs[i + 1]
                    ca = self._coarse_from_place_type_or_poi(a.place_type, a.poi_id)
                    cb = self._coarse_from_place_type_or_poi(b.place_type, b.poi_id)
                    personal_cat_trans[(ca, cb)] = personal_cat_trans.get((ca, cb), 0) + 1
                    global_cat_trans[(ca, cb)] = global_cat_trans.get((ca, cb), 0) + 1

                    personal_poi_trans[(a.poi_id, b.poi_id)] = personal_poi_trans.get((a.poi_id, b.poi_id), 0) + 1
                    global_poi_trans[(a.poi_id, b.poi_id)] = global_poi_trans.get((a.poi_id, b.poi_id), 0) + 1

                    mobility_graph_counts.setdefault(a.poi_id, {})
                    mobility_graph_counts[a.poi_id][b.poi_id] = mobility_graph_counts[a.poi_id].get(b.poi_id, 0) + 1

                    caa = self._poi_coords(a.poi_id)
                    cbb = self._poi_coords(b.poi_id)
                    if caa is not None and cbb is not None:
                        distances.append(haversine_km(caa[0], caa[1], cbb[0], cbb[1]))

                    gap = int(b.minute_of_day) - int(a.minute_of_day)
                    if gap > 0:
                        dwell_samples_by_cat.setdefault(ca, []).append(float(max(5, min(gap, 8 * 60))))

            tot_poi = sum(poi_visit_counts.values())
            prof.anchor_poi_affinity = {k: v / tot_poi for k, v in poi_visit_counts.items()} if tot_poi > 0 else {}
            tot_cat = sum(cat_visit_counts.values())
            prof.category_preference = {k: v / tot_cat for k, v in cat_visit_counts.items()} if tot_cat > 0 else {}

            for coarse, hist in tod_by_cat.items():
                s = float(hist.sum())
                prof.time_of_day_priors_by_category[coarse] = (hist / s) if s > 0 else (np.ones(1440) / 1440.0)

            for dt in ("weekday", "weekend"):
                h = personal_first_time[dt]
                s = float(h.sum())
                prof.personal_first_time_by_day_type[dt] = (h / s) if s > 0 else (np.ones(1440) / 1440.0)

            prof.personal_stop_count_by_day_type = personal_stop_counts
            prof.personal_category_transitions = personal_cat_trans
            prof.personal_poi_transitions = personal_poi_trans
            prof.mobility_radius_km = float(np.median(distances)) if distances else 2.0

            if home_score:
                top = sorted(home_score.items(), key=lambda x: x[1], reverse=True)[:5]
                total = sum(max(0.0, s) for _, s in top)
                if total <= 0:
                    prof.home_poi_candidates = {poi: 1.0 / len(top) for poi, _ in top}
                else:
                    prof.home_poi_candidates = {poi: max(0.0, s) / total for poi, s in top}
            elif prof.anchor_poi_affinity:
                top_poi = max(prof.anchor_poi_affinity.items(), key=lambda x: x[1])[0]
                prof.home_poi_candidates = {top_poi: 1.0}
            else:
                prof.home_poi_candidates = {}

            agent_profiles[agent_id] = prof

        pop_scores = {poi_id: math.log1p(c) for poi_id, c in global_poi_visits.items()}
        s = float(sum(pop_scores.values()))
        global_poi_popularity = {k: (v / s if s > 0 else 0.0) for k, v in pop_scores.items()}

        for poi_id, poi in self.poi_by_id.items():
            poi.base_attractiveness = global_poi_popularity.get(poi_id, 0.0)

        global_first_time: Dict[str, np.ndarray] = {}
        for dt in ("weekday", "weekend"):
            h = first_time_by_type[dt]
            tot = float(h.sum())
            global_first_time[dt] = (h / tot) if tot > 0 else (np.ones(1440) / 1440.0)

        dwell_lognormal: Dict[str, Tuple[float, float]] = {}
        for coarse in self.coarse_categories:
            xs = [x for x in dwell_samples_by_cat.get(coarse, []) if x > 0 and np.isfinite(x)]
            if len(xs) >= 10:
                logs = np.log(np.array(xs, dtype=float))
                mu = float(np.mean(logs))
                sigma = float(np.std(logs) + 1e-6)
                dwell_lognormal[coarse] = (mu, sigma)
            else:
                dwell_lognormal[coarse] = (safe_log(60.0), 0.8)

        graph: Dict[str, List[Tuple[str, int]]] = {}
        for src, dst_counts in mobility_graph_counts.items():
            sorted_dsts = sorted(dst_counts.items(), key=lambda x: x[1], reverse=True)
            graph[src] = [(dst, int(cnt)) for dst, cnt in sorted_dsts]

        global_model = GlobalModel(
            coarse_categories=list(self.coarse_categories),
            poi_ids=sorted(self.poi_by_id.keys()),
            poi_by_id=self.poi_by_id,
            pois_by_coarse_category=self.pois_by_coarse_category,
            global_poi_popularity=global_poi_popularity,
            global_category_transitions=global_cat_trans,
            global_poi_transitions=global_poi_trans,
            stop_count_by_day_type=stop_counts_by_type,
            first_time_by_day_type=global_first_time,
            dwell_lognormal_by_coarse_category=dwell_lognormal,
            mobility_transition_graph=graph,
        )
        return agent_profiles, global_model


def ensure_profiles_for_agents(
    agent_profiles: Dict[str, AgentProfile],
    agent_ids: Iterable[str],
    global_model: GlobalModel,
) -> Dict[str, AgentProfile]:
    out = dict(agent_profiles)
    cats = list(global_model.coarse_categories) or ["Unknown"]
    uniform_tod = np.ones(1440) / 1440.0
    for aid in agent_ids:
        if aid in out:
            continue
        prof = AgentProfile(agent_id=aid)
        prof.category_preference = {c: 1.0 / len(cats) for c in cats}
        for c in cats:
            prof.time_of_day_priors_by_category[c] = uniform_tod
        prof.personal_first_time_by_day_type = {"weekday": uniform_tod, "weekend": uniform_tod}
        prof.personal_stop_count_by_day_type = {"weekday": {0: 1}, "weekend": {0: 1}}
        prof.mobility_radius_km = 2.0
        out[aid] = prof
    return out


# -----------------------------
# Simulator (statistical core)
# -----------------------------
class MobilitySimulator:
    def __init__(
        self,
        global_model: GlobalModel,
        agent_profiles: Dict[str, AgentProfile],
        place_type_to_coarse: Dict[str, str],
        params: CalibratedParameters,
        base_seed: int,
    ) -> None:
        self.global_model = global_model
        self.agent_profiles = agent_profiles
        self.place_type_to_coarse = place_type_to_coarse
        self.params = params
        self.base_seed = int(base_seed)

        if not self.global_model.poi_ids:
            raise ValueError("GlobalModel.poi_ids is empty. Cannot simulate without POIs.")

        self._poi_coords_cache: Dict[str, Optional[Tuple[float, float]]] = {}
        self._coarse_by_poi_cache: Dict[str, str] = {}
        self._place_type_by_poi_cache: Dict[str, str] = {}

    def _place_type_from_poi(self, poi_id: str) -> str:
        if poi_id in self._place_type_by_poi_cache:
            return self._place_type_by_poi_cache[poi_id]
        p = self.global_model.poi_by_id.get(poi_id)
        if p is not None and p.category:
            pt = p.category
        else:
            pt = poi_id.split("#", 1)[0].strip() if "#" in poi_id else "Unknown"
        self._place_type_by_poi_cache[poi_id] = pt
        return pt

    def _coarse_from_poi(self, poi_id: str) -> str:
        if poi_id in self._coarse_by_poi_cache:
            return self._coarse_by_poi_cache[poi_id]
        p = self.global_model.poi_by_id.get(poi_id)
        coarse = p.coarse_category if p is not None else None
        if coarse is None:
            place_type = self._place_type_from_poi(poi_id)
            coarse = self.place_type_to_coarse.get(place_type, place_type)
        self._coarse_by_poi_cache[poi_id] = coarse
        return coarse

    def _coords(self, poi_id: str) -> Optional[Tuple[float, float]]:
        if poi_id in self._poi_coords_cache:
            return self._poi_coords_cache[poi_id]
        p = self.global_model.poi_by_id.get(poi_id)
        if p is None or p.latitude is None or p.longitude is None:
            self._poi_coords_cache[poi_id] = None
            return None
        self._poi_coords_cache[poi_id] = (p.latitude, p.longitude)
        return self._poi_coords_cache[poi_id]

    def _distance_km(self, a: str, b: str, fallback_km: float) -> float:
        ca = self._coords(a)
        cb = self._coords(b)
        if ca is None or cb is None:
            return float(fallback_km)
        return haversine_km(ca[0], ca[1], cb[0], cb[1])

    def _travel_time_minutes(self, distance_km: float) -> float:
        speed = max(1e-6, float(self.params.travel_time_speed_kmph))
        return float(distance_km / speed * 60.0)

    def _day_end_probability(self, minute: int) -> float:
        scale = max(0.0, float(self.params.day_end_hazard_scale))
        shift = int(self.params.day_end_hazard_shift_minute)
        x = (minute - shift) / 60.0
        z = scale * x
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def _sample_stop_count(self, prof: AgentProfile, dt: str, rng: np.random.Generator) -> int:
        personal = prof.personal_stop_count_by_day_type.get(dt, {})
        globalc = self.global_model.stop_count_by_day_type.get(dt, {})
        support = sorted(set(personal.keys()) | set(globalc.keys()))
        if not support:
            return int(rng.integers(1, 5))
        p_counts = np.array([personal.get(k, 0) for k in support], dtype=float) + 1.0
        g_counts = np.array([globalc.get(k, 0) for k in support], dtype=float) + 1.0
        p_probs = p_counts / float(p_counts.sum())
        g_probs = g_counts / float(g_counts.sum())
        w = float(self.params.w_personal_transition)
        probs = w * p_probs + (1.0 - w) * g_probs
        return max(0, int(sample_from_probs(support, probs, rng)))

    def _sample_first_time(self, prof: AgentProfile, dt: str, rng: np.random.Generator) -> int:
        p = prof.personal_first_time_by_day_type.get(dt)
        g = self.global_model.first_time_by_day_type.get(dt)
        if p is None or len(p) != 1440:
            p = np.ones(1440) / 1440.0
        if g is None or len(g) != 1440:
            g = np.ones(1440) / 1440.0
        w = float(self.params.w_personal_transition)
        probs = w * p + (1.0 - w) * g
        probs = probs / float(probs.sum())
        return int(rng.choice(1440, p=probs))

    def _category_transition_probs(
        self,
        prof: AgentProfile,
        prev_cat: Optional[str],
        current_minute: int,
    ) -> Tuple[List[str], np.ndarray]:
        cats = self.global_model.coarse_categories
        if not cats:
            cats = ["Unknown"]

        personal_counts = np.zeros(len(cats), dtype=float)
        global_counts = np.zeros(len(cats), dtype=float)

        if prev_cat is None:
            for i, c in enumerate(cats):
                personal_counts[i] = prof.category_preference.get(c, 0.0)
                global_counts[i] = 1.0
        else:
            for i, c in enumerate(cats):
                personal_counts[i] = float(prof.personal_category_transitions.get((prev_cat, c), 0))
                global_counts[i] = float(self.global_model.global_category_transitions.get((prev_cat, c), 0))

        personal_counts = personal_counts + 1.0
        global_counts = global_counts + 1.0
        personal_probs = personal_counts / float(personal_counts.sum())
        global_probs = global_counts / float(global_counts.sum())

        w = float(self.params.w_personal_transition)
        probs = w * personal_probs + (1.0 - w) * global_probs

        tod = np.ones(len(cats), dtype=float)
        for i, c in enumerate(cats):
            hist = prof.time_of_day_priors_by_category.get(c)
            if hist is not None and len(hist) == 1440:
                tod[i] = float(hist[int(current_minute)])
            else:
                tod[i] = 1.0 / 1440.0
        probs = probs * (tod + 1e-9)
        s = float(probs.sum())
        if s <= 0 or not np.isfinite(s):
            probs = np.ones(len(cats), dtype=float) / float(len(cats))
        else:
            probs = probs / s
        return list(cats), probs

    def _poi_candidate_set(self, current_poi: str, target_coarse: str, topk: int) -> List[str]:
        candidates: List[str] = []
        neigh = self.global_model.mobility_transition_graph.get(current_poi, [])
        if neigh:
            for dst, _cnt in neigh[: max(1, topk)]:
                if self._coarse_from_poi(dst) == target_coarse:
                    candidates.append(dst)

        if not candidates:
            candidates = list(self.global_model.pois_by_coarse_category.get(target_coarse, []))

        seen = set()
        uniq: List[str] = []
        for x in candidates:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    def _choose_poi(
        self,
        prof: AgentProfile,
        current_poi: str,
        target_coarse: str,
        rng: np.random.Generator,
        recently_visited: Optional[set[str]] = None,
    ) -> str:
        topk = int(np.clip(int(self.params.candidate_set_topk_from_transition_graph), 5, 200))
        candidates = self._poi_candidate_set(current_poi, target_coarse, topk=topk)
        if not candidates:
            candidates = list(self.global_model.poi_ids)
        if not candidates:
            raise ValueError("No POIs available to sample from.")

        fallback_dist = max(0.1, float(prof.mobility_radius_km))
        logits = np.zeros(len(candidates), dtype=float)
        for i, poi_id in enumerate(candidates):
            aff = prof.anchor_poi_affinity.get(poi_id, 0.0)
            pop = self.global_model.global_poi_popularity.get(poi_id, 0.0)
            dist = self._distance_km(current_poi, poi_id, fallback_km=fallback_dist)

            recency_penalty = 0.0
            if recently_visited is not None and poi_id in recently_visited:
                recency_penalty = 0.5

            logits[i] = (
                self.params.w_personal_poi_affinity * safe_log(aff + 1e-6)
                + self.params.w_global_poi_popularity * safe_log(pop + 1e-6)
                - float(self.params.w_distance_decay) * float(dist)
                - recency_penalty
            )

        probs = softmax(logits, temperature=float(self.params.softmax_temperature))
        return str(sample_from_probs(candidates, probs, rng))

    def _sample_dwell_minutes(self, coarse: str, rng: np.random.Generator) -> int:
        mu, sigma = self.global_model.dwell_lognormal_by_coarse_category.get(coarse, (safe_log(60.0), 0.8))
        mu = float(mu) + float(self.params.dwell_mu_shift)
        sigma = max(0.05, float(sigma) * float(self.params.dwell_sigma_mult))
        x = float(rng.lognormal(mean=mu, sigma=sigma))
        return int(np.clip(round(x), 5, 8 * 60))

    def simulate_day(self, agent_id: str, d: date, rng: np.random.Generator) -> DayTrajectory:
        prof = self.agent_profiles.get(agent_id)
        if prof is None:
            raise KeyError(f"Missing AgentProfile for agent_id={agent_id!r}.")
        dt = day_type_from_date(d)

        if prof.home_poi_candidates:
            homes = list(prof.home_poi_candidates.keys())
            weights = np.array([prof.home_poi_candidates[h] for h in homes], dtype=float)
            weights = weights / float(weights.sum()) if float(weights.sum()) > 0 else np.ones(len(homes)) / float(len(homes))
            current_poi = str(sample_from_probs(homes, weights, rng))
        else:
            current_poi = str(rng.choice(self.global_model.poi_ids))

        n_stops = self._sample_stop_count(prof, dt, rng)
        if n_stops <= 0:
            return DayTrajectory(agent_id=agent_id, d=d, events=tuple())

        current_time = self._sample_first_time(prof, dt, rng)
        prev_cat: Optional[str] = None
        events: List[Event] = []
        recently_visited: set[str] = set()
        fatigue = 1.0

        for _i in range(n_stops):
            end_p = self._day_end_probability(current_time)
            end_p = min(0.95, max(0.0, end_p + 0.2 * (1.0 - fatigue)))
            if rng.random() < end_p:
                break

            cats, cat_probs = self._category_transition_probs(prof, prev_cat, current_time)
            target_coarse = str(sample_from_probs(cats, cat_probs, rng))
            next_poi = self._choose_poi(prof, current_poi, target_coarse, rng, recently_visited=recently_visited)

            fallback_dist = max(0.1, float(prof.mobility_radius_km))
            dist = self._distance_km(current_poi, next_poi, fallback_km=fallback_dist)
            travel = self._travel_time_minutes(dist)
            dwell = self._sample_dwell_minutes(target_coarse, rng)

            arrival = int(np.clip(round(current_time + travel), 0, 1439))
            place_type = self._place_type_from_poi(next_poi)
            events.append(Event(place_type=place_type, poi_id=next_poi, minute_of_day=arrival))

            current_poi = next_poi
            prev_cat = target_coarse
            current_time = int(np.clip(round(arrival + dwell), 0, 1439))
            recently_visited.add(next_poi)

            fatigue = float(np.clip(fatigue - (travel + dwell) / (24.0 * 60.0), 0.0, 1.0))
            if place_type.lower() == "home" or next_poi.lower().startswith("home#"):
                fatigue = float(np.clip(fatigue + 0.2, 0.0, 1.0))
            if current_time >= 1439:
                break

        events.sort(key=lambda e: e.minute_of_day)
        return DayTrajectory(agent_id=agent_id, d=d, events=tuple(events))

    def rollout(self, validation_split: Dict[str, List[DayTrajectory]], n_runs: int = 5) -> Dict[int, List[DayTrajectory]]:
        results: Dict[int, List[DayTrajectory]] = {}
        for run in range(int(n_runs)):
            rng = np.random.default_rng(self.base_seed + 1000 + run)
            sim_days: List[DayTrajectory] = []
            for agent_id, real_days in validation_split.items():
                for day in sorted(real_days, key=lambda x: x.d):
                    sim_days.append(self.simulate_day(agent_id=agent_id, d=day.d, rng=rng))
            results[run] = sim_days
        return results


# -----------------------------
# LLM-based pipeline (Pattern / Persona / Motivation / Trajectory)
# -----------------------------
class LLMClient:
    def generate(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 600) -> str:
        raise NotImplementedError


class DummyLLMClient(LLMClient):
    """
    Offline fallback: produces a placeholder. Intended to keep the pipeline runnable without external calls.
    """
    def generate(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 600) -> str:
        # A deterministic short response to keep the pipeline moving.
        # The actual trajectory generation is handled by StatisticalTrajectoryGenerator below.
        return "N/A"


class OpenAIHTTPClient(LLMClient):
    """
    Minimal OpenAI-compatible client using urllib (no extra deps).
    Requires OPENAI_API_KEY in env.
    """
    def __init__(self, model: str = "gpt-4o-mini", api_base: str = "https://api.openai.com/v1") -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

    def generate(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 600) -> str:
        import urllib.request

        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Follow the user's format constraints exactly."},
                {"role": "user", "content": prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"]


class PatternPersonaMotivationExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def derive_pattern(self, history_strings: List[str]) -> str:
        prompt = (
            "Summarize the resident's habitual mobility Pattern from the following daily activity strings.\n"
            "Return only a concise paragraph.\n\n"
            + "\n".join(f"- {s}" for s in history_strings[-30:])
        )
        return self.llm.generate(prompt, temperature=0.4, max_tokens=250).strip()

    def infer_persona(self, history_strings: List[str]) -> str:
        prompt = (
            "Infer a single Persona label for this resident based on their mobility history.\n"
            "Examples: office worker, student, night-shift worker, retiree, frequent traveler.\n"
            "Return only the label.\n\n"
            + "\n".join(f"- {s}" for s in history_strings[-30:])
        )
        return self.llm.generate(prompt, temperature=0.4, max_tokens=40).strip()

    def summarize_motivation(self, last_7_strings: List[str], target_date: str) -> str:
        prompt = (
            f"Today is {target_date}. Summarize the resident's Motivation for today based on the past 7 days.\n"
            "Return only a concise sentence.\n\n"
            + "\n".join(f"- {s}" for s in last_7_strings[-7:])
        )
        return self.llm.generate(prompt, temperature=0.6, max_tokens=80).strip()


class LLMTrajectoryGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate_trajectory_string(self, pattern: str, persona: str, motivation: str, target_date: str) -> str:
        prompt = (
            "Generate exactly one daily mobility trajectory string in the dataset format.\n"
            "Rules:\n"
            "1) Output must be a SINGLE LINE.\n"
            "2) Must start with: Activities at YYYY-MM-DD:\n"
            "3) Then a comma+space separated list of stops.\n"
            "4) Each stop must be: PlaceType#PlaceID at HH:MM:SS\n"
            "   (HH may be 1-2 digits; MM and SS are 2 digits)\n"
            "5) End the line with a period.\n"
            "6) No explanations, no extra text.\n\n"
            f"Pattern: {pattern}\n"
            f"Persona: {persona}\n"
            f"Motivation: {motivation}\n"
            f"Target date: {target_date}\n"
        )
        out = self.llm.generate(prompt, temperature=0.8, max_tokens=300).strip()
        # Validate and raise if malformed
        _d, _events = validate_activity_string(out)
        return out


class StatisticalTrajectoryGenerator:
    """
    Offline fallback that still follows the Pattern/Persona/Motivation pipeline interface,
    but generates trajectories using the fitted simulator and serializes to 1921Y format.
    """
    def __init__(self, simulator: MobilitySimulator) -> None:
        self.simulator = simulator

    def generate_day_string(self, agent_id: str, d: date, rng: np.random.Generator) -> Optional[str]:
        traj = self.simulator.simulate_day(agent_id=agent_id, d=d, rng=rng)
        return trajectory_to_activity_string(traj)


# -----------------------------
# Evaluation (spec-required metrics)
# -----------------------------
class Evaluator:
    def __init__(self, poi_by_id: Dict[str, POI], place_type_to_coarse: Dict[str, str], time_bin_minutes: int = 10) -> None:
        self.poi_by_id = poi_by_id
        self.place_type_to_coarse = place_type_to_coarse
        self.time_bin_minutes = int(time_bin_minutes)

    def _coarse_from_event(self, e: Event) -> str:
        return self.place_type_to_coarse.get(e.place_type, e.place_type)

    def _coords(self, poi_id: str) -> Optional[Tuple[float, float]]:
        p = self.poi_by_id.get(poi_id)
        if p is None or p.latitude is None or p.longitude is None:
            return None
        return (p.latitude, p.longitude)

    def _step_distances_km(self, days: List[DayTrajectory]) -> List[float]:
        ds: List[float] = []
        for day in days:
            evs = list(day.events)
            for i in range(len(evs) - 1):
                a = self._coords(evs[i].poi_id)
                b = self._coords(evs[i + 1].poi_id)
                if a is None or b is None:
                    continue
                ds.append(haversine_km(a[0], a[1], b[0], b[1]))
        return ds

    def _step_intervals_minutes(self, days: List[DayTrajectory]) -> List[int]:
        gaps: List[int] = []
        for day in days:
            evs = list(day.events)
            for i in range(len(evs) - 1):
                dt = int(evs[i + 1].minute_of_day) - int(evs[i].minute_of_day)
                if dt > 0:
                    gaps.append(dt)
        return gaps

    @staticmethod
    def _hist_1d(values: Sequence[float], bin_edges: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return np.zeros(len(bin_edges) - 1, dtype=float)
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.zeros(len(bin_edges) - 1, dtype=float)
        h, _ = np.histogram(v, bins=bin_edges)
        return h.astype(float)

    def _dard_hist(self, days: List[DayTrajectory], categories: List[str]) -> np.ndarray:
        T = int(math.ceil(1440 / self.time_bin_minutes))
        idx = {c: i for i, c in enumerate(categories)}
        hist = np.zeros((T, len(categories)), dtype=float)
        for day in days:
            for e in day.events:
                tb = int(e.minute_of_day) // self.time_bin_minutes
                tb = max(0, min(T - 1, tb))
                c = self._coarse_from_event(e)
                if c not in idx:
                    continue
                hist[tb, idx[c]] += 1.0
        return hist.reshape(-1)

    def _stvd_hist(
        self,
        days: List[DayTrajectory],
        lat_edges: np.ndarray,
        lon_edges: np.ndarray,
        t_bins: int,
    ) -> np.ndarray:
        # Flatten (t, lat, lon) 3D histogram
        hist = np.zeros((t_bins, len(lat_edges) - 1, len(lon_edges) - 1), dtype=float)
        for day in days:
            for e in day.events:
                c = self._coords(e.poi_id)
                if c is None:
                    continue
                tb = int(e.minute_of_day) // self.time_bin_minutes
                tb = max(0, min(t_bins - 1, tb))
                lat, lon = c
                li = int(np.searchsorted(lat_edges, lat, side="right") - 1)
                lj = int(np.searchsorted(lon_edges, lon, side="right") - 1)
                if 0 <= li < len(lat_edges) - 1 and 0 <= lj < len(lon_edges) - 1:
                    hist[tb, li, lj] += 1.0
        return hist.reshape(-1)

    def compute_metrics(self, simulated_days: List[DayTrajectory], real_days: List[DayTrajectory]) -> Dict[str, Any]:
        # SD JSD
        sim_sd = self._step_distances_km(simulated_days)
        real_sd = self._step_distances_km(real_days)
        max_d = 1.0
        if sim_sd:
            max_d = max(max_d, float(np.nanmax(sim_sd)))
        if real_sd:
            max_d = max(max_d, float(np.nanmax(real_sd)))
        max_d = float(np.clip(max_d, 1.0, 300.0))
        sd_bins = np.linspace(0.0, max_d, 60)
        p_sd = self._hist_1d(sim_sd, sd_bins)
        q_sd = self._hist_1d(real_sd, sd_bins)
        jsd_sd = js_divergence(p_sd, q_sd, eps=1e-9)

        # SI JSD
        sim_si = self._step_intervals_minutes(simulated_days)
        real_si = self._step_intervals_minutes(real_days)
        si_bins = np.arange(0, 1440 + self.time_bin_minutes, self.time_bin_minutes, dtype=float)
        if len(si_bins) < 3:
            si_bins = np.array([0.0, 1.0, 1440.0], dtype=float)
        p_si = self._hist_1d(sim_si, si_bins)
        q_si = self._hist_1d(real_si, si_bins)
        jsd_si = js_divergence(p_si, q_si, eps=1e-9)

        # DARD JSD
        cats = sorted(
            set(self._coarse_from_event(e) for d in simulated_days for e in d.events)
            | set(self._coarse_from_event(e) for d in real_days for e in d.events)
        )
        if not cats:
            cats = ["Unknown"]
        p_dard = self._dard_hist(simulated_days, cats)
        q_dard = self._dard_hist(real_days, cats)
        jsd_dard = js_divergence(p_dard, q_dard, eps=1e-9)

        # STVD JSD (time bin + lat/lon grid)
        coords_all: List[Tuple[float, float]] = []
        for d in simulated_days:
            for e in d.events:
                c = self._coords(e.poi_id)
                if c is not None:
                    coords_all.append(c)
        for d in real_days:
            for e in d.events:
                c = self._coords(e.poi_id)
                if c is not None:
                    coords_all.append(c)

        t_bins = int(math.ceil(1440 / self.time_bin_minutes))
        if coords_all:
            lats = np.array([c[0] for c in coords_all], dtype=float)
            lons = np.array([c[1] for c in coords_all], dtype=float)
            lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
            lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
            # Add small margins
            lat_pad = max(1e-6, 0.01 * (lat_max - lat_min + 1e-6))
            lon_pad = max(1e-6, 0.01 * (lon_max - lon_min + 1e-6))
            lat_edges = np.linspace(lat_min - lat_pad, lat_max + lat_pad, 26)  # 25 bins
            lon_edges = np.linspace(lon_min - lon_pad, lon_max + lon_pad, 26)
            p_stvd = self._stvd_hist(simulated_days, lat_edges, lon_edges, t_bins)
            q_stvd = self._stvd_hist(real_days, lat_edges, lon_edges, t_bins)
            jsd_stvd = js_divergence(p_stvd, q_stvd, eps=1e-9)
        else:
            jsd_stvd = 0.0

        return {
            "jsd_step_distance_sd": float(jsd_sd),
            "jsd_step_interval_si": float(jsd_si),
            "jsd_daily_activity_routine_distribution_dard": float(jsd_dard),
            "jsd_spatiotemporal_visits_distribution_stvd": float(jsd_stvd),
            "n_days_simulated": int(len(simulated_days)),
            "n_days_real": int(len(real_days)),
        }


# -----------------------------
# Calibration
# -----------------------------
class Calibrator:
    def fit(self, *args: Any, **kwargs: Any) -> CalibratedParameters:
        raise NotImplementedError


class RandomSearchCalibrator(Calibrator):
    def __init__(
        self,
        n_iterations: int,
        n_runs_per_eval: int,
        base_seed: int,
        max_train_days_total: int = 2000,
    ) -> None:
        self.n_iterations = int(n_iterations)
        self.n_runs_per_eval = int(n_runs_per_eval)
        self.base_seed = int(base_seed)
        self.max_train_days_total = int(max_train_days_total)

        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be > 0.")
        if self.n_runs_per_eval <= 0:
            raise ValueError("n_runs_per_eval must be > 0.")

    def _sample_params(self, rng: np.random.Generator, base: CalibratedParameters) -> CalibratedParameters:
        p = CalibratedParameters(**vars(base))
        p.w_personal_transition = float(rng.uniform(0.0, 1.0))
        p.w_distance_decay = float(rng.uniform(0.0, 10.0))
        p.softmax_temperature = float(rng.uniform(0.05, 5.0))
        p.travel_time_speed_kmph = float(rng.uniform(5.0, 80.0))
        p.day_end_hazard_scale = float(rng.uniform(0.0, 5.0))
        p.day_end_hazard_shift_minute = int(rng.integers(18 * 60, 24 * 60))
        p.candidate_set_topk_from_transition_graph = int(rng.integers(5, 201))
        p.dwell_mu_shift = float(rng.uniform(-0.5, 0.5))
        p.dwell_sigma_mult = float(rng.uniform(0.7, 1.5))
        return p

    def _loss_from_metrics(self, m: Dict[str, Any]) -> float:
        # Use the required JSD metrics
        sd = float(m.get("jsd_step_distance_sd", 0.0))
        si = float(m.get("jsd_step_interval_si", 0.0))
        dard = float(m.get("jsd_daily_activity_routine_distribution_dard", 0.0))
        stvd = float(m.get("jsd_spatiotemporal_visits_distribution_stvd", 0.0))
        return float(1.0 * sd + 1.0 * si + 1.0 * dard + 1.0 * stvd)

    def fit(
        self,
        global_model: GlobalModel,
        agent_profiles: Dict[str, AgentProfile],
        place_type_to_coarse: Dict[str, str],
        train_split: Dict[str, List[DayTrajectory]],
        evaluator: Evaluator,
        initial_params: Optional[CalibratedParameters] = None,
    ) -> CalibratedParameters:
        base = initial_params or CalibratedParameters()

        all_train_days: List[DayTrajectory] = []
        for _aid, days in train_split.items():
            all_train_days.extend(days)
        all_train_days = sorted(all_train_days, key=lambda x: (x.agent_id, x.d))
        if len(all_train_days) > self.max_train_days_total:
            all_train_days = all_train_days[: self.max_train_days_total]

        capped_by_agent: Dict[str, List[DayTrajectory]] = {}
        for d in all_train_days:
            capped_by_agent.setdefault(d.agent_id, []).append(d)

        rng = np.random.default_rng(self.base_seed + 777)
        best_params = base
        best_loss = float("inf")
        best_metrics: Optional[Dict[str, Any]] = None

        for it in range(self.n_iterations):
            cand = self._sample_params(rng, base=base)
            simulator = MobilitySimulator(
                global_model=global_model,
                agent_profiles=agent_profiles,
                place_type_to_coarse=place_type_to_coarse,
                params=cand,
                base_seed=self.base_seed + 9000 + it * 13,
            )
            sim_runs = simulator.rollout(validation_split=capped_by_agent, n_runs=self.n_runs_per_eval)
            metrics_list: List[Dict[str, Any]] = []
            for sim_days in sim_runs.values():
                metrics_list.append(evaluator.compute_metrics(sim_days, all_train_days))

            keys = [
                "jsd_step_distance_sd",
                "jsd_step_interval_si",
                "jsd_daily_activity_routine_distribution_dard",
                "jsd_spatiotemporal_visits_distribution_stvd",
            ]
            metrics_mean: Dict[str, Any] = {}
            for k in keys:
                vals = [float(m.get(k, float("nan"))) for m in metrics_list]
                vals = [v for v in vals if np.isfinite(v)]
                metrics_mean[k] = float(np.mean(vals)) if vals else float("nan")

            loss = self._loss_from_metrics(metrics_mean)
            if loss < best_loss:
                best_loss = loss
                best_params = cand
                best_metrics = metrics_mean

        if best_metrics is not None:
            sys.stderr.write("Calibration done. Best training metrics: " + json.dumps(best_metrics, ensure_ascii=False) + "\n")
        return best_params


# -----------------------------
# Results persistence
# -----------------------------
def save_results(
    output_dir: str,
    calibrated_params: CalibratedParameters,
    evaluation_results: Dict[str, Any],
    simulated_rollouts: Dict[int, List[DayTrajectory]],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "calibrated_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(vars(calibrated_params), f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    # Structured output
    sim_struct: Dict[str, Any] = {}
    sim_text: Dict[str, Any] = {}
    for run_idx, days in simulated_rollouts.items():
        sim_struct[str(run_idx)] = [
            {
                "agent_id": d.agent_id,
                "date": d.d.isoformat(),
                "events": [{"place_type": e.place_type, "poi_id": e.poi_id, "minute_of_day": e.minute_of_day} for e in d.events],
            }
            for d in days
        ]
        sim_text[str(run_idx)] = [trajectory_to_activity_string(d) for d in days if trajectory_to_activity_string(d) is not None]

    with open(os.path.join(output_dir, "simulated_validation_trajectories_structured.json"), "w", encoding="utf-8") as f:
        json.dump(sim_struct, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "simulated_validation_trajectories_1921Y_format.json"), "w", encoding="utf-8") as f:
        json.dump(sim_text, f, indent=2, ensure_ascii=False)


# -----------------------------
# Experiment runner
# -----------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    train_year_min: int
    train_year_max: int
    val_year_min: int
    val_year_max: int
    mode: str  # "iid" or "ood"


def run_experiment(
    cfg: ExperimentConfig,
    context: Dict[str, Any],
    seed: int,
    calib_iters: int,
    calib_runs_per_eval: int,
    eval_runs: int,
    max_train_days_total: int,
    output_root: str,
    use_llm: bool,
    llm_model: str,
) -> Dict[str, Any]:
    all_trajs: List[DayTrajectory] = context["trajectories"]

    train_trajs = filter_by_year_range(all_trajs, cfg.train_year_min, cfg.train_year_max)
    val_trajs = filter_by_year_range(all_trajs, cfg.val_year_min, cfg.val_year_max)

    if cfg.mode == "iid":
        split = holdout_split(train_trajs)
        train_map, val_map, train_flat, val_flat = _split_train_validation_maps(split)
        # evaluation is on the in-regime validation
        real_eval_days = val_flat
    elif cfg.mode == "ood":
        # train on all train_trajs, evaluate on all val_trajs
        train_map: Dict[str, List[DayTrajectory]] = {}
        for d in train_trajs:
            train_map.setdefault(d.agent_id, []).append(d)
        val_map: Dict[str, List[DayTrajectory]] = {}
        for d in val_trajs:
            if d.agent_id in train_map:
                val_map.setdefault(d.agent_id, []).append(d)
        train_flat = [d for days in train_map.values() for d in days]
        real_eval_days = [d for days in val_map.values() for d in days]
        val_flat = real_eval_days
    else:
        raise ValueError(f"Unknown experiment mode: {cfg.mode}")

    train_flat = sorted(train_flat, key=lambda x: (x.agent_id, x.d))
    val_flat = sorted(val_flat, key=lambda x: (x.agent_id, x.d))

    if not train_flat:
        raise ValueError(f"[{cfg.name}] Training set is empty. Cannot fit model.")
    if not val_flat:
        sys.stderr.write(f"Warning: [{cfg.name}] validation set is empty.\n")

    fitter = ModelFitter(
        poi_by_id=context["poi_by_id"],
        pois_by_coarse_category=context["pois_by_coarse_category"],
        coarse_categories=context["coarse_categories"],
        place_type_to_coarse=context["place_type_to_coarse"],
    )
    agent_profiles, global_model = fitter.fit(train_map)

    # Ensure profiles exist for val agents (esp. OOD cases)
    agent_profiles = ensure_profiles_for_agents(agent_profiles, val_map.keys(), global_model)

    evaluator = Evaluator(
        poi_by_id=context["poi_by_id"],
        place_type_to_coarse=context["place_type_to_coarse"],
        time_bin_minutes=10,
    )

    calibrator = RandomSearchCalibrator(
        n_iterations=calib_iters,
        n_runs_per_eval=calib_runs_per_eval,
        base_seed=seed,
        max_train_days_total=max_train_days_total,
    )
    calibrated_params = calibrator.fit(
        global_model=global_model,
        agent_profiles=agent_profiles,
        place_type_to_coarse=context["place_type_to_coarse"],
        train_split=train_map,
        evaluator=evaluator,
        initial_params=CalibratedParameters(),
    )

    simulator = MobilitySimulator(
        global_model=global_model,
        agent_profiles=agent_profiles,
        place_type_to_coarse=context["place_type_to_coarse"],
        params=calibrated_params,
        base_seed=seed,
    )

    simulated_rollouts = simulator.rollout(validation_split=val_map, n_runs=eval_runs)

    per_run_metrics: Dict[str, Dict[str, Any]] = {}
    for run_idx, sim_days in simulated_rollouts.items():
        per_run_metrics[str(run_idx)] = evaluator.compute_metrics(sim_days, real_eval_days)

    keys = [
        "jsd_step_distance_sd",
        "jsd_step_interval_si",
        "jsd_daily_activity_routine_distribution_dard",
        "jsd_spatiotemporal_visits_distribution_stvd",
    ]
    summary: Dict[str, Any] = {"per_run": per_run_metrics, "summary": {}, "data_info": {}}
    for k in keys:
        vals = [float(per_run_metrics[str(i)].get(k, float("nan"))) for i in simulated_rollouts.keys()]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            summary["summary"][k] = {"mean": float("nan"), "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
        else:
            arr = np.array(vals, dtype=float)
            summary["summary"][k] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p05": float(np.quantile(arr, 0.05)),
                "p95": float(np.quantile(arr, 0.95)),
            }

    summary["data_info"] = {
        "experiment": cfg.name,
        "mode": cfg.mode,
        "train_year_range": [cfg.train_year_min, cfg.train_year_max],
        "val_year_range": [cfg.val_year_min, cfg.val_year_max],
        "n_agents_total": len(context["agent_ids"]),
        "n_train_days": len(train_flat),
        "n_eval_days": len(real_eval_days),
        "data_dir": DATA_DIR,
        "input_files": context.get("paths") if "paths" in context else None,
        "use_llm": bool(use_llm),
        "llm_model": llm_model,
    }

    out_dir = os.path.join(output_root, cfg.name)
    save_results(out_dir, calibrated_params, summary, simulated_rollouts)

    # Optional: demonstrate LLM pipeline generation for the first run (if enabled)
    if use_llm:
        llm: LLMClient
        try:
            llm = OpenAIHTTPClient(model=llm_model)
        except Exception as e:
            sys.stderr.write(f"LLM disabled (fallback): {e}\n")
            llm = DummyLLMClient()

        extractor = PatternPersonaMotivationExtractor(llm)
        llm_gen = LLMTrajectoryGenerator(llm)
        stat_gen = StatisticalTrajectoryGenerator(simulator)

        # Collect historical strings for each agent from training (for prompts)
        hist_strings_by_agent: Dict[str, List[str]] = {}
        for aid, days in train_map.items():
            hist_strings_by_agent[aid] = [trajectory_to_activity_string(d) for d in days if trajectory_to_activity_string(d)]

        rng = np.random.default_rng(seed + 4242)
        llm_out: Dict[str, List[str]] = {}
        for aid, days in list(val_map.items())[:10]:
            hs = hist_strings_by_agent.get(aid, [])
            if not hs:
                continue
            pattern = extractor.derive_pattern(hs)
            persona = extractor.infer_persona(hs)
            for day in sorted(days, key=lambda x: x.d)[:3]:
                last7 = hs[-7:]
                motivation = extractor.summarize_motivation(last7, day.d.isoformat())
                try:
                    s = llm_gen.generate_trajectory_string(pattern, persona, motivation, day.d.isoformat())
                except Exception:
                    # Fallback to statistical serialization if the LLM output fails validation
                    s = stat_gen.generate_day_string(aid, day.d, rng) or ""
                if s:
                    llm_out.setdefault(aid, []).append(s)

        with open(os.path.join(out_dir, "llm_generated_samples.json"), "w", encoding="utf-8") as f:
            json.dump(llm_out, f, indent=2, ensure_ascii=False)

    return summary


# -----------------------------
# CLI
# -----------------------------
def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily mobility trajectory simulator with calibration/evaluation (and LLM pipeline).")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output-dir", type=str, default=os.path.join(DATA_DIR, "outputs"))
    p.add_argument("--calib-iters", type=int, default=30)
    p.add_argument("--calib-runs-per-eval", type=int, default=3)
    p.add_argument("--eval-runs", type=int, default=10)
    p.add_argument("--max-train-days-total", type=int, default=2000)
    p.add_argument("--use-llm", action="store_true", help="Enable LLM-based Pattern/Persona/Motivation/Trajectory generation (requires OPENAI_API_KEY).")
    p.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_cli(argv)
    set_global_seed(args.seed)

    data = load_data()
    context = build_network_and_agents(data)
    context["paths"] = data.get("paths", {})

    experiments = [
        ExperimentConfig(name="2019_to_2019", train_year_min=2019, train_year_max=2020, val_year_min=2019, val_year_max=2020, mode="iid"),
        ExperimentConfig(name="2021_to_2021", train_year_min=2021, train_year_max=2021, val_year_min=2021, val_year_max=2021, mode="iid"),
        ExperimentConfig(name="2019_to_2021_ood", train_year_min=2019, train_year_max=2020, val_year_min=2021, val_year_max=2021, mode="ood"),
    ]

    all_summaries: Dict[str, Any] = {}
    for cfg in experiments:
        summary = run_experiment(
            cfg=cfg,
            context=context,
            seed=args.seed,
            calib_iters=args.calib_iters,
            calib_runs_per_eval=args.calib_runs_per_eval,
            eval_runs=args.eval_runs,
            max_train_days_total=args.max_train_days_total,
            output_root=args.output_dir,
            use_llm=bool(args.use_llm),
            llm_model=str(args.llm_model),
        )
        all_summaries[cfg.name] = summary["summary"]

    sys.stdout.write(json.dumps(all_summaries, indent=2) + "\n")
    sys.stdout.write(f"Saved outputs under: {args.output_dir}\n")



# Execute main for both direct execution and sandbox wrapper invocation
main()