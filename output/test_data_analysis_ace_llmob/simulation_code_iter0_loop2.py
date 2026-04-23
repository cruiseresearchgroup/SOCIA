import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, TypedDict

import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required data file not found: {path}")


def safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(float(x), eps))


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"softmax_temperature must be > 0. Got: {temperature}")
    z = logits / float(temperature)
    z = z - np.max(z)
    exp_z = np.exp(z)
    s = float(exp_z.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(logits, dtype=float) / max(1, len(logits))
    return exp_z / s


def sample_from_probs(items: Sequence[Any], probs: np.ndarray, rng: np.random.Generator) -> Any:
    if len(items) != len(probs):
        raise ValueError("items and probs must have the same length.")
    if len(items) == 0:
        raise ValueError("Cannot sample from empty items.")
    p = np.asarray(probs, dtype=float)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        p = np.ones_like(p, dtype=float) / len(p)
    else:
        p = p / s
    idx = rng.choice(len(items), p=p)
    return items[int(idx)]


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
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def day_type_from_date(d: date) -> str:
    return "weekend" if d.weekday() >= 5 else "weekday"


_TIME_HMS_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s*$")


def minute_of_day_from_hms(hms: str) -> int:
    """
    Convert H:MM:SS or HH:MM:SS to minute-of-day (0..1439), rounding seconds.
    Dataset contains single-digit hours (e.g. '0:00:00').
    """
    m = _TIME_HMS_RE.match(hms)
    if not m:
        raise ValueError(f"Invalid time token: {hms}")
    hh, mm, ss = map(int, m.groups())
    if hh < 0 or hh > 23 or mm < 0 or mm > 59 or ss < 0 or ss > 59:
        raise ValueError(f"Out-of-range time: {hms}")
    minute = hh * 60 + mm + (1 if ss >= 30 else 0)
    return int(min(1439, max(0, minute)))


_DATE_RE = re.compile(r"Activities at\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


def parse_date_from_activity_string(s: str) -> date:
    m = _DATE_RE.search(s)
    if not m:
        raise ValueError(
            "Failed to parse date from daily activity string. Expected substring like 'Activities at YYYY-MM-DD'."
        )
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def split_events_part(s: str) -> str:
    # Accept ":" or "-" or nothing after date.
    m = re.search(r"Activities at\s+\d{4}-\d{2}-\d{2}\s*[:\-]?\s*(.*)$", s, flags=re.IGNORECASE)
    if not m:
        raise ValueError(
            "Failed to parse events from daily activity string after the date. Expected format like "
            "'Activities at YYYY-MM-DD: ...'."
        )
    return m.group(1).strip()


_TRAILING_PUNCT_RE = re.compile(r"[\s\.\,\;\:]+$")


def _strip_trailing_punct(s: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", s.strip())


_EVENT_RE = re.compile(
    r"""
    ^\s*
    (?P<poi>.+?)                 # poi token (possibly including spaces and '#')
    \s+(?:at)\s+                 # ' at '
    (?P<time>\d{1,2}:\d{2}:\d{2})# time
    \s*[\.\,;:]*\s*              # optional trailing punctuation
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_event_token(token: str) -> Tuple[str, str, int]:
    """
    Parse an event token into (place_type, poi_id, minute_of_day).

    Supports:
      - 'POI#id at HH:MM:SS'
      - 'POI#id at H:MM:SS'
      - Optional trailing punctuation: '.', ',', ';'
    """
    token = token.strip()
    token = _strip_trailing_punct(token)
    if not token:
        raise ValueError("Empty event token encountered.")

    m = _EVENT_RE.match(token)
    if m:
        poi_part = m.group("poi").strip()
        hms = m.group("time")
        minute = minute_of_day_from_hms(hms)
    else:
        # Fallback: attempt to find time at end (tolerate punctuation)
        tm = re.search(r"(\d{1,2}:\d{2}:\d{2})\s*[\.\,;:]*\s*$", token)
        if not tm:
            raise ValueError(
                f"Failed to parse time from event token: '{token}'. Expected time at end like 'HH:MM:SS'."
            )
        hms = tm.group(1)
        minute = minute_of_day_from_hms(hms)
        poi_part = token[: tm.start(1)].strip()
        poi_part = re.sub(r"\bat\b\s*$", "", poi_part, flags=re.IGNORECASE).strip()

    poi_part = _strip_trailing_punct(poi_part)
    if not poi_part:
        raise ValueError(f"Missing POI in event token: '{token}'")

    # place_type is token prefix before last '#', if present, else whole
    if "#" in poi_part:
        idx = poi_part.rfind("#")
        place_type = poi_part[:idx].strip()
        poi_id = poi_part.strip()
        if not place_type:
            place_type = poi_id.split("#")[0].strip() or poi_id
    else:
        place_type = poi_part
        poi_id = poi_part

    return place_type, poi_id, minute


def _split_event_tokens(events_part: str) -> List[str]:
    """
    More robust splitting:
      - allow ',' or ';' delimiters
      - tolerate extra whitespace
    """
    if not events_part:
        return []
    # Remove trailing punctuation that applies to the whole line (common: trailing '.')
    s = events_part.strip()
    # Split on commas/semicolons, but keep tokens with internal commas very unlikely in this dataset.
    parts = re.split(r"\s*[,;]\s*", s)
    out = []
    for p in parts:
        p = p.strip()
        p = _strip_trailing_punct(p)
        if p:
            out.append(p)
    return out


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
    first_time_by_day_type: Dict[str, np.ndarray] = field(default_factory=dict)  # 1440 bins
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
# Data loading and preprocessing
# -----------------------------
def load_json(path: str) -> Any:
    require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_poi_catalog_path(data_dir: str, explicit: Optional[str] = None) -> str:
    if explicit:
        p = explicit if os.path.isabs(explicit) else os.path.join(data_dir, explicit)
        require_file(p)
        return p

    candidates = [
        "poi_category_192021_longitude_latitude_complement_alignment_clean.json",
        "poi_category_192021_longitude_latitude.json",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"Could not find a POI catalog in {data_dir}. Tried: {', '.join(candidates)}. "
        "Pass --poi-file to specify."
    )


class LoadedData(TypedDict):
    y: Any
    poi_catalog: Any
    catto: Any
    paths: Dict[str, str]


def load_data(data_dir: str, y_file: str, poi_file: Optional[str], catto_file: str) -> LoadedData:
    y_path = y_file if os.path.isabs(y_file) else os.path.join(data_dir, y_file)
    catto_path = catto_file if os.path.isabs(catto_file) else os.path.join(data_dir, catto_file)
    poi_path = detect_poi_catalog_path(data_dir, explicit=poi_file)

    return {
        "y": load_json(y_path),
        "poi_catalog": load_json(poi_path),
        "catto": load_json(catto_path),
        "paths": {"1921Y.json": y_path, "poi_catalog": poi_path, "catto": catto_path},
    }


def _extract_place_type_to_coarse(catto_obj: Any) -> Dict[str, str]:
    if isinstance(catto_obj, dict) and "place_type_to_coarse_category" in catto_obj:
        mapping = catto_obj["place_type_to_coarse_category"]
    else:
        mapping = catto_obj
    if not isinstance(mapping, dict):
        raise ValueError(
            "catto.json must contain a dict mapping place_type to coarse_category "
            "or a top-level key 'place_type_to_coarse_category'."
        )
    out: Dict[str, str] = {}
    for k, v in mapping.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def parse_trajectories(y_obj: Any) -> List[DayTrajectory]:
    if not isinstance(y_obj, dict):
        raise ValueError("1921Y.json must be a JSON object (dict) mapping person_id to records.")
    trajs: List[DayTrajectory] = []
    for agent_id, record in y_obj.items():
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Invalid agent_id key in 1921Y.json; expected non-empty string keys.")

        strings: List[str]
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
                raise ValueError(f"Unsupported daily_activity_string type for agent '{agent_id}'.")
        else:
            raise ValueError(
                f"Unsupported record type for agent_id '{agent_id}': {type(record)}. "
                "Expected string/list or dict with 'daily_activity_string'."
            )

        for s in strings:
            d = parse_date_from_activity_string(s)
            events_part = split_events_part(s)
            tokens = _split_event_tokens(events_part)
            if not tokens:
                trajs.append(DayTrajectory(agent_id=agent_id, d=d, events=tuple()))
                continue

            events: List[Event] = []
            for token in tokens:
                try:
                    place_type, poi_id, minute = parse_event_token(token)
                except ValueError:
                    # Try again after more aggressive cleanup (some lines include weird trailing chars)
                    token2 = re.sub(r"\s+", " ", token).strip()
                    token2 = _strip_trailing_punct(token2)
                    place_type, poi_id, minute = parse_event_token(token2)
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
    if not isinstance(poi_catalog_obj, dict):
        raise ValueError("POI catalog must be a JSON object mapping category to list of POI records.")
    poi_by_id: Dict[str, POI] = {}
    pois_by_coarse: Dict[str, List[str]] = {}

    for category, records in poi_catalog_obj.items():
        if not isinstance(category, str) or not category.strip():
            continue
        if not isinstance(records, list):
            raise ValueError(f"POI catalog for category '{category}' must be a list.")
        cat = category.strip()
        coarse = place_type_to_coarse.get(cat, cat)
        for rec in records:
            if not (isinstance(rec, list) and len(rec) >= 3):
                raise ValueError(f"POI record under '{category}' must be [lat, lon, poi_id]. Got: {rec}")
            lat_raw, lon_raw, poi_id_raw = rec[0], rec[1], rec[2]
            if not isinstance(poi_id_raw, str) or not poi_id_raw.strip():
                raise ValueError(f"Invalid poi_id in record: {rec}")
            poi_id = poi_id_raw.strip()

            latitude = None
            longitude = None
            try:
                latitude = float(lat_raw)
                longitude = float(lon_raw)
            except Exception:
                latitude = None
                longitude = None

            poi_by_id[poi_id] = POI(
                poi_id=poi_id,
                category=cat,
                coarse_category=coarse,
                latitude=latitude,
                longitude=longitude,
            )
            pois_by_coarse.setdefault(coarse, []).append(poi_id)

    if not poi_by_id:
        raise ValueError("POI catalog is empty after parsing.")
    coarse_categories = sorted(pois_by_coarse.keys())
    return poi_by_id, pois_by_coarse, coarse_categories


class BuildContext(TypedDict):
    trajectories: List[DayTrajectory]
    poi_by_id: Dict[str, POI]
    pois_by_coarse_category: Dict[str, List[str]]
    coarse_categories: List[str]
    place_type_to_coarse: Dict[str, str]
    agent_ids: List[str]


def build_network_and_agents(data: LoadedData) -> BuildContext:
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


# -----------------------------
# Model fitting (data-derived priors)
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
            return poi_id.split("#")[0].strip() or place_type
        return place_type

    def _poi_coords(self, poi_id: str) -> Optional[Tuple[float, float]]:
        p = self.poi_by_id.get(poi_id)
        if p is None or p.latitude is None or p.longitude is None:
            return None
        return (p.latitude, p.longitude)

    def fit(self, train_split: Dict[str, List[DayTrajectory]]) -> Tuple[Dict[str, AgentProfile], GlobalModel]:
        global_poi_visits: Dict[str, int] = {}
        global_cat_trans: Dict[Tuple[str, str], int] = {}
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
                    tod_by_cat.setdefault(coarse, np.zeros(1440, dtype=float))[e.minute_of_day] += 1.0

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
            prof.mobility_radius_km = float(np.median(distances)) if distances else 2.0

            if home_score:
                top = sorted(home_score.items(), key=lambda x: x[1], reverse=True)[:5]
                total = sum(max(0.0, s) for _, s in top)
                prof.home_poi_candidates = (
                    {poi: max(0.0, s) / total for poi, s in top} if total > 0 else {poi: 1.0 / len(top) for poi, _ in top}
                )
            elif prof.anchor_poi_affinity:
                top_poi = max(prof.anchor_poi_affinity.items(), key=lambda x: x[1])[0]
                prof.home_poi_candidates = {top_poi: 1.0}
            else:
                prof.home_poi_candidates = {}

            agent_profiles[agent_id] = prof

        pop_scores = {poi_id: math.log1p(c) for poi_id, c in global_poi_visits.items()}
        s = sum(pop_scores.values())
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
                logs = np.log(np.asarray(xs, dtype=float))
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
            stop_count_by_day_type=stop_counts_by_type,
            first_time_by_day_type=global_first_time,
            dwell_lognormal_by_coarse_category=dwell_lognormal,
            mobility_transition_graph=graph,
        )
        return agent_profiles, global_model


# -----------------------------
# Statistical Simulator (baseline)
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
        self._poi_coords_cache: Dict[str, Optional[Tuple[float, float]]] = {}
        self._coarse_by_poi_cache: Dict[str, str] = {}

    def _coarse_from_poi(self, poi_id: str) -> str:
        if poi_id in self._coarse_by_poi_cache:
            return self._coarse_by_poi_cache[poi_id]
        p = self.global_model.poi_by_id.get(poi_id)
        coarse = p.coarse_category if p is not None else (poi_id.split("#")[0].strip() if "#" in poi_id else "Unknown")
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
        p_probs = p_counts / p_counts.sum()
        g_probs = g_counts / g_counts.sum()
        w = float(self.params.w_personal_transition)
        probs = w * p_probs + (1.0 - w) * g_probs
        n = int(sample_from_probs(support, probs, rng))
        return max(0, n)

    def _sample_first_time(self, prof: AgentProfile, dt: str, rng: np.random.Generator) -> int:
        p = prof.personal_first_time_by_day_type.get(dt)
        g = self.global_model.first_time_by_day_type.get(dt)
        if p is None or len(p) != 1440:
            p = np.ones(1440) / 1440.0
        if g is None or len(g) != 1440:
            g = np.ones(1440) / 1440.0
        w = float(self.params.w_personal_transition)
        probs = w * p + (1.0 - w) * g
        probs = probs / probs.sum()
        return int(rng.choice(1440, p=probs))

    def _category_transition_probs(
        self,
        prof: AgentProfile,
        prev_cat: Optional[str],
        current_minute: int,
    ) -> Tuple[List[str], np.ndarray]:
        cats = self.global_model.coarse_categories
        if not cats:
            raise ValueError("Global model has no coarse_categories.")
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
        personal_probs = personal_counts / personal_counts.sum()
        global_probs = global_counts / global_counts.sum()
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
        if not np.isfinite(s) or s <= 0:
            probs = np.ones(len(cats), dtype=float) / len(cats)
        else:
            probs = probs / s
        return cats, probs

    def _poi_candidate_set(self, current_poi: str, target_coarse: str, topk: int) -> List[str]:
        candidates: List[str] = []
        neigh = self.global_model.mobility_transition_graph.get(current_poi, [])
        if neigh:
            for dst, _cnt in neigh[: max(1, topk)]:
                if self._coarse_from_poi(dst) == target_coarse:
                    candidates.append(dst)
        if not candidates:
            candidates = list(self.global_model.pois_by_coarse_category.get(target_coarse, []))
        seen: Set[str] = set()
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
        recently_visited: Optional[Set[str]] = None,
    ) -> str:
        topk = int(np.clip(int(self.params.candidate_set_topk_from_transition_graph), 5, 200))
        candidates = self._poi_candidate_set(current_poi, target_coarse, topk=topk)
        if not candidates:
            candidates = self.global_model.poi_ids[:]
        if not candidates:
            raise ValueError("No POIs available to sample from.")

        fallback_dist = max(0.1, float(prof.mobility_radius_km))
        logits = np.zeros(len(candidates), dtype=float)
        for i, poi_id in enumerate(candidates):
            aff = prof.anchor_poi_affinity.get(poi_id, 0.0)
            pop = self.global_model.global_poi_popularity.get(poi_id, 0.0)
            dist = self._distance_km(current_poi, poi_id, fallback_km=fallback_dist) if current_poi else fallback_dist
            recency_penalty = 0.5 if (recently_visited is not None and poi_id in recently_visited) else 0.0
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
            raise KeyError(f"Missing AgentProfile for agent_id='{agent_id}'.")
        dt = day_type_from_date(d)

        if prof.home_poi_candidates:
            homes = list(prof.home_poi_candidates.keys())
            weights = np.array([prof.home_poi_candidates[h] for h in homes], dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(homes)) / len(homes)
            current_poi = str(sample_from_probs(homes, weights, rng))
        else:
            current_poi = str(rng.choice(self.global_model.poi_ids))

        n_stops = self._sample_stop_count(prof, dt, rng)
        if n_stops <= 0:
            return DayTrajectory(agent_id=agent_id, d=d, events=tuple())

        current_time = self._sample_first_time(prof, dt, rng)
        prev_cat: Optional[str] = None
        events: List[Event] = []
        recently_visited: Set[str] = set()
        fatigue = 1.0

        for _i in range(n_stops):
            end_p = self._day_end_probability(current_time)
            end_p = min(0.95, max(0.0, end_p + 0.2 * (1.0 - fatigue)))
            if rng.random() < end_p:
                break

            cats, cat_probs = self._category_transition_probs(prof, prev_cat, current_time)
            target_cat = str(sample_from_probs(cats, cat_probs, rng))

            next_poi = self._choose_poi(prof, current_poi, target_cat, rng, recently_visited=recently_visited)

            fallback_dist = max(0.1, float(prof.mobility_radius_km))
            dist = self._distance_km(current_poi, next_poi, fallback_km=fallback_dist) if current_poi else fallback_dist
            travel = self._travel_time_minutes(dist)
            dwell = self._sample_dwell_minutes(target_cat, rng)

            arrival = int(np.clip(round(current_time + travel), 0, 1439))
            events.append(Event(place_type=target_cat, poi_id=next_poi, minute_of_day=arrival))

            current_poi = next_poi
            prev_cat = target_cat
            current_time = int(np.clip(round(arrival + dwell), 0, 1439))
            recently_visited.add(next_poi)

            fatigue = float(np.clip(fatigue - (travel + dwell) / (24.0 * 60.0), 0.0, 1.0))
            if target_cat.lower() == "home" or next_poi.lower().startswith("home#"):
                fatigue = float(np.clip(fatigue + 0.2, 0.0, 1.0))

            if current_time >= 1439:
                break

        return DayTrajectory(agent_id=agent_id, d=d, events=tuple(events))

    def rollout(self, validation_split: Dict[str, List[DayTrajectory]], n_runs: int = 5) -> Dict[int, List[DayTrajectory]]:
        results: Dict[int, List[DayTrajectory]] = {}
        for run in range(n_runs):
            rng = np.random.default_rng(self.base_seed + 1000 + run)
            sim_days: List[DayTrajectory] = []
            for agent_id, real_days in validation_split.items():
                for day in sorted(real_days, key=lambda x: x.d):
                    sim_days.append(self.simulate_day(agent_id=agent_id, d=day.d, rng=rng))
            results[run] = sim_days
        return results


# -----------------------------
# LLM pipeline (required by specification)
# -----------------------------
@dataclass(frozen=True)
class Pattern:
    text: str


@dataclass(frozen=True)
class Persona:
    text: str


@dataclass(frozen=True)
class Motivation:
    text: str


class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DummyLLMClient(LLMClient):
    """
    Offline fallback: does NOT call any external model.
    It expects the prompt may include a line starting with 'FALLBACK_TRAJECTORY:'.
    If not present, it returns a minimal valid empty-day trajectory.
    """

    def generate(self, prompt: str) -> str:
        m = re.search(r"^FALLBACK_TRAJECTORY:\s*(.+)$", prompt, flags=re.MULTILINE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"Target date:\s*(\d{4}-\d{2}-\d{2})", prompt)
        d = m2.group(1) if m2 else "1970-01-01"
        return f"Activities at {d}:."


def trajectory_to_string(day: DayTrajectory) -> str:
    parts = [f"Activities at {day.d.isoformat()}: "]
    if not day.events:
        parts.append(".")
        return "".join(parts)
    ev_strs = []
    for e in day.events:
        hh = e.minute_of_day // 60
        mm = e.minute_of_day % 60
        # Use HH:MM:SS with seconds always :00
        ev_strs.append(f"{e.poi_id} at {hh:02d}:{mm:02d}:00")
    parts.append(", ".join(ev_strs))
    parts.append(".")
    return "".join(parts)


def validate_trajectory_string(s: str) -> None:
    d = parse_date_from_activity_string(s)
    events_part = split_events_part(s)
    tokens = _split_event_tokens(events_part)
    for t in tokens:
        parse_event_token(t)
    if not isinstance(d, date):
        raise ValueError("Invalid date parsed (internal).")


class LLMPipelineGenerator:
    """
    LLM-based generator: Pattern -> Persona -> Motivation -> trajectory.
    """

    def __init__(
        self,
        llm: LLMClient,
        evaluator_place_type_to_coarse: Dict[str, str],
        fallback_simulator: Optional[MobilitySimulator] = None,
    ) -> None:
        self.llm = llm
        self.place_type_to_coarse = evaluator_place_type_to_coarse
        self.fallback_simulator = fallback_simulator

    def derive_pattern(self, agent_history: List[DayTrajectory]) -> Pattern:
        # Lightweight, deterministic summary that can be replaced by a real LLM prompt.
        counts = [len(d.events) for d in agent_history if d.events]
        mean_stops = float(np.mean(counts)) if counts else 0.0
        prompt = (
            "You are given a resident's historical daily trajectories. Summarize their habitual mobility pattern.\n"
            "Return a concise paragraph.\n"
            f"Stats: n_days={len(agent_history)}, mean_stops={mean_stops:.2f}\n"
        )
        text = self.llm.generate(prompt).strip()
        return Pattern(text=text)

    def derive_persona(self, agent_history: List[DayTrajectory]) -> Persona:
        prompt = (
            "Infer a short persona label/description for a resident based on historical trajectories.\n"
            "Examples: office worker, student, night-shift, retiree, delivery driver.\n"
            "Return 1-2 sentences.\n"
        )
        text = self.llm.generate(prompt).strip()
        return Persona(text=text)

    def derive_motivation(self, last_7_days: List[DayTrajectory], target_date: date) -> Motivation:
        prompt = (
            "Summarize the resident's day-specific motivation for the target date based on the past 7 days.\n"
            "Return 1-2 sentences.\n"
            f"Target date: {target_date.isoformat()}\n"
            f"Past days available: {len(last_7_days)}\n"
        )
        text = self.llm.generate(prompt).strip()
        return Motivation(text=text)

    def generate_day_string(
        self,
        agent_id: str,
        target_date: date,
        agent_history: List[DayTrajectory],
    ) -> str:
        history_sorted = sorted(agent_history, key=lambda x: x.d)
        last_7 = history_sorted[-7:]

        pattern = self.derive_pattern(history_sorted)
        persona = self.derive_persona(history_sorted)
        motivation = self.derive_motivation(last_7, target_date)

        fallback_line = None
        if self.fallback_simulator is not None:
            rng = np.random.default_rng(12345)
            fallback_day = self.fallback_simulator.simulate_day(agent_id=agent_id, d=target_date, rng=rng)
            fallback_line = trajectory_to_string(fallback_day)

        prompt = (
            "You are generating a one-day mobility trajectory.\n"
            "Output MUST be exactly one line in the format:\n"
            "Activities at YYYY-MM-DD: POI#id at HH:MM:SS, POI#id at HH:MM:SS.\n"
            "No explanations.\n\n"
            f"Pattern:\n{pattern.text}\n\n"
            f"Persona:\n{persona.text}\n\n"
            f"Motivation:\n{motivation.text}\n\n"
            f"Target date: {target_date.isoformat()}\n"
        )
        if fallback_line is not None:
            prompt += f"\nFALLBACK_TRAJECTORY: {fallback_line}\n"

        out = self.llm.generate(prompt).strip()
        validate_trajectory_string(out)
        return out


# -----------------------------
# Evaluation (spec-required SD/SI/DARD/STVD + JSD)
# -----------------------------
class Evaluator:
    def __init__(self, poi_by_id: Dict[str, POI], place_type_to_coarse: Dict[str, str]) -> None:
        self.poi_by_id = poi_by_id
        self.place_type_to_coarse = place_type_to_coarse

    def _coarse_from_event(self, e: Event) -> str:
        if e.place_type in self.place_type_to_coarse:
            return self.place_type_to_coarse[e.place_type]
        return e.place_type

    def _coords(self, poi_id: str) -> Optional[Tuple[float, float]]:
        p = self.poi_by_id.get(poi_id)
        if p is None or p.latitude is None or p.longitude is None:
            return None
        return (p.latitude, p.longitude)

    def _step_distances_km(self, days: List[DayTrajectory]) -> List[float]:
        out: List[float] = []
        for d in days:
            evs = list(d.events)
            for i in range(len(evs) - 1):
                ca = self._coords(evs[i].poi_id)
                cb = self._coords(evs[i + 1].poi_id)
                if ca is None or cb is None:
                    continue
                out.append(haversine_km(ca[0], ca[1], cb[0], cb[1]))
        return out

    def _step_intervals_min(self, days: List[DayTrajectory]) -> List[float]:
        out: List[float] = []
        for d in days:
            evs = list(d.events)
            for i in range(len(evs) - 1):
                dt = int(evs[i + 1].minute_of_day) - int(evs[i].minute_of_day)
                if dt > 0:
                    out.append(float(dt))
        return out

    def _hist_from_samples(
        self,
        samples: List[float],
        bin_edges: np.ndarray,
    ) -> np.ndarray:
        if len(samples) == 0:
            hist = np.zeros(len(bin_edges) - 1, dtype=float)
        else:
            hist, _ = np.histogram(np.asarray(samples, dtype=float), bins=bin_edges)
            hist = hist.astype(float)
        hist = hist + 1e-9
        return hist / hist.sum()

    def _auto_bin_edges(self, a: List[float], b: List[float], n_bins: int, min_max: Tuple[float, float]) -> np.ndarray:
        xs = [x for x in a + b if np.isfinite(x)]
        if not xs:
            lo, hi = min_max
            return np.linspace(lo, hi, n_bins + 1)
        arr = np.asarray(xs, dtype=float)
        lo = float(np.quantile(arr, 0.01))
        hi = float(np.quantile(arr, 0.99))
        lo = max(min_max[0], lo)
        hi = min(min_max[1], hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = min_max
        return np.linspace(lo, hi, n_bins + 1)

    def _dard_hist(self, days: List[DayTrajectory], time_bin_minutes: int, cats: List[str]) -> np.ndarray:
        n_tb = int(math.ceil(1440 / time_bin_minutes))
        idx = {c: i for i, c in enumerate(cats)}
        h = np.zeros((n_tb, len(cats)), dtype=float)
        for d in days:
            for e in d.events:
                c = self._coarse_from_event(e)
                if c not in idx:
                    continue
                tbin = min(n_tb - 1, max(0, int(e.minute_of_day) // time_bin_minutes))
                h[tbin, idx[c]] += 1.0
        v = h.reshape(-1)
        v = v + 1e-9
        return v / v.sum()

    def _stvd_hist(
        self,
        days: List[DayTrajectory],
        time_bin_minutes: int,
        lat_edges: np.ndarray,
        lon_edges: np.ndarray,
    ) -> np.ndarray:
        n_tb = int(math.ceil(1440 / time_bin_minutes))
        n_lat = len(lat_edges) - 1
        n_lon = len(lon_edges) - 1
        h = np.zeros((n_tb, n_lat, n_lon), dtype=float)
        for d in days:
            for e in d.events:
                coord = self._coords(e.poi_id)
                if coord is None:
                    continue
                lat, lon = coord
                tb = min(n_tb - 1, max(0, int(e.minute_of_day) // time_bin_minutes))
                li = int(np.searchsorted(lat_edges, lat, side="right") - 1)
                lo = int(np.searchsorted(lon_edges, lon, side="right") - 1)
                if 0 <= li < n_lat and 0 <= lo < n_lon:
                    h[tb, li, lo] += 1.0
        v = h.reshape(-1)
        v = v + 1e-9
        return v / v.sum()

    def compute_metrics(
        self,
        simulated_days: List[DayTrajectory],
        real_days: List[DayTrajectory],
        time_bin_minutes: int = 10,
        stvd_lat_bins: int = 30,
        stvd_lon_bins: int = 30,
        sd_bins: int = 40,
        si_bins: int = 40,
    ) -> Dict[str, Any]:
        # SD / SI distributions -> JSD
        sim_sd = self._step_distances_km(simulated_days)
        real_sd = self._step_distances_km(real_days)
        sd_edges = self._auto_bin_edges(sim_sd, real_sd, n_bins=sd_bins, min_max=(0.0, 200.0))
        p_sd = self._hist_from_samples(sim_sd, sd_edges)
        q_sd = self._hist_from_samples(real_sd, sd_edges)
        jsd_sd = js_divergence(p_sd, q_sd, eps=1e-9)

        sim_si = self._step_intervals_min(simulated_days)
        real_si = self._step_intervals_min(real_days)
        si_edges = self._auto_bin_edges(sim_si, real_si, n_bins=si_bins, min_max=(0.0, 24.0 * 60.0))
        p_si = self._hist_from_samples(sim_si, si_edges)
        q_si = self._hist_from_samples(real_si, si_edges)
        jsd_si = js_divergence(p_si, q_si, eps=1e-9)

        # DARD: joint (time_bin, category) -> JSD
        sim_cats = {self._coarse_from_event(e) for d in simulated_days for e in d.events}
        real_cats = {self._coarse_from_event(e) for d in real_days for e in d.events}
        cats = sorted(sim_cats | real_cats)
        if not cats:
            cats = ["Unknown"]
        p_dard = self._dard_hist(simulated_days, time_bin_minutes=time_bin_minutes, cats=cats)
        q_dard = self._dard_hist(real_days, time_bin_minutes=time_bin_minutes, cats=cats)
        jsd_dard = js_divergence(p_dard, q_dard, eps=1e-9)

        # STVD: joint (time_bin, lat_bin, lon_bin) -> JSD
        coords_all: List[Tuple[float, float]] = []
        for d in simulated_days + real_days:
            for e in d.events:
                c = self._coords(e.poi_id)
                if c is not None:
                    coords_all.append(c)
        if coords_all:
            lats = np.asarray([c[0] for c in coords_all], dtype=float)
            lons = np.asarray([c[1] for c in coords_all], dtype=float)
            lat_lo, lat_hi = float(np.quantile(lats, 0.001)), float(np.quantile(lats, 0.999))
            lon_lo, lon_hi = float(np.quantile(lons, 0.001)), float(np.quantile(lons, 0.999))
            if not (np.isfinite(lat_lo) and np.isfinite(lat_hi) and lat_hi > lat_lo):
                lat_lo, lat_hi = float(np.min(lats)), float(np.max(lats))
            if not (np.isfinite(lon_lo) and np.isfinite(lon_hi) and lon_hi > lon_lo):
                lon_lo, lon_hi = float(np.min(lons)), float(np.max(lons))
        else:
            lat_lo, lat_hi = 30.0, 50.0
            lon_lo, lon_hi = 120.0, 150.0
        lat_edges = np.linspace(lat_lo, lat_hi, stvd_lat_bins + 1)
        lon_edges = np.linspace(lon_lo, lon_hi, stvd_lon_bins + 1)

        p_stvd = self._stvd_hist(simulated_days, time_bin_minutes=time_bin_minutes, lat_edges=lat_edges, lon_edges=lon_edges)
        q_stvd = self._stvd_hist(real_days, time_bin_minutes=time_bin_minutes, lat_edges=lat_edges, lon_edges=lon_edges)
        jsd_stvd = js_divergence(p_stvd, q_stvd, eps=1e-9)

        return {
            "jsd_step_distance_sd": float(jsd_sd),
            "jsd_step_interval_si": float(jsd_si),
            "jsd_daily_activity_routine_distribution_dard": float(jsd_dard),
            "jsd_spatial_temporal_visits_distribution_stvd": float(jsd_stvd),
            "meta": {
                "n_days_simulated": len(simulated_days),
                "n_days_real": len(real_days),
                "time_bin_minutes": int(time_bin_minutes),
                "sd_bins": int(sd_bins),
                "si_bins": int(si_bins),
                "stvd_lat_bins": int(stvd_lat_bins),
                "stvd_lon_bins": int(stvd_lon_bins),
            },
        }


# -----------------------------
# Calibration (random search) using required JSD metrics
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
        eval_time_bin_minutes: int = 10,
    ) -> None:
        self.n_iterations = int(n_iterations)
        self.n_runs_per_eval = int(n_runs_per_eval)
        self.base_seed = int(base_seed)
        self.max_train_days_total = int(max_train_days_total)
        self.eval_time_bin_minutes = int(eval_time_bin_minutes)
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
        keys = [
            "jsd_step_distance_sd",
            "jsd_step_interval_si",
            "jsd_daily_activity_routine_distribution_dard",
            "jsd_spatial_temporal_visits_distribution_stvd",
        ]
        vals = [float(m.get(k, float("nan"))) for k in keys]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return 10.0
        return float(np.mean(vals))

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

            run_metrics: List[Dict[str, Any]] = []
            for _run_idx, sim_days in sim_runs.items():
                run_metrics.append(
                    evaluator.compute_metrics(
                        simulated_days=sim_days,
                        real_days=all_train_days,
                        time_bin_minutes=self.eval_time_bin_minutes,
                    )
                )

            # mean of scalar JSDs
            keys = [
                "jsd_step_distance_sd",
                "jsd_step_interval_si",
                "jsd_daily_activity_routine_distribution_dard",
                "jsd_spatial_temporal_visits_distribution_stvd",
            ]
            metrics_mean: Dict[str, Any] = {}
            for k in keys:
                vs = [float(m.get(k, float("nan"))) for m in run_metrics]
                vs = [v for v in vs if np.isfinite(v)]
                metrics_mean[k] = float(np.mean(vs)) if vs else float("nan")

            loss = self._loss_from_metrics(metrics_mean)
            if loss < best_loss:
                best_loss = loss
                best_params = cand
                best_metrics = metrics_mean

        if best_metrics is not None:
            sys.stderr.write(
                "Calibration done. Best training metrics (mean over runs): " + json.dumps(best_metrics, ensure_ascii=False) + "\n"
            )
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

    params_path = os.path.join(output_dir, "calibrated_parameters.json")
    metrics_path = os.path.join(output_dir, "evaluation_results_on_validation.json")
    sim_path = os.path.join(output_dir, "simulated_validation_trajectories.json")

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(vars(calibrated_params), f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    sim_out: Dict[str, Any] = {}
    for run_idx, days in simulated_rollouts.items():
        run_list = []
        for d in days:
            run_list.append(
                {
                    "agent_id": d.agent_id,
                    "date": d.d.isoformat(),
                    "events": [{"place_type": e.place_type, "poi_id": e.poi_id, "minute_of_day": e.minute_of_day} for e in d.events],
                    "trajectory_string": trajectory_to_string(d),
                }
            )
        sim_out[str(run_idx)] = run_list

    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(sim_out, f, indent=2, ensure_ascii=False)


# -----------------------------
# CLI and orchestration
# -----------------------------
def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily mobility trajectory simulator (statistical baseline + LLM pipeline).")

    p.add_argument("--seed", type=int, default=12345, help="Random seed.")
    p.add_argument("--data-dir", type=str, default=None, help="Data directory. If omitted, uses PROJECT_ROOT/DATA_PATH.")
    p.add_argument("--project-root", type=str, default=os.environ.get("PROJECT_ROOT"), help="Project root (if using DATA_PATH).")
    p.add_argument("--data-path", type=str, default=os.environ.get("DATA_PATH"), help="Relative data path (if using PROJECT_ROOT).")

    p.add_argument("--y-file", type=str, default="1921Y.json", help="Trajectory JSON filename.")
    p.add_argument("--poi-file", type=str, default=None, help="POI catalog filename (optional; auto-detected if omitted).")
    p.add_argument("--catto-file", type=str, default="catto.json", help="catto mapping filename.")

    p.add_argument("--output-dir", type=str, default=None, help="Output directory (default: <data-dir>/outputs).")

    p.add_argument("--generator", type=str, choices=["statistical", "llm"], default="statistical", help="Trajectory generator.")
    p.add_argument("--eval-time-bin-minutes", type=int, default=10, help="Time bin size for DARD/STVD evaluation.")
    p.add_argument("--calib-iters", type=int, default=30, help="Random search iterations for calibration.")
    p.add_argument("--calib-runs-per-eval", type=int, default=3, help="Stochastic runs per calibration eval.")
    p.add_argument("--eval-runs", type=int, default=10, help="Number of stochastic runs for validation rollout.")
    p.add_argument("--max-train-days-total", type=int, default=2000, help="Cap training days during calibration.")
    return p.parse_args(argv)


def _resolve_data_dir(args: argparse.Namespace) -> str:
    if args.data_dir:
        data_dir = args.data_dir
        if not os.path.isabs(data_dir):
            data_dir = os.path.abspath(data_dir)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")
        return data_dir

    if not args.project_root or not args.data_path:
        raise EnvironmentError(
            "Either provide --data-dir or set PROJECT_ROOT and DATA_PATH (or pass --project-root/--data-path)."
        )
    data_dir = os.path.join(args.project_root, args.data_path)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Resolved data directory does not exist: {data_dir}")
    return data_dir


def _split_train_validation_maps(
    split: Dict[str, Dict[str, List[DayTrajectory]]],
) -> Tuple[Dict[str, List[DayTrajectory]], Dict[str, List[DayTrajectory]], List[DayTrajectory], List[DayTrajectory]]:
    train: Dict[str, List[DayTrajectory]] = {}
    val: Dict[str, List[DayTrajectory]] = {}
    train_flat: List[DayTrajectory] = []
    val_flat: List[DayTrajectory] = []
    for aid, parts in split.items():
        train_days = parts.get("train", [])
        val_days = parts.get("validation", [])
        train[aid] = train_days
        val[aid] = val_days
        train_flat.extend(train_days)
        val_flat.extend(val_days)
    return train, val, train_flat, val_flat


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_cli(argv)
    set_global_seed(args.seed)

    data_dir = _resolve_data_dir(args)
    output_dir = args.output_dir or os.path.join(data_dir, "outputs")

    data = load_data(data_dir=data_dir, y_file=args.y_file, poi_file=args.poi_file, catto_file=args.catto_file)
    context = build_network_and_agents(data)
    split = holdout_split(context["trajectories"])
    train_map, val_map, train_flat, val_flat = _split_train_validation_maps(split)

    if not train_flat:
        raise ValueError("Training set is empty after holdout split. Cannot fit model.")

    if not val_flat:
        sys.stderr.write(
            "Warning: validation set is empty after holdout split (e.g., many agents with 1 day). "
            "Evaluation will run on empty validation.\n"
        )

    fitter = ModelFitter(
        poi_by_id=context["poi_by_id"],
        pois_by_coarse_category=context["pois_by_coarse_category"],
        coarse_categories=context["coarse_categories"],
        place_type_to_coarse=context["place_type_to_coarse"],
    )
    agent_profiles, global_model = fitter.fit(train_map)

    evaluator = Evaluator(
        poi_by_id=context["poi_by_id"],
        place_type_to_coarse=context["place_type_to_coarse"],
    )

    calibrator = RandomSearchCalibrator(
        n_iterations=args.calib_iters,
        n_runs_per_eval=args.calib_runs_per_eval,
        base_seed=args.seed,
        max_train_days_total=args.max_train_days_total,
        eval_time_bin_minutes=args.eval_time_bin_minutes,
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
        base_seed=args.seed,
    )

    # Optional LLM generator wiring: used to produce trajectory strings (not used for current statistical rollout).
    llm_generator: Optional[LLMPipelineGenerator] = None
    if args.generator == "llm":
        llm = DummyLLMClient()
        llm_generator = LLMPipelineGenerator(
            llm=llm,
            evaluator_place_type_to_coarse=context["place_type_to_coarse"],
            fallback_simulator=simulator,
        )

    simulated_rollouts = simulator.rollout(validation_split=val_map, n_runs=args.eval_runs)

    # Evaluate across runs
    per_run_metrics: Dict[str, Dict[str, Any]] = {}
    for run_idx, sim_days in simulated_rollouts.items():
        per_run_metrics[str(run_idx)] = evaluator.compute_metrics(
            simulated_days=sim_days,
            real_days=val_flat,
            time_bin_minutes=args.eval_time_bin_minutes,
        )

    # Summarize
    metric_keys = [
        "jsd_step_distance_sd",
        "jsd_step_interval_si",
        "jsd_daily_activity_routine_distribution_dard",
        "jsd_spatial_temporal_visits_distribution_stvd",
    ]
    summary: Dict[str, Any] = {"per_run": per_run_metrics, "summary": {}}
    for k in metric_keys:
        vals = [float(per_run_metrics[str(i)].get(k, float("nan"))) for i in simulated_rollouts.keys()]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            summary["summary"][k] = {"mean": float("nan"), "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
        else:
            arr = np.asarray(vals, dtype=float)
            summary["summary"][k] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p05": float(np.quantile(arr, 0.05)),
                "p95": float(np.quantile(arr, 0.95)),
            }

    summary["data_info"] = {
        "n_agents": len(context["agent_ids"]),
        "n_train_days": len(train_flat),
        "n_validation_days": len(val_flat),
        "data_dir": data_dir,
        "input_files": data["paths"],
        "generator": args.generator,
    }

    # If LLM generator enabled, also generate LLM strings for a small subset (for artifact completeness)
    if llm_generator is not None and val_flat:
        sample = val_flat[: min(20, len(val_flat))]
        llm_out: List[Dict[str, str]] = []
        by_agent_hist: Dict[str, List[DayTrajectory]] = {}
        for aid, ds in train_map.items():
            by_agent_hist[aid] = ds
        for dday in sample:
            hist = by_agent_hist.get(dday.agent_id, [])
            try:
                s = llm_generator.generate_day_string(dday.agent_id, dday.d, hist)
                llm_out.append({"agent_id": dday.agent_id, "date": dday.d.isoformat(), "trajectory_string": s})
            except Exception as e:
                llm_out.append({"agent_id": dday.agent_id, "date": dday.d.isoformat(), "trajectory_string": f"ERROR: {e}"})
        summary["llm_sample_outputs"] = llm_out

    save_results(
        output_dir=output_dir,
        calibrated_params=calibrated_params,
        evaluation_results=summary,
        simulated_rollouts=simulated_rollouts,
    )

    sys.stdout.write(json.dumps(summary["summary"], indent=2) + "\n")
    sys.stdout.write(f"Saved outputs to: {output_dir}\n")



# Execute main for both direct execution and sandbox wrapper invocation
main()