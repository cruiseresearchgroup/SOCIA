import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from hashlib import sha256
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# -----------------------------
# Logging
# -----------------------------
LOGGER = logging.getLogger("llmob_sim")


def setup_logging(verbosity: int = 0) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


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
            f"Check --data-dir/--project-root/--data-path and the file name."
        )


def safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(x, eps))


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"softmax_temperature must be > 0. Got: {temperature}")
    z = logits / temperature
    z = z - np.max(z)
    exp_z = np.exp(z)
    s = exp_z.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(logits) / len(logits)
    return exp_z / s


def sample_from_probs(items: Sequence[Any], probs: np.ndarray, rng: np.random.Generator) -> Any:
    if len(items) != len(probs):
        raise ValueError("items and probs must have the same length.")
    if len(items) == 0:
        raise ValueError("Cannot sample from empty items.")
    p = np.asarray(probs, dtype=float)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        p = np.ones_like(p) / len(p)
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
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size != q.size:
        n = max(p.size, q.size)
        p = np.pad(p, (0, n - p.size))
        q = np.pad(q, (0, n - q.size))
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def wasserstein_1d(u: Sequence[float], v: Sequence[float]) -> float:
    """
    Efficient 1D Wasserstein-1 distance between empirical distributions.

    Computes integral |F_u(x)-F_v(x)| dx using sorted samples and a merge scan.
    """
    u = np.asarray(list(u), dtype=float)
    v = np.asarray(list(v), dtype=float)
    u = u[np.isfinite(u)]
    v = v[np.isfinite(v)]
    if u.size == 0 or v.size == 0:
        return float("nan")
    u.sort()
    v.sort()
    n = u.size
    m = v.size

    i = j = 0
    fu = fv = 0.0
    last_x = min(u[0], v[0])
    area = 0.0
    inv_n = 1.0 / n
    inv_m = 1.0 / m

    while i < n or j < m:
        next_u = u[i] if i < n else float("inf")
        next_v = v[j] if j < m else float("inf")
        x = next_u if next_u <= next_v else next_v

        dx = x - last_x
        if dx > 0:
            area += abs(fu - fv) * dx
            last_x = x

        while i < n and u[i] == x:
            fu += inv_n
            i += 1
        while j < m and v[j] == x:
            fv += inv_m
            j += 1

    return float(area)


def day_type_from_date(d: date) -> str:
    return "weekend" if d.weekday() >= 5 else "weekday"


_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s*$")


def minute_of_day_from_hms(hms: str) -> int:
    """
    Convert H:MM:SS or HH:MM:SS to minute-of-day (0..1439), rounding seconds to nearest minute.

    Accepts single-digit hour (e.g., '0:00:00').
    """
    m = _TIME_RE.match(hms)
    if not m:
        raise ValueError(f"Invalid time token: {hms}")
    hh, mm, ss = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59 or ss < 0 or ss > 59:
        raise ValueError(f"Out-of-range time: {hms}")
    minute = hh * 60 + mm + (1 if ss >= 30 else 0)
    return min(1439, max(0, minute))


def hms_from_minute(minute: int, *, pad_hour: bool = True) -> str:
    minute = int(np.clip(minute, 0, 1439))
    hh = minute // 60
    mm = minute % 60
    if pad_hour:
        return f"{hh:02d}:{mm:02d}:00"
    return f"{hh}:{mm:02d}:00"


def parse_date_from_activity_string(s: str) -> date:
    m = re.search(r"Activities at\s+(\d{4}-\d{2}-\d{2})", s)
    if not m:
        raise ValueError("Failed to parse date from daily_activity_string.")
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def split_events_part(s: str) -> str:
    m = re.search(r"Activities at\s+\d{4}-\d{2}-\d{2}\s*[:\-]?\s*(.*)$", s)
    if not m:
        raise ValueError("Failed to parse events from daily_activity_string.")
    return m.group(1).strip()


def _strip_trailing_punct(token: str) -> str:
    return token.strip().rstrip(" .;,")


def parse_event_token(token: str) -> Tuple[str, str, int]:
    """
    Parse an event token into (place_type, poi_id, minute_of_day).

    Token examples:
      - "Convenience Store#780 at 00:20:00"
      - "small lodging establishment#793 at 0:00:00."
    """
    token = _strip_trailing_punct(token)
    if not token:
        raise ValueError("Empty event token encountered.")
    # Split on " at " to reduce false positives if place names contain times.
    if " at " in token:
        left, right = token.rsplit(" at ", 1)
        poi_part = left.strip()
        time_part = right.strip()
    else:
        # Fallback: time at end
        tm = re.search(r"(\d{1,2}:\d{2}:\d{2})\s*$", token)
        if not tm:
            raise ValueError(f"Failed to parse time from event token: '{token}'.")
        time_part = tm.group(1)
        poi_part = token[: tm.start(1)].strip()
    minute = minute_of_day_from_hms(time_part)

    if "#" not in poi_part:
        place_type = poi_part
        poi_id = poi_part
    else:
        idx = poi_part.rfind("#")
        place_type = poi_part[:idx].strip()
        poi_id = poi_part.strip()
    if not place_type:
        place_type = poi_id.split("#")[0].strip() if "#" in poi_id else poi_id
    return place_type, poi_id, minute


def normalize_counts_dict(counts: Dict[Any, float], alpha: float = 0.0) -> Dict[Any, float]:
    if not counts:
        return {}
    items = list(counts.items())
    total = sum(max(0.0, float(v)) + alpha for _k, v in items)
    if total <= 0:
        return {k: 1.0 / len(items) for k, _v in items}
    return {k: (max(0.0, float(v)) + alpha) / total for k, v in items}


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Event:
    place_type: str  # fine POI type (prefix before '#')
    poi_id: str      # full 'Type#id'
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
    category_preference: Dict[str, float] = field(default_factory=dict)  # coarse category preference
    time_of_day_priors_by_category: Dict[str, np.ndarray] = field(default_factory=dict)  # coarse -> 1440
    mobility_radius_km: float = 2.0
    personal_category_transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    personal_stop_count_by_day_type: Dict[str, Dict[int, int]] = field(default_factory=dict)
    personal_first_time_by_day_type: Dict[str, np.ndarray] = field(default_factory=dict)  # day_type -> 1440


@dataclass
class GlobalModel:
    coarse_categories: List[str]
    poi_ids: List[str]
    poi_by_id: Dict[str, POI]
    pois_by_coarse_category: Dict[str, List[str]]

    global_poi_popularity: Dict[str, float] = field(default_factory=dict)
    global_category_transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    stop_count_by_day_type: Dict[str, Dict[int, int]] = field(default_factory=dict)
    first_time_by_day_type: Dict[str, np.ndarray] = field(default_factory=dict)  # day_type -> 1440
    dwell_lognormal_by_coarse_category: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # coarse -> (mu, sigma)
    mobility_transition_graph: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)  # src -> [(dst, count)]


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


@dataclass(frozen=True)
class ContactEvent:
    run_idx: int
    day: date
    time_bin: int
    poi_id: str
    agent_a: str
    agent_b: str


# -----------------------------
# LLM Abstractions
# -----------------------------
class LLMClient:
    def generate_text(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.2) -> str:
        raise NotImplementedError


class FileCache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key_to_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[str]:
        path = self._key_to_path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "text" in obj and isinstance(obj["text"], str):
                return obj["text"]
        except Exception:
            return None
        return None

    def put(self, key: str, text: str) -> None:
        path = self._key_to_path(key)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"text": text, "ts": time.time()}, f, ensure_ascii=False)
        os.replace(tmp, path)


class CachedLLMClient(LLMClient):
    def __init__(self, backend: LLMClient, cache: Optional[FileCache] = None, rate_limit_qps: float = 0.0) -> None:
        self.backend = backend
        self.cache = cache
        self.rate_limit_qps = float(rate_limit_qps)
        self._last_call = 0.0

    def generate_text(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.2) -> str:
        key_src = (system or "") + "\n" + prompt + f"\n__temp__={temperature}"
        key = sha256(key_src.encode("utf-8")).hexdigest()
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        if self.rate_limit_qps > 0:
            min_dt = 1.0 / self.rate_limit_qps
            now = time.time()
            dt = now - self._last_call
            if dt < min_dt:
                time.sleep(min_dt - dt)

        text = self.backend.generate_text(prompt, system=system, temperature=temperature)
        self._last_call = time.time()
        if self.cache:
            self.cache.put(key, text)
        return text


class HeuristicLLMClient(LLMClient):
    """
    Deterministic fallback "LLM" that returns:
      - For Pattern/Persona/Motivation: short templated summaries
      - For trajectory generation: returns the provided DRAFT trajectory string verbatim if present,
        otherwise returns a minimal valid trajectory.
    """

    def generate_text(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.2) -> str:
        _ = temperature
        txt = prompt

        # Trajectory generation: if "DRAFT_TRAJECTORY:" exists, return the draft line after it.
        m = re.search(r"DRAFT_TRAJECTORY:\s*(Activities at [^\n]+)", txt)
        if m:
            out = m.group(1).strip()
            out = out.rstrip()
            if not out.endswith("."):
                out += "."
            return out

        if "TASK: DERIVE_PATTERN" in txt:
            return "Pattern: Regular daily outings with a mix of errands and leisure; start time tends to be late morning; visits cluster within a moderate travel radius."
        if "TASK: INFER_PERSONA" in txt:
            return "Persona: Resident with flexible daytime schedule (non-commuter / irregular worker)."
        if "TASK: SUMMARIZE_MOTIVATION" in txt:
            return "Motivation: Follow recent routine with slight variation; complete one errand trip and one leisure/food stop."

        # Default minimal valid output
        dm = re.search(r"TARGET_DATE:\s*(\d{4}-\d{2}-\d{2})", txt)
        d = dm.group(1) if dm else "2019-01-01"
        return f"Activities at {d}: Convenience Store#1 at 10:00:00."


@dataclass(frozen=True)
class Pattern:
    text: str


@dataclass(frozen=True)
class Persona:
    text: str


@dataclass(frozen=True)
class Motivation:
    text: str


class PatternExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def derive(self, agent_id: str, history_days: List[DayTrajectory], place_type_to_coarse: Dict[str, str]) -> Pattern:
        summary = summarize_history_stats(history_days, place_type_to_coarse, max_days=60)
        prompt = (
            "TASK: DERIVE_PATTERN\n"
            f"AGENT_ID: {agent_id}\n"
            "INSTRUCTIONS: Produce a concise natural-language Pattern summary of habitual routines, timing, and activity mix.\n"
            "DO NOT include any code or JSON.\n\n"
            f"HISTORY_SUMMARY:\n{summary}\n"
        )
        text = self.llm.generate_text(prompt, system="You are a mobility behavior analyst. Write only the pattern summary.")
        return Pattern(text=text.strip())


class PersonaInferer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def infer(self, agent_id: str, history_days: List[DayTrajectory], place_type_to_coarse: Dict[str, str]) -> Persona:
        summary = summarize_history_stats(history_days, place_type_to_coarse, max_days=60)
        prompt = (
            "TASK: INFER_PERSONA\n"
            f"AGENT_ID: {agent_id}\n"
            "INSTRUCTIONS: Infer a plausible persona label and short description (e.g., office worker, student, night-shift, retiree).\n"
            "Return 1-2 sentences.\n\n"
            f"HISTORY_SUMMARY:\n{summary}\n"
        )
        text = self.llm.generate_text(prompt, system="You are a behavioral scientist. Output only the persona statement.")
        return Persona(text=text.strip())


class MotivationSummarizer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def summarize(
        self,
        agent_id: str,
        target_date: date,
        recent_days: List[DayTrajectory],
        place_type_to_coarse: Dict[str, str],
    ) -> Motivation:
        summary = summarize_history_stats(recent_days, place_type_to_coarse, max_days=7)
        prompt = (
            "TASK: SUMMARIZE_MOTIVATION\n"
            f"AGENT_ID: {agent_id}\n"
            f"TARGET_DATE: {target_date.isoformat()}\n"
            "INSTRUCTIONS: Summarize likely day-specific intent based on the past 7 days. 1-2 sentences.\n\n"
            f"PAST_7_DAYS_SUMMARY:\n{summary}\n"
        )
        text = self.llm.generate_text(prompt, system="You are inferring daily intent. Output only the motivation statement.")
        return Motivation(text=text.strip())


class LLMTrajectoryGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(
        self,
        agent_id: str,
        target_date: date,
        pattern: Pattern,
        persona: Persona,
        motivation: Motivation,
        allowed_pois_hint: Optional[List[str]],
        draft_trajectory_string: Optional[str],
    ) -> str:
        allowed_hint = ""
        if allowed_pois_hint:
            # Avoid huge prompts: just a small list.
            sample = allowed_pois_hint[:30]
            allowed_hint = "ALLOWED_POIS_HINT (subset):\n" + ", ".join(sample) + "\n"

        draft = ""
        if draft_trajectory_string:
            draft = "DRAFT_TRAJECTORY:\n" + draft_trajectory_string.strip() + "\n"

        prompt = (
            "You must output a single daily trajectory string EXACTLY in this format:\n"
            "Activities at YYYY-MM-DD: POI#id at HH:MM:SS, POI#id at HH:MM:SS, ...\n"
            "Rules:\n"
            "- Output ONLY the trajectory string. No rationale.\n"
            "- Use comma+space separators.\n"
            "- End with a period.\n"
            "- Keep times within 00:00:00-23:59:59.\n\n"
            f"AGENT_ID: {agent_id}\n"
            f"TARGET_DATE: {target_date.isoformat()}\n\n"
            f"PATTERN:\n{pattern.text}\n\n"
            f"PERSONA:\n{persona.text}\n\n"
            f"MOTIVATION:\n{motivation.text}\n\n"
            f"{allowed_hint}\n"
            f"{draft}\n"
        )

        out = self.llm.generate_text(prompt, system="You are simulating a resident's day. Output only the trajectory string.")
        out = out.strip()
        # Minimal post-validation/fixups to enforce format
        if not out.startswith("Activities at "):
            out = f"Activities at {target_date.isoformat()}: " + out
        if not out.endswith("."):
            out = out.rstrip(" ,;") + "."
        return out


# -----------------------------
# Parsing / formatting trajectories (I/O compatibility)
# -----------------------------
def parse_trajectories(y_obj: Any, *, strict: bool = True) -> List[DayTrajectory]:
    if not isinstance(y_obj, dict):
        raise ValueError("1921Y.json must be a JSON object (dict) mapping person_id to records.")
    trajs: List[DayTrajectory] = []
    bad_days = 0
    bad_tokens = 0

    for agent_id, record in y_obj.items():
        if not isinstance(agent_id, str) or not agent_id:
            if strict:
                raise ValueError("Invalid agent_id key in 1921Y.json; expected non-empty string keys.")
            LOGGER.warning("Skipping invalid agent_id key: %r", agent_id)
            continue

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
                if strict:
                    raise ValueError(f"Unsupported record structure for agent_id '{agent_id}'.")
                LOGGER.warning("Skipping agent %s: unsupported daily_activity_string format", agent_id)
                continue
        else:
            if strict:
                raise ValueError(f"Unsupported record type for agent_id '{agent_id}': {type(record)}.")
            LOGGER.warning("Skipping agent %s: unsupported record type %s", agent_id, type(record))
            continue

        for s in strings:
            try:
                d = parse_date_from_activity_string(s)
                events_part = split_events_part(s)
            except Exception as e:
                bad_days += 1
                if strict:
                    raise
                LOGGER.warning("Skipping day for agent=%s due to parse error: %s | raw=%r", agent_id, e, s[:2000])
                continue

            if not events_part:
                trajs.append(DayTrajectory(agent_id=agent_id, d=d, events=tuple()))
                continue

            # Split by commas; tolerate trailing punctuation and stray whitespace.
            tokens = [t.strip() for t in events_part.split(",") if t.strip()]
            events: List[Event] = []
            for token in tokens:
                try:
                    place_type, poi_id, minute = parse_event_token(token)
                    events.append(Event(place_type=place_type, poi_id=poi_id, minute_of_day=minute))
                except Exception as e:
                    bad_tokens += 1
                    if strict:
                        raise
                    LOGGER.info(
                        "Skipping bad token agent=%s date=%s token=%r err=%s",
                        agent_id,
                        d.isoformat(),
                        token,
                        e,
                    )
                    continue
            events.sort(key=lambda e: e.minute_of_day)
            trajs.append(DayTrajectory(agent_id=agent_id, d=d, events=tuple(events)))

    if not trajs:
        raise ValueError("No trajectories were parsed from 1921Y.json. Check the input format.")
    if bad_days or bad_tokens:
        LOGGER.warning("Parsing completed with issues: bad_days=%d bad_tokens=%d strict=%s", bad_days, bad_tokens, strict)
    return trajs


def trajectory_to_activity_string(day: DayTrajectory, *, pad_hour: bool = True) -> str:
    parts: List[str] = []
    for e in day.events:
        t = hms_from_minute(e.minute_of_day, pad_hour=pad_hour)
        parts.append(f"{e.poi_id} at {t}")
    if parts:
        return f"Activities at {day.d.isoformat()}: " + ", ".join(parts) + "."
    return f"Activities at {day.d.isoformat()}:."


# -----------------------------
# Data loading and preprocessing
# -----------------------------
def load_json(path: str) -> Any:
    require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_poi_catalog_path(data_dir: str, preferred: Optional[str] = None) -> str:
    candidates = []
    if preferred:
        candidates.append(os.path.join(data_dir, preferred))
    candidates.extend(
        [
            os.path.join(data_dir, "poi_category_192021_longitude_latitude_complement_alignment_clean.json"),
            os.path.join(data_dir, "poi_category_192021_longitude_latitude.json"),
            os.path.join(data_dir, "poi_category_192021_longitude_latitude_complement_alignment_clean.jsonl"),
        ]
    )
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Could not find a POI catalog file in data_dir. Tried: "
        + ", ".join(os.path.basename(x) for x in candidates)
    )


def extract_place_type_to_coarse(catto_obj: Any) -> Dict[str, str]:
    if isinstance(catto_obj, dict) and "place_type_to_coarse_category" in catto_obj:
        mapping = catto_obj["place_type_to_coarse_category"]
    else:
        mapping = catto_obj
    if not isinstance(mapping, dict):
        raise ValueError("catto.json must contain a dict mapping place_type to coarse_category.")
    out: Dict[str, str] = {}
    for k, v in mapping.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def build_poi_catalog(
    poi_catalog_obj: Any,
    place_type_to_coarse: Dict[str, str],
) -> Tuple[Dict[str, POI], Dict[str, List[str]], List[str]]:
    """
    Supports formats:
      A) {category: [[lat, lon, poi_id], ...], ...}  (common)
      B) {poi_id: [lat, lon] or {"lat":..,"lon":..} } (alternative)
    """
    poi_by_id: Dict[str, POI] = {}
    pois_by_coarse: Dict[str, List[str]] = {}

    if isinstance(poi_catalog_obj, dict) and poi_catalog_obj:
        # Detect format A if values are lists of records
        is_format_a = any(isinstance(v, list) and v and isinstance(v[0], list) for v in poi_catalog_obj.values())
        if is_format_a:
            for category, records in poi_catalog_obj.items():
                if not isinstance(category, str) or not category.strip():
                    continue
                if not isinstance(records, list):
                    continue
                for rec in records:
                    if not (isinstance(rec, list) and len(rec) >= 3):
                        continue
                    lat_raw, lon_raw, poi_id_raw = rec[0], rec[1], rec[2]
                    if not isinstance(poi_id_raw, str) or not poi_id_raw.strip():
                        continue
                    poi_id = poi_id_raw.strip()
                    latitude = None
                    longitude = None
                    try:
                        latitude = float(lat_raw)
                        longitude = float(lon_raw)
                    except Exception:
                        latitude = None
                        longitude = None

                    coarse = place_type_to_coarse.get(category.strip(), category.strip())
                    poi = POI(
                        poi_id=poi_id,
                        category=category.strip(),
                        coarse_category=coarse,
                        latitude=latitude,
                        longitude=longitude,
                    )
                    poi_by_id[poi_id] = poi
                    pois_by_coarse.setdefault(coarse, []).append(poi_id)
        else:
            # Format B: poi_id -> coords
            for poi_id, rec in poi_catalog_obj.items():
                if not isinstance(poi_id, str) or not poi_id.strip():
                    continue
                poi_id = poi_id.strip()
                place_type = poi_id.split("#")[0].strip() if "#" in poi_id else poi_id
                coarse = place_type_to_coarse.get(place_type, place_type)
                latitude = None
                longitude = None
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    try:
                        latitude = float(rec[0])
                        longitude = float(rec[1])
                    except Exception:
                        pass
                elif isinstance(rec, dict):
                    try:
                        latitude = float(rec.get("lat"))
                        longitude = float(rec.get("lon"))
                    except Exception:
                        pass
                poi = POI(
                    poi_id=poi_id,
                    category=place_type,
                    coarse_category=coarse,
                    latitude=latitude,
                    longitude=longitude,
                )
                poi_by_id[poi_id] = poi
                pois_by_coarse.setdefault(coarse, []).append(poi_id)
    else:
        raise ValueError("POI catalog must be a non-empty JSON object.")

    if not poi_by_id:
        raise ValueError("POI catalog is empty after parsing.")
    coarse_categories = sorted(pois_by_coarse.keys())
    return poi_by_id, pois_by_coarse, coarse_categories


def load_data(data_dir: str, *, poi_catalog_filename: Optional[str] = None) -> Dict[str, Any]:
    y_file = os.path.join(data_dir, "1921Y.json")
    catto_file = os.path.join(data_dir, "catto.json")
    poi_file = detect_poi_catalog_path(data_dir, preferred=poi_catalog_filename)

    data = {
        "y": load_json(y_file),
        "poi_catalog": load_json(poi_file),
        "catto": load_json(catto_file),
        "paths": {"1921Y.json": y_file, "poi_catalog": poi_file, "catto": catto_file},
    }
    return data


def filter_days_by_years(trajs: List[DayTrajectory], years: Optional[Set[int]]) -> List[DayTrajectory]:
    if not years:
        return trajs
    return [t for t in trajs if t.d.year in years]


def holdout_split(trajectories: List[DayTrajectory]) -> Dict[str, Dict[str, List[DayTrajectory]]]:
    by_agent: Dict[str, List[DayTrajectory]] = {}
    for t in trajectories:
        by_agent.setdefault(t.agent_id, []).append(t)
    split: Dict[str, Dict[str, List[DayTrajectory]]] = {}
    for agent_id, days in by_agent.items():
        days_sorted = sorted(days, key=lambda x: x.d)
        n = len(days_sorted)
        if n == 1:
            split[agent_id] = {"train": days_sorted, "validation": []}
        else:
            train_n = int(math.floor(0.8 * n))
            train_n = max(1, min(n - 1, train_n))
            split[agent_id] = {"train": days_sorted[:train_n], "validation": days_sorted[train_n:]}
    return split


def split_train_validation_maps(
    split: Dict[str, Dict[str, List[DayTrajectory]]],
) -> Tuple[Dict[str, List[DayTrajectory]], Dict[str, List[DayTrajectory]], List[DayTrajectory], List[DayTrajectory]]:
    train: Dict[str, List[DayTrajectory]] = {}
    val: Dict[str, List[DayTrajectory]] = {}
    train_flat: List[DayTrajectory] = []
    val_flat: List[DayTrajectory] = []
    for aid, parts in split.items():
        tr = parts.get("train", [])
        va = parts.get("validation", [])
        train[aid] = tr
        val[aid] = va
        train_flat.extend(tr)
        val_flat.extend(va)
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
        if poi_id in self.poi_by_id:
            return self.poi_by_id[poi_id].coarse_category
        return poi_id.split("#")[0].strip() if "#" in poi_id else place_type

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
                    if coarse not in tod_by_cat:
                        tod_by_cat[coarse] = np.zeros(1440)
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
                prof.home_poi_candidates = normalize_counts_dict(dict(top))
            elif prof.anchor_poi_affinity:
                top_poi = max(prof.anchor_poi_affinity.items(), key=lambda x: x[1])[0]
                prof.home_poi_candidates = {top_poi: 1.0}
            else:
                prof.home_poi_candidates = {}

            agent_profiles[agent_id] = prof

        pop_scores = {poi_id: math.log1p(c) for poi_id, c in global_poi_visits.items()}
        global_poi_popularity = normalize_counts_dict(pop_scores)
        for poi_id, poi in self.poi_by_id.items():
            poi.base_attractiveness = global_poi_popularity.get(poi_id, 0.0)

        global_first_time: Dict[str, np.ndarray] = {}
        for dt in ("weekday", "weekend"):
            h = first_time_by_type[dt]
            s = float(h.sum())
            global_first_time[dt] = (h / s) if s > 0 else (np.ones(1440) / 1440.0)

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
# Simulator (statistical draft generator)
# -----------------------------
class MobilitySimulator:
    def __init__(
        self,
        global_model: GlobalModel,
        agent_profiles: Dict[str, AgentProfile],
        place_type_to_coarse: Dict[str, str],
        params: CalibratedParameters,
        base_seed: int,
        time_bin_minutes: int = 10,
    ) -> None:
        self.global_model = global_model
        self.agent_profiles = agent_profiles
        self.place_type_to_coarse = place_type_to_coarse
        self.params = params
        self.base_seed = base_seed
        self.time_bin_minutes = int(time_bin_minutes)

        self._poi_coords_cache: Dict[str, Optional[Tuple[float, float]]] = {}
        self._coarse_by_poi_cache: Dict[str, str] = {}
        self._type_by_poi_cache: Dict[str, str] = {}

    def _coarse_from_poi(self, poi_id: str) -> str:
        if poi_id in self._coarse_by_poi_cache:
            return self._coarse_by_poi_cache[poi_id]
        p = self.global_model.poi_by_id.get(poi_id)
        coarse = p.coarse_category if p is not None else (poi_id.split("#")[0].strip() if "#" in poi_id else "Unknown")
        self._coarse_by_poi_cache[poi_id] = coarse
        return coarse

    def _type_from_poi(self, poi_id: str) -> str:
        if poi_id in self._type_by_poi_cache:
            return self._type_by_poi_cache[poi_id]
        p = self.global_model.poi_by_id.get(poi_id)
        t = p.category if p is not None else (poi_id.split("#")[0].strip() if "#" in poi_id else poi_id)
        self._type_by_poi_cache[poi_id] = t
        return t

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
        if p is None:
            p = np.ones(1440) / 1440.0
        if g is None:
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
        probs = (probs / s) if s > 0 and np.isfinite(s) else (np.ones(len(cats)) / len(cats))
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

    def simulate_day(
        self,
        agent_id: str,
        d: date,
        rng: np.random.Generator,
        *,
        return_intervals: bool = False,
    ) -> Tuple[DayTrajectory, Optional[List[Tuple[str, int, int]]]]:
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
            day_traj = DayTrajectory(agent_id=agent_id, d=d, events=tuple())
            return (day_traj, []) if return_intervals else (day_traj, None)

        current_time = self._sample_first_time(prof, dt, rng)
        prev_cat: Optional[str] = None
        events: List[Event] = []
        intervals: List[Tuple[str, int, int]] = []  # (poi_id, start_min, end_min)
        recently_visited: Set[str] = set()
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
            dist = self._distance_km(current_poi, next_poi, fallback_km=fallback_dist) if current_poi else fallback_dist
            travel = self._travel_time_minutes(dist)
            dwell = self._sample_dwell_minutes(target_coarse, rng)

            arrival = int(np.clip(round(current_time + travel), 0, 1439))
            depart = int(np.clip(round(arrival + dwell), 0, 1440))  # allow 1440 for interval end
            place_type = self._type_from_poi(next_poi)

            events.append(Event(place_type=place_type, poi_id=next_poi, minute_of_day=arrival))
            intervals.append((next_poi, arrival, depart))

            current_poi = next_poi
            prev_cat = target_coarse
            current_time = int(np.clip(depart, 0, 1439))
            recently_visited.add(next_poi)

            fatigue = float(np.clip(fatigue - (travel + dwell) / (24.0 * 60.0), 0.0, 1.0))
            if place_type.lower() == "home" or next_poi.lower().startswith("home#"):
                fatigue = float(np.clip(fatigue + 0.2, 0.0, 1.0))

            if current_time >= 1439:
                break

        day_traj = DayTrajectory(agent_id=agent_id, d=d, events=tuple(events))
        return (day_traj, intervals) if return_intervals else (day_traj, None)

    def rollout(
        self,
        validation_split: Dict[str, List[DayTrajectory]],
        n_runs: int = 5,
        *,
        compute_contacts: bool = False,
        max_contacts_per_run: int = 200000,
    ) -> Tuple[Dict[int, List[DayTrajectory]], Dict[int, List[ContactEvent]]]:
        results: Dict[int, List[DayTrajectory]] = {}
        contacts_by_run: Dict[int, List[ContactEvent]] = {}

        for run in range(n_runs):
            rng = np.random.default_rng(self.base_seed + 1000 + run)
            sim_days: List[DayTrajectory] = []
            intervals_by_day: Dict[date, List[Tuple[str, str, int, int]]] = {}
            # (poi_id, agent_id, start, end)

            for agent_id, real_days in validation_split.items():
                for day in sorted(real_days, key=lambda x: x.d):
                    sim, intervals = self.simulate_day(agent_id=agent_id, d=day.d, rng=rng, return_intervals=compute_contacts)
                    sim_days.append(sim)
                    if compute_contacts and intervals is not None:
                        for poi_id, s, e in intervals:
                            intervals_by_day.setdefault(day.d, []).append((poi_id, agent_id, s, e))

            results[run] = sim_days

            if compute_contacts:
                contacts_by_run[run] = compute_contacts_from_intervals(
                    run_idx=run,
                    intervals_by_day=intervals_by_day,
                    time_bin_minutes=self.time_bin_minutes,
                    max_contacts=max_contacts_per_run,
                )
            else:
                contacts_by_run[run] = []

        return results, contacts_by_run


def compute_contacts_from_intervals(
    *,
    run_idx: int,
    intervals_by_day: Dict[date, List[Tuple[str, str, int, int]]],
    time_bin_minutes: int,
    max_contacts: int,
) -> List[ContactEvent]:
    """
    Co-location: two agents have contact if they are at the same POI during the same time bin.

    Note: derived from simulated dwell intervals; not computed from real data.
    """
    tb = max(1, int(time_bin_minutes))
    contacts: List[ContactEvent] = []
    for d, records in intervals_by_day.items():
        bins_map: Dict[Tuple[int, str], List[str]] = {}  # (bin, poi_id) -> [agent_id]
        for poi_id, agent_id, start, end in records:
            if end <= start:
                continue
            b0 = int(start // tb)
            b1 = int((max(start, end - 1)) // tb)
            for b in range(b0, b1 + 1):
                bins_map.setdefault((b, poi_id), []).append(agent_id)

        for (b, poi_id), agents in bins_map.items():
            if len(agents) < 2:
                continue
            # Deduplicate within bin-poi
            uniq = sorted(set(agents))
            for a, c in combinations(uniq, 2):
                contacts.append(ContactEvent(run_idx=run_idx, day=d, time_bin=b, poi_id=poi_id, agent_a=a, agent_b=c))
                if len(contacts) >= max_contacts:
                    return contacts
    return contacts


# -----------------------------
# Spec metrics evaluator (SD, SI, DARD, STVD with JSD)
# -----------------------------
class Evaluator:
    def __init__(
        self,
        poi_by_id: Dict[str, POI],
        place_type_to_coarse: Dict[str, str],
        *,
        time_bin_minutes: int = 10,
        stvd_lat_bins: int = 50,
        stvd_lon_bins: int = 50,
    ) -> None:
        self.poi_by_id = poi_by_id
        self.place_type_to_coarse = place_type_to_coarse
        self.time_bin_minutes = int(time_bin_minutes)
        self.stvd_lat_bins = int(stvd_lat_bins)
        self.stvd_lon_bins = int(stvd_lon_bins)

        lats = [p.latitude for p in poi_by_id.values() if p.latitude is not None and np.isfinite(p.latitude)]
        lons = [p.longitude for p in poi_by_id.values() if p.longitude is not None and np.isfinite(p.longitude)]
        if lats and lons:
            self.lat_min = float(min(lats))
            self.lat_max = float(max(lats))
            self.lon_min = float(min(lons))
            self.lon_max = float(max(lons))
        else:
            # Fallback bounds (Tokyo-ish)
            self.lat_min, self.lat_max = 35.0, 36.0
            self.lon_min, self.lon_max = 139.0, 140.0

    def _coarse_from_event(self, e: Event) -> str:
        if e.place_type in self.place_type_to_coarse:
            return self.place_type_to_coarse[e.place_type]
        return self.place_type_to_coarse.get(e.poi_id.split("#")[0].strip(), e.poi_id.split("#")[0].strip())

    def _coords(self, poi_id: str) -> Optional[Tuple[float, float]]:
        p = self.poi_by_id.get(poi_id)
        if p is None or p.latitude is None or p.longitude is None:
            return None
        return (p.latitude, p.longitude)

    def _step_distances_km(self, days: List[DayTrajectory]) -> List[float]:
        ds: List[float] = []
        for d in days:
            evs = list(d.events)
            for i in range(len(evs) - 1):
                ca = self._coords(evs[i].poi_id)
                cb = self._coords(evs[i + 1].poi_id)
                if ca is None or cb is None:
                    continue
                ds.append(haversine_km(ca[0], ca[1], cb[0], cb[1]))
        return ds

    def _step_intervals_min(self, days: List[DayTrajectory]) -> List[int]:
        gaps: List[int] = []
        for d in days:
            evs = list(d.events)
            for i in range(len(evs) - 1):
                gap = int(evs[i + 1].minute_of_day) - int(evs[i].minute_of_day)
                if gap > 0:
                    gaps.append(gap)
        return gaps

    def _hist_1d(self, xs: Sequence[float], *, bins: Sequence[float]) -> np.ndarray:
        xs = np.asarray(list(xs), dtype=float)
        xs = xs[np.isfinite(xs)]
        if xs.size == 0:
            return np.zeros(len(bins) - 1, dtype=float)
        h, _ = np.histogram(xs, bins=np.asarray(bins, dtype=float))
        return h.astype(float)

    def _dard_hist(self, days: List[DayTrajectory]) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
        tb = max(1, int(self.time_bin_minutes))
        cats = sorted(set(self.place_type_to_coarse.values()) | {self._coarse_from_event(e) for d in days for e in d.events})
        n_time = int(math.ceil(1440 / tb))
        keys: List[Tuple[int, str]] = [(t, c) for t in range(n_time) for c in cats]
        idx = {k: i for i, k in enumerate(keys)}
        hist = np.zeros(len(keys), dtype=float)
        for d in days:
            for e in d.events:
                tbin = int(e.minute_of_day // tb)
                c = self._coarse_from_event(e)
                k = (tbin, c)
                if k in idx:
                    hist[idx[k]] += 1.0
        return hist, keys

    def _stvd_hist(self, days: List[DayTrajectory]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        tb = max(1, int(self.time_bin_minutes))
        n_time = int(math.ceil(1440 / tb))
        lat_bins = max(5, self.stvd_lat_bins)
        lon_bins = max(5, self.stvd_lon_bins)

        # Avoid zero range
        lat_min, lat_max = self.lat_min, self.lat_max
        lon_min, lon_max = self.lon_min, self.lon_max
        if not (lat_max > lat_min):
            lat_max = lat_min + 1e-3
        if not (lon_max > lon_min):
            lon_max = lon_min + 1e-3

        hist = np.zeros((n_time, lat_bins, lon_bins), dtype=float)
        for d in days:
            for e in d.events:
                c = self._coords(e.poi_id)
                if c is None:
                    continue
                tbin = int(e.minute_of_day // tb)
                lat, lon = c
                # digitize into 0..bins-1
                li = int(np.floor((lat - lat_min) / (lat_max - lat_min) * lat_bins))
                lj = int(np.floor((lon - lon_min) / (lon_max - lon_min) * lon_bins))
                li = int(np.clip(li, 0, lat_bins - 1))
                lj = int(np.clip(lj, 0, lon_bins - 1))
                hist[tbin, li, lj] += 1.0

        return hist.reshape(-1), (n_time, lat_bins, lon_bins)

    def compute_spec_metrics(self, simulated_days: List[DayTrajectory], real_days: List[DayTrajectory]) -> Dict[str, Any]:
        # SD: step distances; use fixed bins
        sd_sim = self._step_distances_km(simulated_days)
        sd_real = self._step_distances_km(real_days)
        # 0..100km with finer bins at short distances
        sd_bins = np.concatenate([np.linspace(0, 5, 51), np.linspace(5, 50, 46), np.linspace(50, 100, 11)])
        sd_h_sim = self._hist_1d(sd_sim, bins=sd_bins)
        sd_h_real = self._hist_1d(sd_real, bins=sd_bins)
        sd_jsd = js_divergence(sd_h_sim, sd_h_real, eps=1e-9)

        # SI: step intervals in minutes; bins up to 12h
        si_sim = self._step_intervals_min(simulated_days)
        si_real = self._step_intervals_min(real_days)
        si_bins = np.concatenate([np.arange(0, 181, 5), np.arange(180, 721, 15), np.arange(720, 721 + 1, 1)])
        si_h_sim = self._hist_1d(si_sim, bins=si_bins)
        si_h_real = self._hist_1d(si_real, bins=si_bins)
        si_jsd = js_divergence(si_h_sim, si_h_real, eps=1e-9)

        # DARD
        dard_sim, dard_keys = self._dard_hist(simulated_days)
        dard_real, _keys2 = self._dard_hist(real_days)
        # Align by key order from union
        # We rebuild both on union keys for correct alignment
        union_cats = sorted(
            set(self.place_type_to_coarse.values())
            | {self._coarse_from_event(e) for d in simulated_days for e in d.events}
            | {self._coarse_from_event(e) for d in real_days for e in d.events}
        )
        tb = max(1, int(self.time_bin_minutes))
        n_time = int(math.ceil(1440 / tb))
        union_keys: List[Tuple[int, str]] = [(t, c) for t in range(n_time) for c in union_cats]
        idx_union = {k: i for i, k in enumerate(union_keys)}

        def rebuild(days: List[DayTrajectory]) -> np.ndarray:
            h = np.zeros(len(union_keys), dtype=float)
            for dd in days:
                for e in dd.events:
                    tbin = int(e.minute_of_day // tb)
                    c = self._coarse_from_event(e)
                    k = (tbin, c)
                    j = idx_union.get(k)
                    if j is not None:
                        h[j] += 1.0
            return h

        dard_sim_u = rebuild(simulated_days)
        dard_real_u = rebuild(real_days)
        dard_jsd = js_divergence(dard_sim_u, dard_real_u, eps=1e-9)

        # STVD
        stvd_sim, stvd_shape = self._stvd_hist(simulated_days)
        stvd_real, _shape2 = self._stvd_hist(real_days)
        stvd_jsd = js_divergence(stvd_sim, stvd_real, eps=1e-9)

        return {
            "SD_jsd": float(sd_jsd),
            "SI_jsd": float(si_jsd),
            "DARD_jsd": float(dard_jsd),
            "STVD_jsd": float(stvd_jsd),
            "SD_n_samples_sim": int(len(sd_sim)),
            "SD_n_samples_real": int(len(sd_real)),
            "SI_n_samples_sim": int(len(si_sim)),
            "SI_n_samples_real": int(len(si_real)),
            "DARD_dim": int(len(dard_sim_u)),
            "STVD_dim": int(len(stvd_sim)),
            "STVD_shape": {"time_bins": stvd_shape[0], "lat_bins": stvd_shape[1], "lon_bins": stvd_shape[2]},
        }

    # Legacy metrics kept for convenience/diagnostics
    def compute_legacy_metrics(self, simulated_days: List[DayTrajectory], real_days: List[DayTrajectory], topk: int = 10) -> Dict[str, Any]:
        # Stop count KL + mean error
        def stop_count_distribution(days: List[DayTrajectory]) -> Dict[int, int]:
            dist: Dict[int, int] = {}
            for d in days:
                n = len(d.events)
                dist[n] = dist.get(n, 0) + 1
            return dist

        def hist_from_counts(counts: Dict[int, int]) -> np.ndarray:
            if not counts:
                return np.array([1.0])
            mx = max(counts.keys())
            h = np.zeros(mx + 1, dtype=float)
            for k, v in counts.items():
                if k >= 0:
                    h[k] = float(v)
            return h

        sim_stop = stop_count_distribution(simulated_days)
        real_stop = stop_count_distribution(real_days)
        sim_hist = hist_from_counts(sim_stop)
        real_hist = hist_from_counts(real_stop)
        n = max(sim_hist.size, real_hist.size)
        sim_hist = np.pad(sim_hist, (0, n - sim_hist.size))
        real_hist = np.pad(real_hist, (0, n - real_hist.size))
        stop_kl = kl_divergence(sim_hist, real_hist, eps=1e-9)
        sim_mean = float(np.average(np.arange(n), weights=(sim_hist + 1e-9)))
        real_mean = float(np.average(np.arange(n), weights=(real_hist + 1e-9)))
        stop_mean_abs_error = abs(sim_mean - real_mean)

        # Distance Wasserstein
        sim_ds = self._step_distances_km(simulated_days)
        real_ds = self._step_distances_km(real_days)
        w1 = wasserstein_1d(sim_ds, real_ds)

        # topK POI recall
        def topk_pois(days: List[DayTrajectory], k: int) -> Dict[str, List[str]]:
            by_agent: Dict[str, Dict[str, int]] = {}
            for d in days:
                m = by_agent.setdefault(d.agent_id, {})
                for e in d.events:
                    m[e.poi_id] = m.get(e.poi_id, 0) + 1
            out: Dict[str, List[str]] = {}
            for aid, counts in by_agent.items():
                items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                out[aid] = [poi for poi, _ in items[:k]]
            return out

        real_topk = topk_pois(real_days, k=topk)
        sim_seen: Dict[str, Set[str]] = {}
        for d in simulated_days:
            s = sim_seen.setdefault(d.agent_id, set())
            for e in d.events:
                s.add(e.poi_id)
        recalls: List[float] = []
        for aid, top_list in real_topk.items():
            if not top_list:
                continue
            simset = sim_seen.get(aid, set())
            hits = sum(1 for p in top_list if p in simset)
            recalls.append(hits / float(len(top_list)))
        topk_recall = float(np.mean(recalls)) if recalls else 0.0

        return {
            "legacy_daily_stop_count_distribution_kl": float(stop_kl),
            "legacy_daily_stop_count_mean_abs_error": float(stop_mean_abs_error),
            "legacy_trip_distance_distribution_wasserstein_km": float(w1) if np.isfinite(w1) else float("nan"),
            "legacy_topk_poi_recall": float(topk_recall),
        }


# -----------------------------
# Calibration (kept; optional)
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
        topk_recall_k: int = 10,
        max_train_days_total: int = 2000,
    ) -> None:
        self.n_iterations = int(n_iterations)
        self.n_runs_per_eval = int(n_runs_per_eval)
        self.base_seed = int(base_seed)
        self.topk_recall_k = int(topk_recall_k)
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
        # Primary: sum of spec JSDs
        sd = float(m.get("SD_jsd", 0.0))
        si = float(m.get("SI_jsd", 0.0))
        dard = float(m.get("DARD_jsd", 0.0))
        stvd = float(m.get("STVD_jsd", 0.0))
        return float(sd + si + dard + stvd)

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
                time_bin_minutes=evaluator.time_bin_minutes,
            )
            sim_runs, _contacts = simulator.rollout(validation_split=capped_by_agent, n_runs=self.n_runs_per_eval)

            run_metrics: List[Dict[str, Any]] = []
            for _run_idx, sim_days in sim_runs.items():
                m = evaluator.compute_spec_metrics(simulated_days=sim_days, real_days=all_train_days)
                run_metrics.append(m)

            keys = ["SD_jsd", "SI_jsd", "DARD_jsd", "STVD_jsd"]
            mean_m: Dict[str, Any] = {}
            for k in keys:
                vals = [float(m.get(k, float("nan"))) for m in run_metrics]
                vals = [v for v in vals if np.isfinite(v)]
                mean_m[k] = float(np.mean(vals)) if vals else float("nan")

            loss = self._loss_from_metrics(mean_m)
            if loss < best_loss:
                best_loss = loss
                best_params = cand
                best_metrics = mean_m

        if best_metrics is not None:
            LOGGER.warning("Calibration done. Best training metrics (mean): %s", json.dumps(best_metrics, ensure_ascii=False))
        return best_params


# -----------------------------
# Pattern/Persona/Motivation helpers
# -----------------------------
def summarize_history_stats(days: List[DayTrajectory], place_type_to_coarse: Dict[str, str], max_days: int = 60) -> str:
    days_sorted = sorted(days, key=lambda x: x.d)[-max_days:]
    n_days = len(days_sorted)
    if n_days == 0:
        return "No history."

    stop_counts = [len(d.events) for d in days_sorted]
    mean_stops = float(np.mean(stop_counts)) if stop_counts else 0.0
    med_stops = float(np.median(stop_counts)) if stop_counts else 0.0

    start_times = [d.events[0].minute_of_day for d in days_sorted if d.events]
    mean_start = float(np.mean(start_times)) if start_times else float("nan")

    coarse_counts: Dict[str, int] = {}
    for d in days_sorted:
        for e in d.events:
            pt = e.place_type
            coarse = place_type_to_coarse.get(pt, place_type_to_coarse.get(e.poi_id.split("#")[0].strip(), pt))
            coarse_counts[coarse] = coarse_counts.get(coarse, 0) + 1
    top_coarse = sorted(coarse_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    lines = [
        f"days={n_days}",
        f"mean_stops={mean_stops:.2f}",
        f"median_stops={med_stops:.1f}",
    ]
    if np.isfinite(mean_start):
        lines.append(f"mean_first_activity_time={hms_from_minute(int(round(mean_start)))}")
    if top_coarse:
        lines.append("top_categories=" + ", ".join([f"{k}({v})" for k, v in top_coarse]))
    return "\n".join(lines)


def build_recent_window(all_days: List[DayTrajectory], target: date, window_days: int = 7) -> List[DayTrajectory]:
    lo = target - timedelta(days=window_days)
    return [d for d in all_days if lo <= d.d < target]


# -----------------------------
# LLM-driven orchestration: generate daily strings and parse back
# -----------------------------
class LLMDrivenSimulator:
    def __init__(
        self,
        draft_simulator: MobilitySimulator,
        llm_generator: LLMTrajectoryGenerator,
        pattern_extractor: PatternExtractor,
        persona_inferer: PersonaInferer,
        motivation_summarizer: MotivationSummarizer,
        place_type_to_coarse: Dict[str, str],
        *,
        pad_hour: bool = True,
    ) -> None:
        self.draft_simulator = draft_simulator
        self.llm_generator = llm_generator
        self.pattern_extractor = pattern_extractor
        self.persona_inferer = persona_inferer
        self.motivation_summarizer = motivation_summarizer
        self.place_type_to_coarse = place_type_to_coarse
        self.pad_hour = pad_hour

        self._pattern_cache: Dict[str, Pattern] = {}
        self._persona_cache: Dict[str, Persona] = {}

    def prepare_agent(self, agent_id: str, train_days: List[DayTrajectory]) -> None:
        if agent_id not in self._pattern_cache:
            self._pattern_cache[agent_id] = self.pattern_extractor.derive(agent_id, train_days, self.place_type_to_coarse)
        if agent_id not in self._persona_cache:
            self._persona_cache[agent_id] = self.persona_inferer.infer(agent_id, train_days, self.place_type_to_coarse)

    def generate_day(
        self,
        agent_id: str,
        target_date: date,
        rng: np.random.Generator,
        all_known_days_for_motivation: List[DayTrajectory],
    ) -> Tuple[str, DayTrajectory]:
        # Draft simulation
        draft_day, _intervals = self.draft_simulator.simulate_day(agent_id=agent_id, d=target_date, rng=rng, return_intervals=False)
        draft_str = trajectory_to_activity_string(draft_day, pad_hour=self.pad_hour)

        pattern = self._pattern_cache.get(agent_id, Pattern("Pattern: Unknown"))
        persona = self._persona_cache.get(agent_id, Persona("Persona: Unknown"))
        recent = build_recent_window(all_known_days_for_motivation, target_date, window_days=7)
        motivation = self.motivation_summarizer.summarize(agent_id, target_date, recent, self.place_type_to_coarse)

        allowed = None
        prof = self.draft_simulator.agent_profiles.get(agent_id)
        if prof and prof.home_poi_candidates:
            allowed = list(prof.home_poi_candidates.keys()) + list(self.draft_simulator.global_model.poi_ids[:100])

        out_str = self.llm_generator.generate(
            agent_id=agent_id,
            target_date=target_date,
            pattern=pattern,
            persona=persona,
            motivation=motivation,
            allowed_pois_hint=allowed,
            draft_trajectory_string=draft_str,
        )

        # Parse LLM output back to DayTrajectory for evaluation
        try:
            parsed = parse_trajectories({agent_id: [out_str]}, strict=False)
            # Should yield exactly one day
            day = max(parsed, key=lambda x: x.d)
        except Exception:
            # Fallback to draft
            day = draft_day
            out_str = draft_str

        # Ensure correct date and ordering
        if day.d != target_date:
            day = DayTrajectory(agent_id=agent_id, d=target_date, events=day.events)
            out_str = trajectory_to_activity_string(day, pad_hour=self.pad_hour)

        return out_str, day

    def rollout(
        self,
        train_map: Dict[str, List[DayTrajectory]],
        validation_map: Dict[str, List[DayTrajectory]],
        n_runs: int,
        base_seed: int,
    ) -> Tuple[Dict[int, List[DayTrajectory]], Dict[int, Dict[str, List[str]]]]:
        """
        Returns:
          - run_idx -> simulated DayTrajectory list
          - run_idx -> {agent_id -> list of daily activity strings (1921Y format)}
        """
        results: Dict[int, List[DayTrajectory]] = {}
        strings_out: Dict[int, Dict[str, List[str]]] = {}

        # Build per-agent "all known" history for motivation using train only (no validation leakage from labels).
        all_known_by_agent: Dict[str, List[DayTrajectory]] = {aid: list(days) for aid, days in train_map.items()}

        for run in range(n_runs):
            rng = np.random.default_rng(base_seed + 2020 + run * 997)
            days_out: List[DayTrajectory] = []
            strings_out_run: Dict[str, List[str]] = {aid: [] for aid in validation_map.keys()}

            for agent_id, val_days in validation_map.items():
                self.prepare_agent(agent_id, train_map.get(agent_id, []))
                for day in sorted(val_days, key=lambda x: x.d):
                    s, traj = self.generate_day(
                        agent_id=agent_id,
                        target_date=day.d,
                        rng=rng,
                        all_known_days_for_motivation=all_known_by_agent.get(agent_id, []),
                    )
                    days_out.append(traj)
                    strings_out_run.setdefault(agent_id, []).append(s)
                    # Update history with generated day to enable rolling motivation
                    all_known_by_agent.setdefault(agent_id, []).append(traj)

            results[run] = days_out
            strings_out[run] = strings_out_run

        return results, strings_out


# -----------------------------
# Results persistence
# -----------------------------
def save_results(
    output_dir: str,
    calibrated_params: CalibratedParameters,
    evaluation_results: Dict[str, Any],
    simulated_rollouts: Dict[int, List[DayTrajectory]],
    generated_strings_by_run: Optional[Dict[int, Dict[str, List[str]]]] = None,
    contacts_by_run: Optional[Dict[int, List[ContactEvent]]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    params_path = os.path.join(output_dir, "calibrated_parameters.json")
    metrics_path = os.path.join(output_dir, "evaluation_results_on_validation.json")
    sim_path = os.path.join(output_dir, "simulated_validation_trajectories.json")
    gen_path = os.path.join(output_dir, "generated_1921Y_format_by_run.json")
    contact_path = os.path.join(output_dir, "simulated_contacts_by_run.json")

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
                }
            )
        sim_out[str(run_idx)] = run_list

    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(sim_out, f, indent=2, ensure_ascii=False)

    if generated_strings_by_run is not None:
        out2: Dict[str, Any] = {str(k): v for k, v in generated_strings_by_run.items()}
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(out2, f, indent=2, ensure_ascii=False)

    if contacts_by_run is not None:
        out3: Dict[str, Any] = {}
        for run_idx, contacts in contacts_by_run.items():
            out3[str(run_idx)] = [
                {
                    "day": c.day.isoformat(),
                    "time_bin": c.time_bin,
                    "poi_id": c.poi_id,
                    "agent_a": c.agent_a,
                    "agent_b": c.agent_b,
                }
                for c in contacts
            ]
        with open(contact_path, "w", encoding="utf-8") as f:
            json.dump(out3, f, indent=2, ensure_ascii=False)


# -----------------------------
# CLI and Orchestration
# -----------------------------
def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-driven daily mobility simulator with calibration and spec metrics evaluation.")
    p.add_argument("--seed", type=int, default=12345, help="Global random seed for deterministic runs.")
    p.add_argument("--verbosity", type=int, default=0, help="0=warn,1=info,2=debug")

    # Data paths (no import-time env validation)
    p.add_argument("--project-root", type=str, default=os.environ.get("PROJECT_ROOT", ""), help="Project root; default from $PROJECT_ROOT")
    p.add_argument("--data-path", type=str, default=os.environ.get("DATA_PATH", ""), help="Relative data path; default from $DATA_PATH")
    p.add_argument("--data-dir", type=str, default="", help="Explicit data dir (overrides --project-root/--data-path)")
    p.add_argument("--poi-catalog-filename", type=str, default="", help="Optional POI catalog filename (auto-detect otherwise).")
    p.add_argument("--strict-parse", action="store_true", help="Fail on parse errors instead of skipping bad tokens/days.")

    # Experimental setting
    p.add_argument(
        "--setting",
        type=str,
        default="all-all",
        choices=["all-all", "2019-2019", "2021-2021", "2019-2021"],
        help="Train/Eval year regimes: '2019-2021' means train on 2019, evaluate on 2021 (OOD).",
    )

    # Calibration and evaluation
    p.add_argument("--skip-calibration", action="store_true", help="Skip random-search calibration (use defaults).")
    p.add_argument("--calib-iters", type=int, default=30, help="Random search iterations for calibration.")
    p.add_argument("--calib-runs-per-eval", type=int, default=3, help="Stochastic runs per parameter evaluation.")
    p.add_argument("--eval-runs", type=int, default=10, help="Number of stochastic runs for validation rollout.")
    p.add_argument("--max-train-days-total", type=int, default=2000, help="Cap training days used during calibration.")

    # Spec metric settings
    p.add_argument("--time-bin-minutes", type=int, default=10, help="Time bin size for DARD/STVD and contact computation.")
    p.add_argument("--stvd-lat-bins", type=int, default=50, help="Latitude bins for STVD.")
    p.add_argument("--stvd-lon-bins", type=int, default=50, help="Longitude bins for STVD.")

    # LLM options
    p.add_argument("--llm-backend", type=str, default="heuristic", choices=["heuristic"], help="LLM backend (heuristic is offline fallback).")
    p.add_argument("--llm-cache-dir", type=str, default="", help="Optional cache dir for LLM calls.")
    p.add_argument("--llm-rate-limit-qps", type=float, default=0.0, help="Optional rate limit for LLM calls.")
    p.add_argument("--pad-hour", action="store_true", help="Output HH:MM:SS with 2-digit hours (default).")
    p.set_defaults(pad_hour=True)

    # Contacts
    p.add_argument("--compute-contacts", action="store_true", help="Compute co-location contact events for simulated trajectories.")
    p.add_argument("--max-contacts-per-run", type=int, default=200000, help="Max contact events stored per run.")

    # Output
    p.add_argument("--output-dir", type=str, default="", help="Output directory (default: <data_dir>/outputs).")
    return p.parse_args(argv)


def resolve_data_dir(args: argparse.Namespace) -> str:
    if args.data_dir:
        return args.data_dir
    project_root = args.project_root
    data_path = args.data_path
    if not project_root or not data_path:
        raise EnvironmentError(
            "You must provide data path via --data-dir or both --project-root and --data-path "
            "(or set env vars PROJECT_ROOT and DATA_PATH)."
        )
    return os.path.join(project_root, data_path)


def years_for_setting(setting: str) -> Tuple[Optional[Set[int]], Optional[Set[int]]]:
    if setting == "2019-2019":
        return {2019}, {2019}
    if setting == "2021-2021":
        return {2021}, {2021}
    if setting == "2019-2021":
        return {2019}, {2021}
    return None, None  # all-all


def build_llm(args: argparse.Namespace) -> LLMClient:
    if args.llm_backend == "heuristic":
        backend: LLMClient = HeuristicLLMClient()
    else:
        backend = HeuristicLLMClient()

    cache = FileCache(args.llm_cache_dir) if args.llm_cache_dir else None
    return CachedLLMClient(backend=backend, cache=cache, rate_limit_qps=float(args.llm_rate_limit_qps))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_cli(argv)
    setup_logging(args.verbosity)
    set_global_seed(args.seed)

    data_dir = resolve_data_dir(args)
    if not args.output_dir:
        args.output_dir = os.path.join(data_dir, "outputs")

    data = load_data(data_dir, poi_catalog_filename=(args.poi_catalog_filename or None))
    place_type_to_coarse = extract_place_type_to_coarse(data["catto"])

    trajectories_all = parse_trajectories(data["y"], strict=bool(args.strict_parse))

    train_years, eval_years = years_for_setting(args.setting)
    traj_train_pool = filter_days_by_years(trajectories_all, train_years)
    traj_eval_pool = filter_days_by_years(trajectories_all, eval_years)

    # If using OOD setting, splitting must be done separately by years.
    # Train split from train_pool; validation set from eval_pool (no per-agent temporal holdout across years).
    if args.setting == "2019-2021":
        split_train = holdout_split(traj_train_pool)
        train_map, _val_unused, train_flat, _val_unused_flat = split_train_validation_maps(split_train)

        # Validation is *all* eval_year days for those agents
        by_agent_eval: Dict[str, List[DayTrajectory]] = {}
        for t in traj_eval_pool:
            by_agent_eval.setdefault(t.agent_id, []).append(t)
        val_map = {aid: sorted(days, key=lambda x: x.d) for aid, days in by_agent_eval.items()}
        val_flat = [d for days in val_map.values() for d in days]
    else:
        # Standard temporal split within chosen pool (or all-all)
        split = holdout_split(traj_eval_pool if eval_years else trajectories_all)
        train_map, val_map, train_flat, val_flat = split_train_validation_maps(split)
        # If train years restricted (e.g., 2019-2019), apply it to training
        if train_years:
            train_map = {aid: [d for d in days if d.d.year in train_years] for aid, days in train_map.items()}
            train_flat = [d for days in train_map.values() for d in days]

    if not train_flat:
        raise ValueError("Training set is empty after applying setting/split. Cannot fit model.")
    if not val_flat:
        LOGGER.warning("Validation set is empty after split/setting. Evaluation will be empty.")

    poi_by_id, pois_by_coarse, coarse_categories = build_poi_catalog(data["poi_catalog"], place_type_to_coarse)

    fitter = ModelFitter(
        poi_by_id=poi_by_id,
        pois_by_coarse_category=pois_by_coarse,
        coarse_categories=coarse_categories,
        place_type_to_coarse=place_type_to_coarse,
    )
    agent_profiles, global_model = fitter.fit(train_map)

    evaluator = Evaluator(
        poi_by_id=poi_by_id,
        place_type_to_coarse=place_type_to_coarse,
        time_bin_minutes=int(args.time_bin_minutes),
        stvd_lat_bins=int(args.stvd_lat_bins),
        stvd_lon_bins=int(args.stvd_lon_bins),
    )

    params = CalibratedParameters()
    if not args.skip_calibration and train_flat:
        calibrator = RandomSearchCalibrator(
            n_iterations=args.calib_iters,
            n_runs_per_eval=args.calib_runs_per_eval,
            base_seed=args.seed,
            topk_recall_k=10,
            max_train_days_total=args.max_train_days_total,
        )
        params = calibrator.fit(
            global_model=global_model,
            agent_profiles=agent_profiles,
            place_type_to_coarse=place_type_to_coarse,
            train_split=train_map,
            evaluator=evaluator,
            initial_params=params,
        )

    # Draft simulator (statistical), used as a scaffold for LLM
    draft_sim = MobilitySimulator(
        global_model=global_model,
        agent_profiles=agent_profiles,
        place_type_to_coarse=place_type_to_coarse,
        params=params,
        base_seed=args.seed,
        time_bin_minutes=int(args.time_bin_minutes),
    )

    # LLM pipeline
    llm = build_llm(args)
    llm_sim = LLMDrivenSimulator(
        draft_simulator=draft_sim,
        llm_generator=LLMTrajectoryGenerator(llm),
        pattern_extractor=PatternExtractor(llm),
        persona_inferer=PersonaInferer(llm),
        motivation_summarizer=MotivationSummarizer(llm),
        place_type_to_coarse=place_type_to_coarse,
        pad_hour=bool(args.pad_hour),
    )

    simulated_rollouts, generated_strings_by_run = llm_sim.rollout(
        train_map=train_map,
        validation_map=val_map,
        n_runs=int(args.eval_runs),
        base_seed=int(args.seed),
    )

    # Optionally compute contacts using the draft simulator intervals (not LLM output)
    contacts_by_run: Optional[Dict[int, List[ContactEvent]]] = None
    if args.compute_contacts:
        # Re-simulate with intervals for contact computation to avoid trying to infer dwell from LLM string
        sim2, contacts = draft_sim.rollout(
            validation_split=val_map,
            n_runs=int(args.eval_runs),
            compute_contacts=True,
            max_contacts_per_run=int(args.max_contacts_per_run),
        )
        # Keep contacts but keep trajectory rollouts from LLM
        contacts_by_run = contacts

    # Evaluate across runs
    per_run: Dict[str, Dict[str, Any]] = {}
    for run_idx, sim_days in simulated_rollouts.items():
        spec_m = evaluator.compute_spec_metrics(simulated_days=sim_days, real_days=val_flat)
        legacy_m = evaluator.compute_legacy_metrics(simulated_days=sim_days, real_days=val_flat, topk=10)
        per_run[str(run_idx)] = {**spec_m, **legacy_m}

    # Aggregate spec metrics
    metric_keys = ["SD_jsd", "SI_jsd", "DARD_jsd", "STVD_jsd"]
    summary: Dict[str, Any] = {"per_run": per_run, "summary": {}}
    for k in metric_keys:
        vals = [float(per_run[str(i)].get(k, float("nan"))) for i in simulated_rollouts.keys()]
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
        "setting": args.setting,
        "n_agents_train": len([a for a, days in train_map.items() if days]),
        "n_agents_validation": len([a for a, days in val_map.items() if days]),
        "n_train_days": len(train_flat),
        "n_validation_days": len(val_flat),
        "data_dir": data_dir,
        "input_files": data["paths"],
        "time_bin_minutes": int(args.time_bin_minutes),
        "stvd_lat_bins": int(args.stvd_lat_bins),
        "stvd_lon_bins": int(args.stvd_lon_bins),
        "llm_backend": args.llm_backend,
    }

    save_results(
        output_dir=args.output_dir,
        calibrated_params=params,
        evaluation_results=summary,
        simulated_rollouts=simulated_rollouts,
        generated_strings_by_run=generated_strings_by_run,
        contacts_by_run=contacts_by_run,
    )

    sys.stdout.write(json.dumps(summary["summary"], indent=2) + "\n")
    sys.stdout.write(f"Saved outputs to: {args.output_dir}\n")



# Execute main for both direct execution and sandbox wrapper invocation
main()