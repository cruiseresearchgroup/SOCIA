import json
import math
import random
import re
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, TypedDict


# ----------------------------
# Utilities
# ----------------------------

def safe_read_json(path: Path) -> Optional[Any]:
    """
    Read JSON from path safely.
    Returns None on error and logs a message.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode error for {path}: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error reading {path}: {e}")
    return None


def write_json(path: Path, obj: Any) -> None:
    """
    Write an object to a JSON file with UTF-8 encoding.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Failed to write {path}: {e}")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """
    Write an iterable of JSON records to a JSONL file.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write {path}: {e}")


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    """
    Basic alphanumeric tokenizer with lowercase normalization.
    """
    return [t.lower() for t in _WORD_RE.findall(text or "")]


POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "like", "liked", "awesome", "fantastic",
    "perfect", "solid", "reliable", "durable", "beautiful", "sleek", "comfortable", "easy", "convenient",
    "worth", "valuable", "recommend", "highly", "happy", "impressed", "best", "nice", "positive"
}
NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "hate", "hated", "dislike", "disliked", "broken", "broke", "cheap",
    "expensive", "overpriced", "difficult", "hard", "confusing", "annoying", "buggy", "lag", "laggy", "slow",
    "flimsy", "issue", "issues", "problem", "problems", "worse", "worst", "disappointed", "negative"
}

PROFANITY_WORDS = {
    "damn", "shit", "crap", "hell", "sucks", "stupid", "idiot", "dumb", "suck", "bloody"
}


ASPECT_SYNONYMS: Dict[str, List[str]] = {
    "quality": ["quality", "build", "craftsmanship", "finish", "material"],
    "durability": ["durable", "sturdy", "rugged", "last", "lasting", "broke", "broken", "wear"],
    "performance": ["performance", "speed", "smooth", "fast", "slow", "framerate", "frame", "lag", "fps"],
    "value": ["value", "price", "worth", "expensive", "cheap", "overpriced"],
    "design": ["design", "look", "style", "aesthetic"],
    "usability": ["usability", "easy", "difficult", "install", "installation", "setup", "configure", "instruction", "instructions"],
    "comfort": ["comfort", "comfortable", "ergonomic", "grip", "feel"],
    "battery": ["battery", "charge", "charging", "life", "power"],
    "sound": ["sound", "audio", "noise", "mic", "microphone"],
    "fit": ["fit", "size", "fitment", "tight", "loose"],
    "packaging": ["packaging", "box", "sealed", "wrap"],
    "delivery": ["delivery", "shipping", "arrived", "ship", "prime"],
    "materials": ["plastic", "metal", "aluminum", "rubber", "steel"],
    "warranty": ["warranty", "support", "customer", "service"],
    "instructions": ["manual", "instructions", "guide", "documentation"],
}

# Precompute tokenized synonyms for efficient aspect detection
_ASPECT_SYNONYMS_TOKENS: Dict[str, List[Tuple[str, ...]]] = {
    a: [tuple(tokenize(s)) for s in syns] for a, syns in ASPECT_SYNONYMS.items()
}


def clean_profanity(text: str) -> Tuple[str, int]:
    """
    Replace profane words with a redacted form. Returns (cleaned_text, num_replacements).
    """
    tokens = text.split()
    count = 0
    cleaned_tokens: List[str] = []
    prof_set = {w.lower() for w in PROFANITY_WORDS}
    for w in tokens:
        wl = re.sub(r"[^\w']", "", w).lower()
        if wl in prof_set:
            count += 1
            if len(w) > 2:
                red = w[0] + "*" * (len(w) - 2) + w[-1]
            else:
                red = "*" * len(w)
            cleaned_tokens.append(red)
        else:
            cleaned_tokens.append(w)
    return " ".join(cleaned_tokens), count


def detect_aspects(text: str, aspect_vocab: List[str]) -> List[str]:
    """
    Detect aspects by matching tokenized synonyms and phrases.
    Optimized by precomputed synonym tokens.
    """
    tokens = tokenize(text)
    token_set = set(tokens)
    found: set = set()
    for aspect in aspect_vocab:
        syn_tokens_list = _ASPECT_SYNONYMS_TOKENS.get(aspect, [(a,) for a in [aspect]])
        for s_tokens in syn_tokens_list:
            if len(s_tokens) == 1:
                if s_tokens[0] in token_set:
                    found.add(aspect)
                    break
            else:
                L = len(s_tokens)
                for i in range(len(tokens) - L + 1):
                    if tuple(tokens[i:i + L]) == s_tokens:
                        found.add(aspect)
                        break
                if aspect in found:
                    break
    return list(found)


def sentiment_score(text: str) -> float:
    """
    Compute a crude sentiment score in [-1, 1] from token lexicons and punctuation.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    excl = text.count("!")
    raw = pos - neg + 0.2 * excl
    norm = raw / (len(tokens) ** 0.6 + 1e-6)
    return max(-1.0, min(1.0, norm))


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    """
    Jaccard similarity between two iterables of strings.
    """
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def cosine_similarity_counter(a: Dict[str, int], b: Dict[str, int]) -> float:
    """
    Cosine similarity between two bag-of-words counters.
    """
    if not a and not b:
        return 1.0
    num = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
    den1 = math.sqrt(sum(v * v for v in a.values()))
    den2 = math.sqrt(sum(v * v for v in b.values()))
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)


def bow_counts(tokens: List[str]) -> Dict[str, int]:
    """
    Construct a bag-of-words counter.
    """
    d: Dict[str, int] = {}
    for t in tokens:
        d[t] = d.get(t, 0) + 1
    return d


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


# ----------------------------
# Typed payloads
# ----------------------------

class ObjectiveWeightsDict(TypedDict):
    stars: float
    text: float
    consistency: float


class PersonaDict(TypedDict, total=False):
    user_id: str
    baseline_leniency: float
    verbosity_prior: float
    style: str
    aspect_weights: Dict[str, float]
    recent_sentiment_bias: float


class ItemProfileDict(TypedDict, total=False):
    item_id: str
    quality_prior: float
    controversy: float
    aspect_summary: Dict[str, float]
    freshness_score: float


class PlanDict(TypedDict, total=False):
    planned_aspects: List[str]
    tone_target: str
    tone_value: float
    length_target: int


class MetricsDict(TypedDict, total=False):
    MAE_stars: float
    RMSE_contrib: float
    Text_Similarity: float
    Aspect_Coverage: float
    Sentiment_Agreement: float
    QA_Consistency: float
    Length_Deviation: float
    Review_Generation: float


# ----------------------------
# Config and Parameters
# ----------------------------

@dataclass
class ObjectiveWeights:
    stars: float = 0.6
    text: float = 0.35
    consistency: float = 0.05

    def normalize(self) -> "ObjectiveWeights":
        s = self.stars + self.text + self.consistency
        if s == 0:
            return ObjectiveWeights(0.6, 0.35, 0.05)
        return ObjectiveWeights(self.stars / s, self.text / s, self.consistency / s)


@dataclass
class Parameters:
    """
    Global parameter bundle for the multi-agent simulator.
    """
    # Planning/Composition
    neighbor_weight: float = 0.0
    ctx_merge_weight: float = 0.5
    aspect_topk: int = 4
    length_target_mean: int = 4
    plan_diversity_temp: float = 0.7

    # Authoring
    llm_temperature: float = 0.5
    style_alignment_weight: float = 0.7
    max_revision_loops: int = 1

    # Rating
    mapping_slope: float = 4.0
    mapping_intercept: float = 0.0
    user_bias_weight: float = 0.5
    item_bias_weight: float = 0.4
    uncertainty_scale: float = 0.3

    # QA
    consistency_threshold: float = 0.75
    max_auto_fix_attempts: int = 1

    # Objective
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["objective_weights"] = asdict(self.objective_weights.normalize())
        return d


# ----------------------------
# Data Indexer
# ----------------------------

class DataIndexer:
    """
    Memory and indexing agent for user/item histories and priors.
    Loads data, builds lookup indices, and computes priors.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.aspect_vocab = [
            "quality", "durability", "performance", "value", "design", "usability",
            "comfort", "battery", "sound", "fit", "packaging", "delivery",
            "materials", "warranty", "instructions"
        ]
        self.platform_policy = {
            "source": "amazon",
            "type": "product",
            "style_constraints": {
                "length_sentences_min": 2,
                "length_sentences_max": 6,
                "no_profanity": True,
            }
        }

        self.train_records: List[Dict[str, Any]] = []
        self.test_records: List[Dict[str, Any]] = []
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.item_metadata: Dict[str, Dict[str, Any]] = {}
        self.historical_reviews: List[Dict[str, Any]] = []

        self.user_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.item_histories: Dict[str, List[Dict[str, Any]]] = {}

        self.global_mean: float = 3.5
        self.user_means: Dict[str, float] = {}
        self.item_means: Dict[str, float] = {}

        self._load_all()

    def _coerce_record(self, rec_id: str, rec: Dict[str, Any], split: str) -> Dict[str, Any]:
        """
        Coerce a raw record into a normalized schema with safe types.
        Missing stars -> None (excluded from priors).
        """
        star_val = rec.get("stars")
        stars = float(star_val) if star_val is not None else None
        return {
            "rec_id": rec_id,
            "type": rec.get("type", "user_behavior_simulation"),
            "user_id": rec.get("user_id", ""),
            "item_id": rec.get("item_id", ""),
            "stars": stars,
            "review": rec.get("review", "") or "",
            "datatype": split,
        }

    def _load_interactions(self, filename: str, split: str) -> List[Dict[str, Any]]:
        path = self.base_dir / filename
        data = safe_read_json(path)
        res: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            for k, v in data.items():
                res.append(self._coerce_record(k, v, split))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                res.append(self._coerce_record(str(i), v, split))
        elif data is None:
            pass
        else:
            print(f"[WARN] Unexpected format for {filename}")
        return res

    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """
        Load user profiles, filtering to Amazon source if available.
        Also parse friends into a list for neighbor modeling.
        """
        path = self.base_dir / "user_sample.json"
        data = safe_read_json(path)
        res: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, list):
            for u in data:
                if u.get("source") and u.get("source") != "amazon":
                    # Skip non-Amazon profiles to avoid mismatch
                    continue
                uid = str(u.get("user_id", ""))
                if uid:
                    friends_raw = u.get("friends", "")
                    friends_list = []
                    if isinstance(friends_raw, str):
                        friends_list = [f.strip() for f in friends_raw.split(",") if f.strip()]
                    u = dict(u)
                    u["friends_list"] = friends_list
                    res[uid] = u
        return res

    def _load_items(self) -> Dict[str, Dict[str, Any]]:
        """
        Load item metadata, filtering to Amazon source if available.
        """
        path = self.base_dir / "item_sample.json"
        data = safe_read_json(path)
        res: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, list):
            for it in data:
                if it.get("source") and it.get("source") != "amazon":
                    continue
                iid = str(it.get("item_id", ""))
                if iid:
                    res[iid] = it
        return res

    def _load_reviews(self) -> List[Dict[str, Any]]:
        """
        Load additional historical reviews to enrich memory agent.
        Only ingest Amazon product reviews to avoid platform leakage.
        """
        path = self.base_dir / "review_sample.json"
        data = safe_read_json(path)
        res: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for r in data:
                if r.get("source") == "amazon" and r.get("type") == "product":
                    # harmonize fields
                    res.append({
                        "user_id": r.get("user_id", ""),
                        "item_id": r.get("item_id", ""),
                        "stars": float(r["stars"]) if r.get("stars") is not None else None,
                        "review": r.get("text", "") or "",
                        "datatype": "historical"
                    })
        return res

    def _build_indices(self) -> None:
        """
        Build user and item histories using training interactions and optional historical reviews.
        """
        self.user_histories = {}
        self.item_histories = {}
        def add_record(rec: Dict[str, Any]) -> None:
            uid = rec.get("user_id", "")
            iid = rec.get("item_id", "")
            if uid:
                self.user_histories.setdefault(uid, []).append(rec)
            if iid:
                self.item_histories.setdefault(iid, []).append(rec)

        for rec in self.train_records:
            add_record(rec)
        for rec in self.historical_reviews:
            add_record(rec)

    def _compute_priors(self) -> None:
        """
        Compute global, user, and item priors from train split only.
        """
        stars = [r["stars"] for r in self.train_records if r.get("stars") is not None]
        if stars:
            self.global_mean = statistics.mean(stars)
        else:
            self.global_mean = 3.5
        self.user_means = {}
        for uid, hist in self.user_histories.items():
            u_stars = [r["stars"] for r in hist if r.get("stars") is not None and r.get("datatype") == "train"]
            if u_stars:
                self.user_means[uid] = statistics.mean(u_stars)
        self.item_means = {}
        for iid, hist in self.item_histories.items():
            i_stars = [r["stars"] for r in hist if r.get("stars") is not None and r.get("datatype") == "train"]
            if i_stars:
                self.item_means[iid] = statistics.mean(i_stars)

    def _load_all(self) -> None:
        self.train_records = self._load_interactions("amazon_train_sample.json", "train")
        self.test_records = self._load_interactions("amazon_test_sample.json", "test")
        self.user_profiles = self._load_users()
        self.item_metadata = self._load_items()
        self.historical_reviews = self._load_reviews()
        self._build_indices()
        self._compute_priors()
        print(f"[INFO] Loaded {len(self.train_records)} train and {len(self.test_records)} test records.")
        print(f"[INFO] Global mean stars (train): {self.global_mean:.3f}")

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return self.user_profiles.get(user_id, {})

    def get_item_metadata(self, item_id: str) -> Dict[str, Any]:
        return self.item_metadata.get(item_id, {})

    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        return self.user_histories.get(user_id, [])

    def get_item_history(self, item_id: str) -> List[Dict[str, Any]]:
        return self.item_histories.get(item_id, [])

    def get_user_mean(self, user_id: str) -> Optional[float]:
        return self.user_means.get(user_id)

    def get_item_mean(self, item_id: str) -> Optional[float]:
        return self.item_means.get(item_id)

    def get_global_mean(self) -> float:
        return self.global_mean

    def get_train(self) -> List[Dict[str, Any]]:
        return list(self.train_records)

    def get_test(self) -> List[Dict[str, Any]]:
        return list(self.test_records)

    def get_aspect_vocab(self) -> List[str]:
        return list(self.aspect_vocab)

    def get_platform_policy(self) -> Dict[str, Any]:
        return dict(self.platform_policy)

    def has_social_graph(self) -> bool:
        """
        Returns True if any profile has non-empty friends list.
        """
        for u in self.user_profiles.values():
            if u.get("friends_list"):
                return True
        return False


# ----------------------------
# Persona Profiler
# ----------------------------

class PersonaProfiler:
    """
    Infers a user's persona from history and optional neighbor influence.
    """

    def __init__(self, indexer: DataIndexer, params: Parameters):
        self.indexer = indexer
        self.params = params

    def _neighbor_influence(self, user_id: str) -> Tuple[float, Dict[str, float]]:
        """
        Compute neighbor-averaged leniency and aspect weights if social graph exists.
        """
        neighbor_weight = clamp(self.params.neighbor_weight, 0.0, 0.5)
        if neighbor_weight <= 0.0:
            return 0.0, {}
        profile = self.indexer.get_user_profile(user_id)
        friends: List[str] = profile.get("friends_list", [])
        if not friends:
            return 0.0, {}
        aspect_vocab = self.indexer.get_aspect_vocab()
        # Average neighbors' leniency and aspect preferences
        n = 0
        leniencies: List[float] = []
        aspect_counts: Dict[str, int] = {a: 0 for a in aspect_vocab}
        for fid in friends[:50]:  # limit to 50 to avoid heavy loops
            hist = self.indexer.get_user_history(fid)
            if not hist:
                continue
            u_mean = self.indexer.get_user_mean(fid)
            if u_mean is not None:
                leniencies.append(u_mean - self.indexer.get_global_mean())
            for rec in hist:
                aspects = detect_aspects(rec.get("review", ""), aspect_vocab)
                for a in aspects:
                    aspect_counts[a] += 1
            n += 1
        if n == 0:
            return 0.0, {}
        leniency_avg = statistics.mean(leniencies) if leniencies else 0.0
        total = sum(aspect_counts.values()) or 1
        aspect_weights = {a: aspect_counts[a] / total for a in aspect_vocab}
        return leniency_avg * neighbor_weight, {a: w * neighbor_weight for a, w in aspect_weights.items()}

    def profile(self, user_id: str) -> PersonaDict:
        history = self.indexer.get_user_history(user_id)
        user_profile = self.indexer.get_user_profile(user_id)
        u_mean = self.indexer.get_user_mean(user_id) or self.indexer.get_global_mean()
        leniency = u_mean - self.indexer.get_global_mean()

        # Neighbor influence
        n_leniency, n_aspect = self._neighbor_influence(user_id)

        review_count = user_profile.get("review_count", len(history))
        fans = user_profile.get("fans", 0)
        verbosity_prior = (review_count / (review_count + 10)) * 0.5 + (min(fans, 20) / 20.0) * 0.5
        funny = user_profile.get("funny", 0)
        cool = user_profile.get("cool", 0)
        style = "informative"
        if funny >= 20:
            style = "humorous"
        elif cool >= 20:
            style = "enthusiastic"

        # Aspect preferences inferred from history
        aspect_vocab = self.indexer.get_aspect_vocab()
        counts: Dict[str, int] = {a: 0 for a in aspect_vocab}
        for rec in history:
            aspects = detect_aspects(rec.get("review", ""), aspect_vocab)
            for a in aspects:
                counts[a] += 1
        total = sum(counts.values()) or 1
        aspect_weights = {a: counts[a] / total for a in aspect_vocab}

        # Blend with neighbor info
        if n_aspect:
            for a in aspect_vocab:
                aspect_weights[a] = clamp(aspect_weights.get(a, 0.0) * (1 - self.params.neighbor_weight) + n_aspect.get(a, 0.0), 0.0, 1.0)
        leniency += n_leniency

        persona: PersonaDict = {
            "user_id": user_id,
            "baseline_leniency": leniency,
            "verbosity_prior": verbosity_prior,
            "style": style,
            "aspect_weights": aspect_weights,
            "recent_sentiment_bias": 0.0,
        }
        return persona


# ----------------------------
# Item Profiler
# ----------------------------

class ItemProfiler:
    """
    Aggregates item-level priors (quality, controversy) and aspect summaries from histories.
    """

    def __init__(self, indexer: DataIndexer):
        self.indexer = indexer

    def profile(self, item_id: str) -> ItemProfileDict:
        history = self.indexer.get_item_history(item_id)
        aspect_vocab = self.indexer.get_aspect_vocab()
        i_mean = self.indexer.get_item_mean(item_id)
        quality_prior = i_mean if i_mean is not None else self.indexer.get_global_mean()
        stars = [r["stars"] for r in history if r.get("stars") is not None]
        variance = statistics.pvariance(stars) if len(stars) >= 2 else 0.0

        aspect_counts: Dict[str, int] = {a: 0 for a in aspect_vocab}
        for rec in history:
            aspects = detect_aspects(rec.get("review", ""), aspect_vocab)
            for a in aspects:
                aspect_counts[a] += 1
        total = sum(aspect_counts.values()) or 1
        aspect_summary = {a: aspect_counts[a] / total for a in aspect_vocab}
        profile: ItemProfileDict = {
            "item_id": item_id,
            "quality_prior": quality_prior,
            "controversy": variance,
            "aspect_summary": aspect_summary,
            "freshness_score": 0.5,  # placeholder without timestamps
        }
        return profile


# ----------------------------
# Plan Composer
# ----------------------------

class PlanComposer:
    """
    Creates an outline plan for review authoring (aspects, tone, length).
    """

    def __init__(self, indexer: DataIndexer, params: Parameters):
        self.indexer = indexer
        self.params = params

    def compose(self, persona: PersonaDict, item_profile: ItemProfileDict) -> PlanDict:
        aspect_vocab = self.indexer.get_aspect_vocab()
        w_p = persona.get("aspect_weights", {a: 1.0 / len(aspect_vocab) for a in aspect_vocab})
        w_i = item_profile.get("aspect_summary", {a: 1.0 / len(aspect_vocab) for a in aspect_vocab})
        alpha = self.params.ctx_merge_weight
        combined = {a: alpha * w_p.get(a, 0.0) + (1 - alpha) * w_i.get(a, 0.0) for a in aspect_vocab}

        # Softmax with temperature for diversity
        temp = max(0.1, self.params.plan_diversity_temp)
        exps = {a: math.exp(combined[a] / temp) for a in aspect_vocab}
        Z = sum(exps.values()) or 1.0
        probs = {a: exps[a] / Z for a in aspect_vocab}

        # Sample top-k aspects deterministically by probs rank
        topk = int(round(self.params.aspect_topk))
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        planned_aspects = [a for a, _ in ranked[:topk]]

        # Tone target from user leniency and item quality
        user_norm = clamp(persona.get("baseline_leniency", 0.0) / 2.0, -1.0, 1.0)
        item_norm = clamp((item_profile.get("quality_prior", self.indexer.get_global_mean()) - self.indexer.get_global_mean()) / 2.0, -1.0, 1.0)
        tone_val = clamp(0.6 * item_norm + 0.4 * user_norm, -1.0, 1.0)
        tone = "mixed"
        if tone_val > 0.2:
            tone = "positive"
        elif tone_val < -0.2:
            tone = "negative"

        # Length policy
        mean_len = int(round(self.params.length_target_mean))
        min_len = self.indexer.get_platform_policy()["style_constraints"]["length_sentences_min"]
        max_len = self.indexer.get_platform_policy()["style_constraints"]["length_sentences_max"]
        length_target = int(clamp(mean_len + random.choice([-1, 0, 1]), min_len, max_len))

        plan: PlanDict = {
            "planned_aspects": planned_aspects,
            "tone_target": tone,
            "tone_value": tone_val,
            "length_target": length_target,
        }
        return plan


# ----------------------------
# Review Author
# ----------------------------

class ReviewAuthor:
    """
    Generates review text from a plan and persona, honoring platform policy.
    """

    def __init__(self, indexer: DataIndexer, params: Parameters):
        self.indexer = indexer
        self.params = params

    def _temperature_pick(self, arr: List[str], temperature: float) -> str:
        """
        Pick an element with diversity proportional to temperature.
        """
        if not arr:
            return ""
        if temperature <= 0.25:
            return arr[0]
        idx = int(min(len(arr) - 1, max(0, round(random.random() ** (1.0 / max(0.01, temperature)) * (len(arr) - 1)))))
        return arr[idx]

    def _sentence_for_aspect(self, aspect: str, tone: str, style: str) -> str:
        pos_templates = {
            "quality": "The overall build quality feels solid and well put together.",
            "durability": "It seems durable and should hold up well over time.",
            "performance": "Performance is snappy with no noticeable lag.",
            "value": "Given the price, the value is excellent.",
            "design": "The design is sleek and looks great.",
            "usability": "Setup was straightforward and it's easy to use.",
            "comfort": "It feels comfortable in hand even after long use.",
            "battery": "Battery life easily lasts through a full day.",
            "sound": "Sound quality is clear with good detail.",
            "fit": "The fit is just right and feels secure.",
            "packaging": "Packaging was tidy and protective.",
            "delivery": "Shipping was quick and the item arrived on time.",
            "materials": "The materials feel premium and sturdy.",
            "warranty": "Support and warranty coverage inspire confidence.",
            "instructions": "The instructions are clear and helpful.",
        }
        neg_templates = {
            "quality": "The build quality feels a bit cheap and uneven.",
            "durability": "Durability is questionable; it may not last long.",
            "performance": "Performance is sluggish and there are occasional hiccups.",
            "value": "For the price, the value is disappointing.",
            "design": "The design feels dated and not very appealing.",
            "usability": "Setup is confusing and not very user-friendly.",
            "comfort": "It becomes uncomfortable during longer sessions.",
            "battery": "Battery life drains faster than expected.",
            "sound": "Sound quality is muddy and lacks clarity.",
            "fit": "The fit is off and doesn't feel secure.",
            "packaging": "Packaging could be better and offered minimal protection.",
            "delivery": "Shipping took longer than expected.",
            "materials": "The materials feel flimsy and cheap.",
            "warranty": "Customer support and warranty terms are underwhelming.",
            "instructions": "The instructions are vague and unhelpful.",
        }
        neutral_templates = {
            "quality": "The build quality is acceptable for everyday use.",
            "durability": "Durability seems fine so far.",
            "performance": "Performance is adequate for basic tasks.",
            "value": "The value is fair for what you get.",
            "design": "The design is simple and functional.",
            "usability": "Usability is okay once you get used to it.",
            "comfort": "Comfort is average and should suit most people.",
            "battery": "Battery life is typical; neither great nor terrible.",
            "sound": "Sound quality is decent for the price.",
            "fit": "The fit is generally okay.",
            "packaging": "Packaging was standard and did the job.",
            "delivery": "Delivery time was reasonable.",
            "materials": "Materials feel fine for the category.",
            "warranty": "Warranty terms are standard.",
            "instructions": "Instructions get the point across.",
        }
        # choose template based on tone
        if tone == "positive":
            base = pos_templates.get(aspect, "This aspect is well executed.")
        elif tone == "negative":
            base = neg_templates.get(aspect, "This aspect could be improved.")
        else:
            base = neutral_templates.get(aspect, "This aspect is acceptable.")

        # style variations with alignment weight
        align = clamp(self.params.style_alignment_weight, 0.0, 1.0)
        if style == "humorous" and align > 0.5:
            base += " Not a deal-breaker unless you expect miracles."
        elif style == "enthusiastic" and align > 0.5:
            base = base.replace(".", "!") + " Highly recommended!"

        # minor adjective/intensifier variants by temperature
        temp = clamp(self.params.llm_temperature, 0.2, 0.9)
        if random.random() < temp * 0.6:
            intensifiers = ["quite", "really", "surprisingly", "fairly", "notably"]
            adverb = self._temperature_pick(intensifiers, temp)
            base = re.sub(r"(^|\s)(is|feels|seems)\s", r"\1\2 " + adverb + " ", base, count=1)
        return base

    def generate(self, persona: PersonaDict, plan: PlanDict, item_profile: ItemProfileDict) -> str:
        style = persona.get("style", "informative")
        tone = plan.get("tone_target", "mixed")
        aspects = plan.get("planned_aspects", [])
        length_target = plan.get("length_target", 4)
        temp = clamp(self.params.llm_temperature, 0.2, 0.9)
        sentences: List[str] = []

        # Intro sentence
        intro_options = {
            "positive": [
                "This product exceeded my expectations.",
                "I'm impressed with how well this works.",
                "A great addition to my setup."
            ],
            "negative": [
                "I wanted to like this, but it falls short.",
                "Not quite what I hoped for.",
                "Several issues keep this from being a good buy."
            ],
            "mixed": [
                "There are things to like here, but also a few caveats.",
                "Overall it's decent, with some room for improvement.",
                "A balanced experience with strengths and weaknesses."
            ],
        }
        intro = self._temperature_pick(intro_options.get(tone, intro_options["mixed"]), temp)
        # amplify style
        if style == "enthusiastic" and random.random() < self.params.style_alignment_weight:
            intro = intro.replace(".", "!") + " Love the thoughtful touches."
        if style == "humorous" and random.random() < self.params.style_alignment_weight * 0.7:
            intro += " It won't make coffee, but it tries its best."

        sentences.append(intro)

        # Aspect sentences
        for aspect in aspects:
            if len(sentences) >= length_target:
                break
            sentences.append(self._sentence_for_aspect(aspect, tone, style))

        # Fill if needed
        generic_pos = "In daily use, it performs reliably and feels well thought out."
        generic_neg = "In daily use, several frustrations add up and limit its usefulness."
        generic_neu = "For everyday tasks, it gets the job done."
        while len(sentences) < length_target:
            pick_pool = {"positive": generic_pos, "negative": generic_neg, "mixed": generic_neu}
            candidate = pick_pool.get(tone, generic_neu)
            if random.random() < temp * 0.4:
                candidate = candidate.replace("daily", "regular").replace("everyday", "day-to-day")
            sentences.append(candidate)

        # Outro sentence
        outro = {
            "positive": "Overall, a solid buy if it fits your needs.",
            "negative": "Overall, I'd look for alternatives at this price.",
            "mixed": "Overall, consider your priorities to see if it fits.",
        }[tone]
        if len(sentences) < length_target + 1:
            sentences.append(outro)

        # Truncate or keep within policy bounds
        max_len = self.indexer.get_platform_policy()["style_constraints"]["length_sentences_max"]
        sentences = sentences[:max_len]

        text = " ".join(sentences)
        # profanity filter enforcement
        text, _ = clean_profanity(text)
        return text


# ----------------------------
# Star Rater
# ----------------------------

class StarRater:
    """
    Maps sentiment to a 1-5 star rating with user/item bias priors and calibrated mapping.
    """

    def __init__(self, indexer: DataIndexer, params: Parameters):
        self.indexer = indexer
        self.params = params

    def rate(self, text: str, user_id: str, item_id: str) -> float:
        s = sentiment_score(text)  # [-1, 1]
        u_mean = self.indexer.get_user_mean(user_id)
        i_mean = self.indexer.get_item_mean(item_id)
        g = self.indexer.get_global_mean()

        u_bias = ((u_mean - g) / 2.0) if u_mean is not None else 0.0
        i_bias = ((i_mean - g) / 2.0) if i_mean is not None else 0.0

        z = self.params.mapping_slope * (
            s + self.params.mapping_intercept
            + self.params.user_bias_weight * u_bias
            + self.params.item_bias_weight * i_bias
        )
        base = 1.0 + 4.0 * sigmoid(z)

        noise = random.gauss(0.0, 0.2 * self.params.uncertainty_scale)
        stars = clamp(base + noise, 1.0, 5.0)
        stars = round(stars * 2) / 2.0
        return stars


# ----------------------------
# QA Consistency
# ----------------------------

class QAConsistency:
    """
    Checks rating-text consistency and can adjust stars or text within a budget.
    """

    def __init__(self, indexer: DataIndexer, params: Parameters):
        self.indexer = indexer
        self.params = params

    def _expected_stars_with_bias(self, s: float, user_id: str, item_id: str) -> float:
        g = self.indexer.get_global_mean()
        u_mean = self.indexer.get_user_mean(user_id)
        i_mean = self.indexer.get_item_mean(item_id)
        u_bias = ((u_mean - g) / 2.0) if u_mean is not None else 0.0
        i_bias = ((i_mean - g) / 2.0) if i_mean is not None else 0.0
        z = self.params.mapping_slope * (s + self.params.mapping_intercept + self.params.user_bias_weight * u_bias + self.params.item_bias_weight * i_bias)
        expected = 1.0 + 4.0 * sigmoid(z)
        return round(clamp(expected, 1.0, 5.0) * 2) / 2.0

    def _revise_text_toward(self, text: str, target: str) -> str:
        """
        Heuristic small revision: add a sentence nudging sentiment toward target ('positive'|'negative')
        and lightly adjust adjectives.
        """
        if target == "positive":
            add = " After more use, I'm even happier with it."
            text = text + add
            text = re.sub(r"\badequate\b", "good", text)
            text = re.sub(r"\bdecent\b", "solid", text)
        elif target == "negative":
            add = " However, some issues keep dragging the experience down."
            text = text + add
            text = re.sub(r"\bgood\b", "okay", text)
            text = re.sub(r"\bexcellent\b", "decent", text)
        text, _ = clean_profanity(text)
        return text

    def check_and_fix(self, text: str, stars: float, plan: PlanDict, user_id: str, item_id: str) -> Tuple[float, float, str, float, int]:
        """
        Check and optionally fix consistency between sentiment(text) and stars.
        Returns: (qa_consistency_score, sentiment_value, possibly_revised_text, possibly_adjusted_stars, attempts_used)
        """
        attempts = 0
        revision_loops = 0
        while attempts <= self.params.max_auto_fix_attempts:
            s = sentiment_score(text)
            expected = self._expected_stars_with_bias(s, user_id, item_id)
            diff = abs(expected - stars) / 4.0
            consistency_score = 1.0 - diff
            # Style violations penalty (e.g., profanity)
            if self.indexer.get_platform_policy()["style_constraints"].get("no_profanity", True):
                _, pcount = clean_profanity(text)
                if pcount > 0:
                    consistency_score = max(0.0, consistency_score - min(0.5, 0.1 * pcount))

            if consistency_score >= self.params.consistency_threshold:
                return consistency_score, s, text, stars, attempts

            # Try revision of text toward expected tone if budget remains
            if revision_loops < self.params.max_revision_loops:
                tone_dir = "positive" if expected > stars else "negative"
                text = self._revise_text_toward(text, tone_dir)
                revision_loops += 1
            else:
                # Adjust stars toward expected
                stars = round(((stars + expected) / 2.0) * 2) / 2.0
                attempts += 1

        # Final computation after exits
        s = sentiment_score(text)
        expected = self._expected_stars_with_bias(s, user_id, item_id)
        diff = abs(expected - stars) / 4.0
        consistency_score = 1.0 - diff
        if self.indexer.get_platform_policy()["style_constraints"].get("no_profanity", True):
            _, pcount = clean_profanity(text)
            if pcount > 0:
                consistency_score = max(0.0, consistency_score - min(0.5, 0.1 * pcount))
        return consistency_score, s, text, stars, attempts


# ----------------------------
# Evaluator
# ----------------------------

class Evaluator:
    """
    Computes per-record and aggregate metrics for simulation outputs.
    """

    def __init__(self, indexer: DataIndexer):
        self.indexer = indexer

    def per_record_metrics(
        self,
        generated_text: str,
        generated_stars: float,
        reference_text: str,
        reference_stars: float,
        plan: PlanDict
    ) -> MetricsDict:
        # Preference metrics
        star_err = abs(generated_stars - reference_stars)
        star_sq = (generated_stars - reference_stars) ** 2

        # Text similarity
        gen_tokens = tokenize(generated_text)
        ref_tokens = tokenize(reference_text)
        sim = jaccard_similarity(gen_tokens, ref_tokens)

        # Aspect coverage and topic relevance
        aspect_vocab = self.indexer.get_aspect_vocab()
        gen_aspects = set(detect_aspects(generated_text, aspect_vocab))
        ref_aspects = set(detect_aspects(reference_text, aspect_vocab))
        planned_aspects = set(plan.get("planned_aspects", []))
        aspect_coverage = 1.0 if not planned_aspects else len(gen_aspects & planned_aspects) / max(1, len(planned_aspects))
        topic_jaccard = jaccard_similarity(gen_aspects, ref_aspects)

        # Sentiment agreement (between text polarity and stars)
        s_text = sentiment_score(generated_text)
        gen_att = 1.0 if generated_stars >= 4.0 else 0.0 if generated_stars <= 2.5 else 0.5
        ref_att = 1.0 if sentiment_score(reference_text) >= 0.2 else 0.0 if sentiment_score(reference_text) <= -0.2 else 0.5
        sentiment_agreement = 1.0 - abs(gen_att - (1.0 if s_text >= 0.2 else 0.0 if s_text <= -0.2 else 0.5))

        # Length deviation
        target_len = plan.get("length_target", 4)
        gen_len = max(1, generated_text.count(".") + generated_text.count("!") + generated_text.count("?"))
        length_deviation = abs(gen_len - target_len)

        # Composite "Review Generation" metric components
        emotional_tone_error = abs((s_text + 1) / 2.0 - (sentiment_score(reference_text) + 1) / 2.0)
        sentiment_attitude_error = abs(gen_att - ref_att)
        topic_relevance_error = 1.0 - topic_jaccard
        review_generation = 1.0 - (0.25 * emotional_tone_error + 0.25 * sentiment_attitude_error + 0.5 * topic_relevance_error)

        return {
            "MAE_stars": star_err,
            "RMSE_contrib": star_sq,
            "Text_Similarity": sim,
            "Aspect_Coverage": aspect_coverage,
            "Sentiment_Agreement": sentiment_agreement,
            "Length_Deviation": length_deviation,
            "Review_Generation": review_generation,
        }

    def aggregate_metrics(self, records: List[MetricsDict]) -> Dict[str, Any]:
        """
        Aggregate metrics over records. Returns a full metrics dict even if empty.
        """
        def defaults() -> Dict[str, Any]:
            return {
                "RMSE_stars": 0.0,
                "MAE_stars": 0.0,
                "Text_Similarity": 0.0,
                "Sentiment_Agreement": 0.0,
                "Aspect_Coverage": 0.0,
                "QA_Consistency": 0.0,
                "Consistency_Score": 0.0,
                "Length_Deviation": 0.0,
                "Preference_Estimation": 0.0,
                "Review_Generation": 0.0,
                "Overall_Quality": 0.0,
            }

        if not records:
            return defaults()

        mae = statistics.mean([r.get("MAE_stars", 0.0) for r in records]) if records else 0.0
        rmse = math.sqrt(statistics.mean([r.get("RMSE_contrib", 0.0) for r in records])) if records else 0.0
        text_sim = statistics.mean([r.get("Text_Similarity", 0.0) for r in records]) if records else 0.0
        senti_agree = statistics.mean([r.get("Sentiment_Agreement", 0.0) for r in records]) if records else 0.0
        aspect_cov = statistics.mean([r.get("Aspect_Coverage", 0.0) for r in records]) if records else 0.0
        length_dev = statistics.mean([r.get("Length_Deviation", 0.0) for r in records]) if records else 0.0
        review_gen = statistics.mean([r.get("Review_Generation", 0.0) for r in records]) if records else 0.0
        qa_consistency = statistics.mean([r.get("QA_Consistency", 0.0) for r in records if "QA_Consistency" in r]) if records else 0.0

        preference_estimation = 1.0 - min(1.0, mae / 4.0)
        overall_quality = (preference_estimation + review_gen) / 2.0

        return {
            "RMSE_stars": rmse,
            "MAE_stars": mae,
            "Text_Similarity": text_sim,
            "Sentiment_Agreement": senti_agree,
            "Aspect_Coverage": aspect_cov,
            "QA_Consistency": qa_consistency,
            "Consistency_Score": qa_consistency,  # alias per spec
            "Length_Deviation": length_dev,
            "Preference_Estimation": preference_estimation,
            "Review_Generation": review_gen,
            "Overall_Quality": overall_quality,
        }


# ----------------------------
# Parameter Tuner
# ----------------------------

class ParameterTuner:
    """
    Random search parameter tuner that minimizes a weighted objective on train split.
    """

    def __init__(self, indexer: DataIndexer, base_params: Parameters):
        self.indexer = indexer
        self.base_params = base_params
        self.history: List[Dict[str, Any]] = []
        self.best_params: Parameters = base_params
        self.best_objective: float = float("inf")

    def _sample_params(self) -> Parameters:
        nw = random.uniform(0.0, 0.5) if self.indexer.has_social_graph() else 0.0
        p = Parameters(
            neighbor_weight=nw,
            ctx_merge_weight=random.uniform(0.2, 0.8),
            aspect_topk=int(round(random.uniform(3, 6))),
            length_target_mean=int(round(random.uniform(2, 6))),
            plan_diversity_temp=random.uniform(0.3, 1.2),
            llm_temperature=random.uniform(0.2, 0.9),
            style_alignment_weight=random.uniform(0.3, 1.0),
            mapping_slope=random.uniform(2.0, 8.0),
            mapping_intercept=random.uniform(-2.0, 2.0),
            user_bias_weight=random.uniform(0.0, 1.0),
            item_bias_weight=random.uniform(0.0, 1.0),
            uncertainty_scale=random.uniform(0.1, 1.0),
            consistency_threshold=random.uniform(0.6, 0.9),
            max_auto_fix_attempts=int(round(random.uniform(0, 2))),
            objective_weights=self.base_params.objective_weights.normalize()
        )
        return p

    def _simulate_once(self, params: Parameters, train_subset: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        if not train_subset:
            # Return a no-op objective and default metrics
            return float("inf"), {
                "RMSE_stars": 0.0,
                "MAE_stars": 0.0,
                "Text_Similarity": 0.0,
                "Sentiment_Agreement": 0.0,
                "Aspect_Coverage": 0.0,
                "QA_Consistency": 0.0,
                "Consistency_Score": 0.0,
                "Length_Deviation": 0.0,
                "Preference_Estimation": 0.0,
                "Review_Generation": 0.0,
                "Overall_Quality": 0.0,
            }

        persona_profiler = PersonaProfiler(self.indexer, params)
        item_profiler = ItemProfiler(self.indexer)
        composer = PlanComposer(self.indexer, params)
        author = ReviewAuthor(self.indexer, params)
        rater = StarRater(self.indexer, params)
        qa = QAConsistency(self.indexer, params)
        evaluator = Evaluator(self.indexer)
        per_rec: List[MetricsDict] = []
        for rec in train_subset:
            uid = rec["user_id"]
            iid = rec["item_id"]
            ref_text = rec.get("review", "")
            ref_stars = rec.get("stars", 3.0) or 3.0
            persona = persona_profiler.profile(uid)
            item_profile = item_profiler.profile(iid)
            plan = composer.compose(persona, item_profile)
            text = author.generate(persona, plan, item_profile)
            stars = rater.rate(text, uid, iid)
            qa_consistency, s_text, text, stars, attempts = qa.check_and_fix(text, stars, plan, uid, iid)
            # Evaluate against reference (train)
            m = evaluator.per_record_metrics(text, stars, ref_text, ref_stars, plan)
            # include QA consistency
            m["QA_Consistency"] = qa_consistency
            per_rec.append(m)
        agg = evaluator.aggregate_metrics(per_rec)
        # Objective to minimize: weighted loss
        w = params.objective_weights.normalize()
        mae_norm = agg.get("MAE_stars", 0.0) / 4.0
        text_loss = 1.0 - agg.get("Text_Similarity", 0.0)
        consistency_loss = 1.0 - agg.get("QA_Consistency", 0.0)
        objective = w.stars * mae_norm + w.text * text_loss + w.consistency * consistency_loss
        return objective, agg

    def tune(self, num_trials: int = 12, seed: int = 42, max_train_subset: int = 60) -> None:
        random.seed(seed)
        train_all = self.indexer.get_train()
        # sample subset for speed
        if len(train_all) > max_train_subset:
            train_subset = random.sample(train_all, max_train_subset)
        else:
            train_subset = train_all

        if not train_subset:
            print("[TUNE] No training records available. Skipping tuning and using base parameters.")
            self.best_params = self.base_params
            self.best_objective = float("inf")
            self.history.append({"params": self.base_params.to_dict(), "objective": self.best_objective, "metrics": {}})
            return

        # Baseline objective from base params
        base_obj, base_agg = self._simulate_once(self.base_params, train_subset)
        self.best_objective = base_obj
        self.best_params = self.base_params
        self.history.append({"params": self.base_params.to_dict(), "objective": base_obj, "metrics": base_agg})
        for t in range(num_trials):
            params = self._sample_params()
            obj, agg = self._simulate_once(params, train_subset)
            self.history.append({"params": params.to_dict(), "objective": obj, "metrics": agg})
            if obj < self.best_objective:
                self.best_objective = obj
                self.best_params = params
                print(f"[TUNE] New best at trial {t+1}/{num_trials}: objective={obj:.4f} metrics={agg}")
        print(f"[TUNE] Best objective: {self.best_objective:.4f}")


# ----------------------------
# Simulation Orchestrator
# ----------------------------

class Simulator:
    """
    Orchestrates per-record simulation from persona to review and rating.
    """

    def __init__(self, indexer: DataIndexer, params: Parameters, seed: int = 123):
        self.indexer = indexer
        self.params = params
        random.seed(seed)

        self.persona_profiler = PersonaProfiler(indexer, params)
        self.item_profiler = ItemProfiler(indexer)
        self.composer = PlanComposer(indexer, params)
        self.author = ReviewAuthor(indexer, params)
        self.rater = StarRater(indexer, params)
        self.qa = QAConsistency(indexer, params)
        self.evaluator = Evaluator(indexer)

    def simulate_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        uid = rec["user_id"]
        iid = rec["item_id"]
        persona = self.persona_profiler.profile(uid)
        item_profile = self.item_profiler.profile(iid)
        plan = self.composer.compose(persona, item_profile)
        text = self.author.generate(persona, plan, item_profile)
        stars = self.rater.rate(text, uid, iid)
        consistency_score, s_text, text, stars, fix_attempts = self.qa.check_and_fix(text, stars, plan, uid, iid)
        return {
            "user_id": uid,
            "item_id": iid,
            "generated_text": text,
            "generated_stars": stars,
            "plan": plan,
            "persona": {
                "baseline_leniency": persona.get("baseline_leniency", 0.0),
                "verbosity_prior": persona.get("verbosity_prior", 0.0),
                "style": persona.get("style", "informative")
            },
            "item_profile": {
                "quality_prior": item_profile.get("quality_prior", self.indexer.get_global_mean()),
                "controversy": item_profile.get("controversy", 0.0)
            },
            "consistency_score": consistency_score,
            "consistency_fix_attempts": fix_attempts
        }

    def run_split(self, split: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if split == "train":
            records = self.indexer.get_train()
        else:
            records = self.indexer.get_test()
        simulation_outputs: List[Dict[str, Any]] = []
        per_record_metrics: List[Dict[str, Any]] = []
        for rec in records:
            out = self.simulate_record(rec)
            # Evaluate with ground truth from split (only for evaluation/logging; gen did not use test reference)
            ref_text = rec.get("review", "")
            ref_stars = rec.get("stars", 3.0) or 3.0
            m = self.evaluator.per_record_metrics(
                out["generated_text"], out["generated_stars"], ref_text, ref_stars, out["plan"]
            )
            # attach QA consistency for aggregation
            m["QA_Consistency"] = out["consistency_score"]
            simulation_outputs.append({
                "split": split,
                "input": {"user_id": out["user_id"], "item_id": out["item_id"]},
                "persona": out["persona"],
                "item_profile": out["item_profile"],
                "plan": out["plan"],
                "generated": {"text": out["generated_text"], "stars": out["generated_stars"]},
                "metrics": m,
                "consistency": {
                    "score": out["consistency_score"],
                    "fix_attempts": out["consistency_fix_attempts"]
                }
            })
            per_rec = dict(m)
            per_rec["user_id"] = out["user_id"]
            per_rec["item_id"] = out["item_id"]
            per_record_metrics.append(per_rec)
        return simulation_outputs, per_record_metrics


# ----------------------------
# Ablation Runner
# ----------------------------

def run_ablation(indexer: DataIndexer, base_params: Parameters, name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    params_dict = base_params.to_dict()
    params_dict.update(overrides)
    # keep objective_weights object
    params = Parameters(**{**params_dict, "objective_weights": base_params.objective_weights})
    sim = Simulator(indexer, params)
    _, per_metrics = sim.run_split("test")
    agg = sim.evaluator.aggregate_metrics(per_metrics)
    return {"name": name, "params": params.to_dict(), "metrics": agg}


# ----------------------------
# Main orchestration
# ----------------------------

def main():
    base_dir = Path("data_fitting/agent_society")
    out_dir = Path("outputs/agent_society")
    random.seed(1234)

    indexer = DataIndexer(base_dir)

    # Parameter tuning on training split
    base_params = Parameters()
    tuner = ParameterTuner(indexer, base_params)
    tuner.tune(num_trials=10, seed=1234, max_train_subset=min(60, max(10, len(indexer.get_train()))))
    best_params = tuner.best_params

    # Persist calibrated parameters and tuning history
    write_json(out_dir / "calibrated_parameters.json", best_params.to_dict())
    write_json(out_dir / "tuning_history.json", tuner.history)

    # Run simulator on test split with tuned parameters
    simulator = Simulator(indexer, best_params, seed=5678)
    test_traces, test_per_rec = simulator.run_split("test")

    # Persist traces
    write_jsonl(out_dir / "simulation_traces.jsonl", test_traces)

    # Aggregate evaluation
    evaluator = simulator.evaluator
    overall_metrics = evaluator.aggregate_metrics(test_per_rec)

    # Compute by-segment metrics (user frequency tertiles)
    user_train_counts = {u: len(indexer.get_user_history(u)) for u in set(r["user_id"] for r in indexer.get_train())}
    counts_nonzero = sorted([c for c in user_train_counts.values() if c > 0])
    if counts_nonzero:
        q1_idx = max(0, int(0.33 * (len(counts_nonzero) - 1)))
        q2_idx = max(0, int(0.66 * (len(counts_nonzero) - 1)))
        q1 = counts_nonzero[q1_idx]
        q2 = counts_nonzero[q2_idx]
    else:
        q1 = q2 = 0

    segments: Dict[str, List[Dict[str, Any]]] = {"cold": [], "low": [], "mid": [], "high": []}
    for r in test_per_rec:
        uid = r["user_id"]
        cnt = user_train_counts.get(uid, 0)
        if cnt == 0:
            segments["cold"].append(r)
        elif cnt <= q1:
            segments["low"].append(r)
        elif cnt <= q2:
            segments["mid"].append(r)
        else:
            segments["high"].append(r)

    by_segment_metrics = {}
    for seg_name, seg_records in segments.items():
        by_segment_metrics[seg_name] = evaluator.aggregate_metrics(seg_records) if seg_records else evaluator.aggregate_metrics([])

    # Preference Estimation, Review Generation, Overall Quality per spec
    evaluation_metrics = {
        "overall": overall_metrics,
        "by_user_frequency": by_segment_metrics
    }

    write_json(out_dir / "evaluation_metrics.json", evaluation_metrics)

    # Ablation study
    ablations: List[Dict[str, Any]] = []
    # 1) Consistency off
    ablations.append(run_ablation(indexer, best_params, "consistency_off", {"max_auto_fix_attempts": 0, "max_revision_loops": 0}))
    # 2) Persona-dominant planning
    ablations.append(run_ablation(indexer, best_params, "persona_dominant", {"ctx_merge_weight": 0.8}))
    # 3) Item-dominant planning
    ablations.append(run_ablation(indexer, best_params, "item_dominant", {"ctx_merge_weight": 0.2}))
    write_json(out_dir / "ablation_report.json", {"ablations": ablations})

    print("[DONE] Simulation complete. Outputs written to:", out_dir)


# Execute main for direct execution
main()