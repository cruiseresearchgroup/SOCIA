import os
import sys
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def generate_small_world(n: int, k: int, beta: float, rng: random.Random) -> List[List[int]]:
    if n <= 1:
        return [[] for _ in range(n)]
    k = max(2, int(k))
    if k % 2 == 1:
        k += 1
    half_k = k // 2
    nbrs = [set() for _ in range(n)]
    # Ring lattice
    for i in range(n):
        for d in range(1, half_k + 1):
            j = (i + d) % n
            nbrs[i].add(j)
            nbrs[j].add(i)
    # Rewire edges
    for i in range(n):
        for j in list(nbrs[i]):
            if j > i and rng.random() < beta:
                nbrs[i].discard(j)
                nbrs[j].discard(i)
                candidates = [x for x in range(n) if x != i and x not in nbrs[i]]
                if candidates:
                    new_j = rng.choice(candidates)
                    nbrs[i].add(new_j)
                    nbrs[new_j].add(i)
    return [sorted(s) for s in nbrs]


@dataclass
class Person:
    id: int
    age: int
    household_id: int
    socioeconomic_status: float
    risk_perception: float
    trust_in_government: float
    trust_in_media: float
    baseline_compliance_propensity: float
    mask_attitude: float
    wearing_mask: bool
    masks_inventory: int
    social_network_neighbors: List[int]
    daily_mobility_profile: Dict[str, float]
    workplace_id: Optional[int] = None
    school_id: Optional[int] = None
    health_status: str = "healthy"
    information_exposure_level: float = 0.0
    social_susceptibility: float = 0.5

    def decide_to_wear_mask(
        self,
        peer_adoption_rate: float,
        observed_public_wearing: float,
        policy_strength: float,
        enforcement_probability: float,
        market_availability: float,
        price_per_mask: float,
        weights: Dict[str, float],
        rng: random.Random,
    ) -> bool:
        # Inventory/economic gating
        if self.masks_inventory <= 0:
            affordability = clamp01((self.socioeconomic_status + 0.1) - 0.5 * max(0.0, price_per_mask - 1.0))
            if rng.random() > (market_availability * affordability):
                return False

        social_term = weights.get("social_influence_weight", 0.4) * self.social_susceptibility * peer_adoption_rate
        personal_term = weights.get("personal_attitude_weight", 0.3) * (
            0.5 * self.baseline_compliance_propensity + 0.5 * ((self.mask_attitude + 1.0) / 2.0)
        )
        policy_term = weights.get("policy_compliance_weight", 0.2) * (
            self.trust_in_government * policy_strength + enforcement_probability
        )
        observation_term = weights.get("observation_weight", 0.1) * observed_public_wearing
        risk_term = 0.3 * (self.risk_perception - 0.5)

        utility = social_term + personal_term + policy_term + observation_term + risk_term
        p = sigmoid(2.5 * (utility - 0.5))
        return rng.random() < p

    def update_beliefs_from_peers(self, neighbor_attitudes: List[float]) -> None:
        if not neighbor_attitudes:
            return
        avg_att = sum(neighbor_attitudes) / len(neighbor_attitudes)
        self.mask_attitude = clamp01(((self.mask_attitude + 1.0) / 2.0) * 0.8 + ((avg_att + 1.0) / 2.0) * 0.2) * 2.0 - 1.0

    def update_beliefs_from_media(self, media_delta: Dict[str, float]) -> None:
        self.risk_perception = clamp01(self.risk_perception + media_delta.get("risk", 0.0) * self.trust_in_media)
        self.trust_in_government = clamp01(self.trust_in_government + media_delta.get("trust_gov", 0.0))
        self.mask_attitude = clamp01(((self.mask_attitude + 1.0) / 2.0) + media_delta.get("attitude", 0.0)) * 2.0 - 1.0

    def respond_to_policy(self, compliance_decay_rate: float) -> None:
        if self.wearing_mask:
            self.mask_attitude = clamp01(((self.mask_attitude + 1.0) / 2.0) + 0.003) * 2.0 - 1.0
        else:
            self.baseline_compliance_propensity = clamp01(self.baseline_compliance_propensity - compliance_decay_rate)

    def purchase_masks(self, retailer: "Retailer", max_purchase: int = 2) -> Tuple[int, bool]:
        desired = max(0, max_purchase - self.masks_inventory)
        if desired <= 0:
            return 0, False
        affordable_qty = int(max(0, math.floor((self.socioeconomic_status * 10.0) / max(0.1, retailer.price_per_mask))))
        qty = clamp01(affordable_qty / max(1.0, desired)) * desired
        qty = int(min(desired, max(0, round(qty))))
        purchased, shortage = retailer.sell_masks(qty)
        self.masks_inventory += purchased
        return purchased, shortage

    def travel_and_interact_at_locations(self, location_types: List[str], rng: random.Random) -> List[str]:
        visits = []
        for loc in location_types:
            prob = self.daily_mobility_profile.get(loc, 0.0)
            if rng.random() < prob:
                visits.append(loc)
        return visits


@dataclass
class Household:
    id: int
    member_ids: List[int]
    household_income: float
    norm_strength: float
    mask_inventory: int = 0

    def share_norms(self, persons: Dict[int, Person]) -> None:
        if not self.member_ids:
            return
        avg_att = sum(persons[i].mask_attitude for i in self.member_ids) / len(self.member_ids)
        for i in self.member_ids:
            p = persons[i]
            p.mask_attitude = clamp01(
                ((p.mask_attitude + 1.0) / 2.0) * (1.0 - 0.2 * self.norm_strength) + ((avg_att + 1.0) / 2.0) * 0.2 * self.norm_strength
            ) * 2.0 - 1.0

    def pool_resources_for_masks(self, persons: Dict[int, Person]) -> None:
        for i in self.member_ids:
            if self.mask_inventory <= 0:
                break
            if persons[i].masks_inventory <= 0:
                persons[i].masks_inventory += 1
                self.mask_inventory -= 1


@dataclass
class Location:
    id: int
    type: str
    capacity: int
    mask_policy: str
    enforcement_level: float
    foot_traffic_rate: float

    def enforce_mask_policy(self, visitor_persons: List[Person], authority: "HealthAuthority", rng: random.Random) -> int:
        if self.mask_policy != "mandate" or self.enforcement_level <= 0.0:
            return 0
        actions = 0
        inspect_prob = clamp01(self.enforcement_level * authority.enforcement_intensity)
        for p in visitor_persons:
            if not p.wearing_mask and rng.random() < inspect_prob:
                actions += 1
                p.baseline_compliance_propensity = clamp01(p.baseline_compliance_propensity + 0.05)
                p.risk_perception = clamp01(p.risk_perception + 0.02)
        return actions

    def broadcast_signage(self) -> float:
        if self.mask_policy == "mandate":
            return 0.05
        if self.mask_policy == "recommended":
            return 0.02
        return 0.0


@dataclass
class HealthAuthority:
    id: int
    mandate_on: bool
    mandate_start_day: int
    mandate_scope: str
    fine_amount: float
    enforcement_resources: float
    enforcement_intensity: float = 0.3
    guidance_strength: float = 0.3

    def set_or_update_policy(self, day: int) -> None:
        if day >= self.mandate_start_day:
            self.mandate_on = True

    def allocate_enforcement(self, locations: List[Location]) -> None:
        if not locations:
            return
        base = clamp01(self.enforcement_resources)
        for loc in locations:
            if loc.mask_policy == "mandate":
                loc.enforcement_level = clamp01(base * (0.5 + 0.5 * loc.foot_traffic_rate))
            else:
                loc.enforcement_level = clamp01(0.2 * base * loc.foot_traffic_rate)

    def issue_public_guidance(self) -> Dict[str, float]:
        return {
            "risk": 0.01 * self.guidance_strength,
            "trust_gov": 0.01 * self.guidance_strength,
            "attitude": 0.005 * self.guidance_strength,
        }


@dataclass
class InformationSource:
    id: int
    credibility: float
    message_intensity: float
    message_slant: float
    reach: float
    misinformation_rate: float = 0.0
    campaign_intensity: float = 0.0

    def broadcast_messages(self, rng: random.Random) -> Dict[str, float]:
        sign = 1.0 if self.message_slant >= 0 else -1.0
        misinf = 1.0 if rng.random() < self.misinformation_rate else 0.0
        attitude_delta = (0.01 * self.message_intensity * sign * self.credibility) * (1.0 - 2.0 * misinf)
        risk_delta = 0.01 * self.message_intensity * max(0.0, sign) * self.credibility
        return {"risk": risk_delta, "attitude": attitude_delta}


@dataclass
class Retailer:
    id: int
    inventory_level: int
    restock_rate: float
    price_per_mask: float
    rationing_policy: int = 2

    def sell_masks(self, quantity: int) -> Tuple[int, bool]:
        quantity = int(max(0, min(self.rationing_policy, quantity)))
        if self.inventory_level <= 0:
            return 0, True
        sold = min(self.inventory_level, quantity)
        self.inventory_level -= sold
        shortage = sold < quantity
        return sold, shortage

    def adjust_prices(self) -> None:
        base = 1.0
        scarcity = 0.0 if self.inventory_level > 500 else (1.0 - (self.inventory_level / 500.0))
        self.price_per_mask = base * (1.0 + 0.5 * scarcity)

    def restock(self, capacity: int = 1000) -> None:
        restock_qty = int(max(0, (capacity - self.inventory_level) * self.restock_rate))
        self.inventory_level += restock_qty


class Simulation:
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params
        self.rng = random.Random(int(params.get("random_seed", 42)))
        self.population_size = int(params.get("population_size", 500))
        self.time_horizon_days = int(params.get("time_horizon_days", 60))
        self.avg_degree = int(params.get("avg_degree", params.get("average_degree", 8)))
        # FIXED: removed extra parenthesis here
        self.rewiring_prob = float(params.get("rewiring_prob", 0.05))
        self.initial_adoption_rate = float(params.get("initial_adoption_rate", 0.1))
        self.mask_availability_initial = float(params.get("mask_availability_initial", params.get("mask_availability", 0.9)))
        self.retailer_restock_rate = float(params.get("retailer_restock_rate", 0.1))
        self.mask_price = float(params.get("mask_price", 1.0))
        self.price_elasticity = float(params.get("price_elasticity_of_demand", -0.2))
        self.risk_signal_initial = float(params.get("risk_signal_initial", 0.2))
        self.risk_signal_amplitude = float(params.get("risk_signal_amplitude", 0.3))
        self.risk_signal_decay = float(params.get("risk_signal_decay", 0.01))
        self.compliance_decay_rate = float(params.get("compliance_decay_rate", 0.005))

        self.weights = {
            "social_influence_weight": float(params.get("social_influence_weight", params.get("social_influence_strength", 0.5))),
            "personal_attitude_weight": float(params.get("personal_attitude_weight", 0.3)),
            "policy_compliance_weight": float(params.get("policy_compliance_weight", params.get("government_guidance_influence", 0.35))),
            "observation_weight": float(params.get("observation_weight", params.get("observation_effect", 0.4))),
        }

        self.people: Dict[int, Person] = {}
        self.households: Dict[int, Household] = {}
        self.locations: List[Location] = []

        self.authority = HealthAuthority(
            id=0,
            mandate_on=False,
            mandate_start_day=int(params.get("mandate_start_day", 30)),
            mandate_scope=str(params.get("mandate_scope", "indoor_public")),
            fine_amount=float(params.get("fine_amount", 50.0)),
            enforcement_resources=float(params.get("enforcement_level", 0.3)),
            enforcement_intensity=float(params.get("enforcement_level", 0.3)),
            guidance_strength=float(params.get("government_guidance_influence", 0.35)),
        )
        self.media = InformationSource(
            id=0,
            credibility=float(params.get("media_credibility", 0.7)),
            message_intensity=float(params.get("media_influence_strength", 0.3)),
            message_slant=float(params.get("media_message_slant", 1.0)),
            reach=float(params.get("media_reach", 0.6)),
            misinformation_rate=float(params.get("misinformation_rate", 0.05)),
            campaign_intensity=float(params.get("campaign_intensity", 0.2)),
        )
        self.retailer = Retailer(
            id=0,
            inventory_level=int(params.get("initial_inventory", 1000)),
            restock_rate=self.retailer_restock_rate,
            price_per_mask=self.mask_price,
            rationing_policy=int(params.get("rationing_policy", 2)),
        )

        # Network
        self.network = generate_small_world(self.population_size, self.avg_degree, self.rewiring_prob, self.rng)

        # Locations setup
        loc_types = params.get(
            "location_types",
            ["home", "work", "school", "store", "public_transport", "other_public"],
        )
        foot_mult = params.get(
            "foot_traffic_multipliers",
            {"home": 1.0, "work": 0.6, "school": 0.5, "store": 0.2, "public_transport": 0.15, "other_public": 0.3},
        )
        loc_id = 0
        for t in loc_types:
            count = {"home": self.population_size // 4, "work": self.population_size // 10, "school": self.population_size // 15}.get(t, 3)
            for _ in range(max(1, count)):
                self.locations.append(
                    Location(
                        id=loc_id,
                        type=t,
                        capacity=max(10, int(self.population_size * 0.1)),
                        mask_policy="none",
                        enforcement_level=0.0,
                        foot_traffic_rate=float(foot_mult.get(t, 0.3)),
                    )
                )
                loc_id += 1

        self._initialize_people_and_households()

        # Metrics
        self.metrics: Dict[str, Any] = {
            "adoption_rate_over_time": [],
            "compliance_over_time": [],
            "churn_over_time": [],
            "mask_shortage_incidents": 0,
            "enforcement_actions_count": 0,
        }
        self.retailer_inventory_history: List[int] = []

        # Internal state
        self._last_wearing_snapshot: List[bool] = [p.wearing_mask for p in self.people.values()]

    def _initialize_people_and_households(self) -> None:
        n = self.population_size
        # Households
        hh_id = 0
        i = 0
        while i < n:
            size = max(1, min(5, int(self.rng.gauss(2.5, 1.0))))
            member_ids = list(range(i, min(n, i + size)))
            income = clamp01(self.rng.random() * (0.7 + 0.3 * self.rng.random()))
            norm_strength = clamp01(self.rng.random())
            self.households[hh_id] = Household(hh_id, member_ids, income, norm_strength, mask_inventory=self.rng.randint(0, 3))
            i += size
            hh_id += 1

        # People
        for pid in range(n):
            # assign household_id
            hid = next(h for h, hh in self.households.items() if pid in hh.member_ids)
            ses = clamp01(self.households[hid].household_income + self.rng.uniform(-0.2, 0.2))
            risk = clamp01(self.risk_signal_initial + self.rng.uniform(-0.1, 0.1))
            trust_gov = clamp01(0.4 + self.rng.uniform(-0.2, 0.2))
            trust_media = clamp01(0.5 + self.rng.uniform(-0.3, 0.3))
            baseline = clamp01(0.5 + self.rng.uniform(-0.3, 0.3))
            attitude = clamp01(0.5 + self.rng.uniform(-0.3, 0.3)) * 2.0 - 1.0
            wearing = self.rng.random() < self.initial_adoption_rate
            inv = self.rng.randint(0, 2)
            mobility = {
                "home": 1.0,
                "work": self.rng.uniform(0.0, 0.7),
                "school": self.rng.uniform(0.0, 0.5),
                "store": self.rng.uniform(0.1, 0.3),
                "public_transport": self.rng.uniform(0.0, 0.2),
                "other_public": self.rng.uniform(0.1, 0.4),
            }
            self.people[pid] = Person(
                id=pid,
                age=int(max(1, min(90, int(self.rng.gauss(40, 18))))),
                household_id=hid,
                socioeconomic_status=ses,
                risk_perception=risk,
                trust_in_government=trust_gov,
                trust_in_media=trust_media,
                baseline_compliance_propensity=baseline,
                mask_attitude=attitude,
                wearing_mask=wearing,
                masks_inventory=inv,
                social_network_neighbors=self.network[pid] if pid < len(self.network) else [],
                daily_mobility_profile=mobility,
                social_susceptibility=clamp01(0.5 + self.rng.uniform(-0.2, 0.2)),
            )

    def _update_location_policies(self) -> None:
        # Simple mapping: if mandate_on, non-home locations are 'mandate', else 'recommended' for store/transport
        if self.authority.mandate_on:
            for loc in self.locations:
                if loc.type != "home":
                    loc.mask_policy = "mandate"
                else:
                    loc.mask_policy = "none"
        else:
            for loc in self.locations:
                if loc.type in ("store", "public_transport", "other_public"):
                    loc.mask_policy = "recommended"
                else:
                    loc.mask_policy = "none"

    def _risk_signal(self, day: int) -> float:
        # Simple decaying sinusoid around initial risk
        amp = self.risk_signal_amplitude * math.exp(-self.risk_signal_decay * day)
        return clamp01(self.risk_signal_initial + amp * math.sin(day / 7.0 * 2 * math.pi))

    def step(self, day: int) -> None:
        self.authority.set_or_update_policy(day)
        self._update_location_policies()
        self.authority.allocate_enforcement(self.locations)

        # Media and guidance combined delta; only a fraction reached
        media_delta = self.media.broadcast_messages(self.rng)
        guidance = self.authority.issue_public_guidance()
        combined = {"risk": media_delta["risk"] + guidance["risk"], "trust_gov": guidance["trust_gov"], "attitude": media_delta["attitude"] + guidance["attitude"]}

        # Global observed wearing from previous day
        prev_wearing_rate = sum(1 for p in self.people.values() if p.wearing_mask) / max(1, len(self.people))
        signage_boost = sum(loc.broadcast_signage() for loc in self.locations) / max(1, len(self.locations))
        observed_public_wearing = clamp01(prev_wearing_rate + signage_boost)

        # Retailer dynamics
        self.retailer.adjust_prices()
        self.retailer.restock()

        # Precompute enforcement probability proxy (avg across non-home)
        non_home_locs = [loc for loc in self.locations if loc.type != "home"]
        avg_enforcement = 0.0
        if non_home_locs:
            avg_enforcement = sum(loc.enforcement_level for loc in non_home_locs) / len(non_home_locs)

        # Decisions
        shortage_incidents = 0
        new_wearing_flags: List[bool] = [False] * len(self.people)
        for pid, person in self.people.items():
            # Update beliefs
            if self.rng.random() < self.media.reach:
                person.update_beliefs_from_media(combined)
            # Risk drift
            person.risk_perception = clamp01(0.9 * person.risk_perception + 0.1 * self._risk_signal(day))

            # Peer influence
            neighbor_att = [self.people[nid].mask_attitude for nid in person.social_network_neighbors if 0 <= nid < len(self.people)]
            person.update_beliefs_from_peers(neighbor_att)
            peer_wearing = 0.0
            if person.social_network_neighbors:
                peer_wearing = sum(1 for nid in person.social_network_neighbors if self.people.get(nid, person).wearing_mask) / max(1, len(person.social_network_neighbors))

            # If out of masks, try to purchase before decision
            if person.masks_inventory <= 0:
                _, shortage = person.purchase_masks(self.retailer, max_purchase=2)
                if shortage:
                    shortage_incidents += 1

            policy_strength = 1.0 if self.authority.mandate_on else 0.0
            decision = person.decide_to_wear_mask(
                peer_adoption_rate=peer_wearing,
                observed_public_wearing=observed_public_wearing,
                policy_strength=policy_strength,
                enforcement_probability=avg_enforcement,
                market_availability=self.mask_availability_initial,
                price_per_mask=self.retailer.price_per_mask,
                weights=self.weights,
                rng=self.rng,
            )

            # Consume a mask if wearing
            if decision and person.masks_inventory > 0:
                person.masks_inventory -= 1
                person.wearing_mask = True
            else:
                person.wearing_mask = False

            person.respond_to_policy(self.compliance_decay_rate)
            new_wearing_flags[pid] = person.wearing_mask

        # Households share norms and pool masks
        for hh in self.households.values():
            hh.share_norms(self.people)
            hh.pool_resources_for_masks(self.people)

        # Simple enforcement pass at locations: select a small sample of visitors
        enforcement_actions = 0
        location_types = list({loc.type for loc in self.locations})
        # Assign random visits and enforce
        for loc in self.locations:
            # sample visitors proportional to foot traffic
            visitors: List[Person] = []
            for person in self.people.values():
                if person.daily_mobility_profile.get(loc.type, 0.0) > 0 and self.rng.random() < (0.05 * loc.foot_traffic_rate):
                    visitors.append(person)
                    if len(visitors) >= loc.capacity:
                        break
            if visitors:
                enforcement_actions += loc.enforce_mask_policy(visitors, self.authority, self.rng)

        # Metrics
        adoption_rate = sum(1 for v in new_wearing_flags if v) / max(1, len(new_wearing_flags))
        churn = sum(1 for prev, cur in zip(self._last_wearing_snapshot, new_wearing_flags) if prev != cur) / max(1, len(new_wearing_flags))
        compliance = adoption_rate if self.authority.mandate_on else adoption_rate  # placeholder

        self.metrics["adoption_rate_over_time"].append(adoption_rate)
        self.metrics["compliance_over_time"].append(compliance)
        self.metrics["churn_over_time"].append(churn)
        self.metrics["mask_shortage_incidents"] += int(shortage_incidents)
        self.metrics["enforcement_actions_count"] += int(enforcement_actions)
        self.retailer_inventory_history.append(self.retailer.inventory_level)
        self._last_wearing_snapshot = new_wearing_flags

    def run(self) -> Dict[str, Any]:
        for day in range(self.time_horizon_days):
            self.step(day)
        return {
            "metrics": self.metrics,
            "retailer_inventory_history": self.retailer_inventory_history,
            "time_horizon_days": self.time_horizon_days,
            "population_size": self.population_size,
        }


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--params', type=str, default=None, help='Path to JSON params or JSON string')
        args, _ = parser.parse_known_args()
        # Load params
        params: Dict[str, Any] = {}
        if args.params:
            if os.path.exists(args.params):
                with open(args.params, 'r') as f:
                    params = json.load(f)
            else:
                params = json.loads(args.params)
        else:
            raw = sys.stdin.read().strip()
            if raw:
                params = json.loads(raw)

        # Baseline run
        sim = Simulation(params)
        result = sim.run()

        # Counterfactual for policy_effect_size: mandate off (by starting far in the future)
        cf_params = dict(params)
        cf_params['mandate_start_day'] = int(cf_params.get('time_horizon_days', sim.time_horizon_days)) + 1000
        cf_params['random_seed'] = int(params.get('random_seed', 42))
        cf_sim = Simulation(cf_params)
        cf_result = cf_sim.run()

        series = result['metrics']['adoption_rate_over_time']
        cf_series = cf_result['metrics']['adoption_rate_over_time']
        horizon = min(len(series), len(cf_series))
        auc_diff = sum(series[:horizon]) - sum(cf_series[:horizon])
        result['metrics']['policy_effect_size'] = auc_diff / max(1, horizon)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))


# Execute main for both direct execution and sandbox wrapper invocation

# Execute main for both direct execution and sandbox wrapper invocation
main()