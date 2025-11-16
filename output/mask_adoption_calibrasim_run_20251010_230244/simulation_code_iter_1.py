def main():
    pass

import os
import json
import math
import random
from typing import Dict, List, Any, Tuple, Optional, Deque
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import networkx as nx


# Path handling per instruction
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.getcwd()
DATA_PATH = os.environ.get("DATA_PATH") or "data_fitting/mask_adoption_data"
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a numeric value between lower and upper bounds.
    """
    return max(lo, min(hi, x))


def bernoulli(p: float) -> bool:
    """
    Draw a Bernoulli random variable.
    """
    p = clamp(p, 0.0, 1.0)
    return random.random() < p


def logistic(z: float, temperature: float = 1.0) -> float:
    """
    Temperature-scaled logistic function.
    """
    t = max(1e-6, float(temperature))
    try:
        return 1.0 / (1.0 + math.exp(-z / t))
    except OverflowError:
        if z > 0:
            return 1.0
        else:
            return 0.0


# Model plan embedded for dynamic configuration and orchestration
MODEL_PLAN: Dict[str, Any] = {
    "model_type": "agent_based",
    "description": "Multi-agent, multiplex-network simulation of mask-wearing behavior diffusion via social influence, policy intervention (from Day 10), information propagation, and market supply constraints. Calibrated on the first 30 days; predicts days 30–39.",
    "entities": [
        {
            "name": "PersonAgent",
            "attributes": [
                "id",
                "age",
                "occupation",
                "socioeconomic_status",
                "trust_in_authorities",
                "risk_perception",
                "network_neighbors",
                "adoption_level",
                "propensity_to_adopt",
                "sensitivity_to_social_influence",
                "sensitivity_to_policy",
                "sensitivity_to_cost",
                "fatigue_decay_rate",
                "info_exposure_level",
                "compliance_probability",
                "last_adoption_change_day",
                "household_id",
                "workplace_id",
            ],
            "behaviors": [
                "observe_neighbors",
                "receive_information",
                "evaluate_adoption_decision",
                "respond_to_policy",
                "purchase_masks",
                "share_opinion",
                "update_risk_perception",
            ],
            "initialization": {
                "method": "data_driven",
                "parameters": {
                    "source_agent_attributes": "data_fitting/mask_adoption_data/agent_attributes.csv",
                    "source_network": "data_fitting/mask_adoption_data/social_network.json",
                    "init_adoption_rate_source": "data_fitting/mask_adoption_data/train_data.csv",
                    "fallback_initial_adoption_rate": 0.05,
                    "impute_trust_if_missing_beta": [2.0, 2.0],
                    "impute_ses_if_missing_normal": {"mean": 0.0, "std": 1.0},
                    "risk_perception_bounds": [0.0, 1.0],
                },
            },
        },
        {
            "name": "PolicyEnvironment",
            "attributes": [
                "mandate_level",
                "enforcement_probability",
                "fine_amount",
                "communication_strength",
                "mask_subsidy_level",
                "active_start_day",
                "active_end_day",
            ],
            "behaviors": [
                "issue_or_update_mandate",
                "adjust_enforcement",
                "run_public_campaign",
                "set_subsidies",
            ],
            "initialization": {
                "method": "specified",
                "parameters": {
                    "active_start_day": 10,
                    "active_end_day": 120,
                    "mandate_level": 0.7,
                    "enforcement_probability": 0.2,
                    "fine_amount": 50.0,
                    "communication_strength": 0.6,
                    "mask_subsidy_level": 0.3,
                },
            },
        },
        {
            "name": "SupplyMarket",
            "attributes": [
                "inventory_level",
                "production_rate_per_day",
                "distribution_delay_days",
                "mask_price",
                "pending_orders_queue",
            ],
            "behaviors": ["update_inventory", "adjust_price", "fulfill_demand"],
            "initialization": {
                "method": "specified",
                "parameters": {
                    "inventory_level": 100000,
                    "production_rate_per_day": 10000,
                    "distribution_delay_days": 2,
                    "mask_price": 1.0,
                },
            },
        },
        {
            "name": "InformationSource",
            "attributes": [
                "message_frequency",
                "message_slant",
                "credibility",
                "reach",
            ],
            "behaviors": ["broadcast_message", "target_audience_selection"],
            "initialization": {
                "method": "specified",
                "parameters": {
                    "message_frequency": 1.0,
                    "message_slant": "pro_mask",
                    "credibility": 0.8,
                    "reach": 0.4,
                },
            },
        },
    ],
    "modules": [
        {
            "name": "PolicyEnvironment",
            "description": "Activates and updates policy signals (mandate, enforcement, communication, subsidies) from Day 10 onward.",
            "inputs": ["time.day"],
            "outputs": ["signal.policy_state"],
            "dependencies": [],
            "tick_rate": {"unit": "days", "value": 1},
            "update_rule": {
                "pseudocode": "if day >= policy_active_start_day and day <= policy_active_end_day:\n  policy_state.mandate_level = mandate_level\n  policy_state.enforcement_probability = enforcement_probability\n  policy_state.fine_amount = fine_amount\n  policy_state.communication_strength = communication_strength\n  policy_state.mask_subsidy_level = mask_subsidy_level\nelse:\n  policy_state.mandate_level = 0.0\n  policy_state.enforcement_probability = 0.0\n  policy_state.fine_amount = 0.0\n  policy_state.communication_strength = 0.0\n  policy_state.mask_subsidy_level = 0.0\nemit(signal.policy_state)",
                "notes": [
                    "Simple step function policy; can be extended to ramps or time series.",
                    "Policy affects both adoption (mandate) and information (communication) and supply (subsidies).",
                ],
            },
            "requirements": ["entities.PolicyEnvironment"],
            "module_parameters": [
                "policy_active_start_day",
                "policy_active_end_day",
                "mandate_level",
                "enforcement_probability",
                "fine_amount",
                "communication_strength",
                "mask_subsidy_level",
            ],
        },
        {
            "name": "InformationDiffusion",
            "description": "Updates agents' info_exposure_level via exogenous broadcast and peer social sharing on the multiplex network.",
            "inputs": [
                "signal.policy_state",
                "state.agent_info",
                "graph.family",
                "graph.work",
                "graph.community",
                "state.agent_mask",
            ],
            "outputs": ["state.agent_info"],
            "dependencies": ["PolicyEnvironment"],
            "tick_rate": {"unit": "days", "value": 1},
            "update_rule": {
                "pseudocode": "for each agent i:\n  # Broadcast (exogenous) info\n  b = exogenous_info_rate * info_broadcast_reach * info_broadcast_credibility * (1.0 + signal.policy_state.communication_strength)\n  # Peer sharing from neighbors who are currently wearing masks (as proxy for sharing pro-mask info)\n  peer_family = 1.0 - prod_{j in N_family(i)} (1.0 - share_probability * peer_info_rate_family * 1[state.agent_mask[j]==1])\n  peer_work   = 1.0 - prod_{j in N_work(i)}   (1.0 - share_probability * peer_info_rate_work   * 1[state.agent_mask[j]==1])\n  peer_comm   = 1.0 - prod_{j in N_comm(i)}   (1.0 - share_probability * peer_info_rate_comm   * 1[state.agent_mask[j]==1])\n  peer_total = 1.0 - (1.0 - peer_family)*(1.0 - peer_work)*(1.0 - peer_comm)\n  # Decay and update exposure level\n  new_info = (1.0 - info_decay_rate) * state.agent_info[i] + b + peer_total\n  state.agent_info[i] = min(1.0, max(0.0, new_info))\nemit(state.agent_info)",
                "notes": [
                    "Peer information uses complement-of-products to combine independent transmission channels.",
                    "Using mask-wearing neighbors to approximate pro-mask sharing directionally aligns info and behavior.",
                ],
            },
            "requirements": [
                "entities.PersonAgent",
                "entities.InformationSource",
                "environment.network",
            ],
            "module_parameters": [
                "exogenous_info_rate",
                "info_broadcast_credibility",
                "info_broadcast_reach",
                "peer_info_rate_family",
                "peer_info_rate_work",
                "peer_info_rate_comm",
                "info_decay_rate",
                "share_probability",
            ],
        },
        {
            "name": "NetworkStats",
            "description": "Computes per-layer neighbor mask fractions with observation noise.",
            "inputs": ["state.agent_mask", "graph.family", "graph.work", "graph.community"],
            "outputs": ["signal.neighbor_mask_fractions"],
            "dependencies": [],
            "tick_rate": {"unit": "days", "value": 1},
            "update_rule": {
                "pseudocode": "for each agent i:\n  for L in {family, work, community}:\n    deg = |N_L(i)|\n    frac = 0.0 if deg==0 else (sum_{j in N_L(i)} state.agent_mask[j]) / deg\n    noisy_frac = clamp(frac + Normal(0, neighbor_observation_noise_sigma), 0.0, 1.0)\n    neighbor_mask_fractions[i][L] = noisy_frac\nemit(signal.neighbor_mask_fractions)",
                "notes": [
                    "Observation noise models imperfect perception of neighbors' behavior.",
                    "Edges are treated as undirected simple graphs after preprocessing.",
                ],
            },
            "requirements": ["environment.network"],
            "module_parameters": ["neighbor_observation_noise_sigma"],
        },
        {
            "name": "SupplyMarket",
            "description": "Tracks inventory, sets daily mask price, and fulfills queued orders after a distribution delay.",
            "inputs": ["signal.policy_state", "signal.pending_orders"],
            "outputs": ["signal.mask_price", "signal.deliveries_due_today"],
            "dependencies": ["PolicyEnvironment"],
            "tick_rate": {"unit": "days", "value": 1},
            "update_rule": {
                "pseudocode": "# Update inventory with production\ninventory_level += production_rate_per_day\n# Price adjusts toward base with markup during shortages and subsidy reductions\nsubsidy = signal.policy_state.mask_subsidy_level\neffective_base = max(min_mask_price, mask_price_base * (1.0 - subsidy_pass_through * subsidy))\n# Compute backlog demand scheduled to be delivered today\ndeliveries_due_today = pop_orders_from_queue(day)\n# Fulfill deliveries due today (inventory reduces now)\nallocated_today = min(inventory_level, sum(deliveries_due_today.quantities))\nallocation_ratio = 0.0 if sum(deliveries_due_today.quantities)==0 else allocated_today / sum(deliveries_due_today.quantities)\nfor each order in deliveries_due_today:\n  deliver_qty = floor(order.quantity * allocation_ratio)\n  emit_delivery(order.agent_id, deliver_qty)\ninventory_level -= allocated_today\n# Observe new pending orders (placed today) and push to queue with delivery_day = day + distribution_delay_days\nfor order in signal.pending_orders:\n  push_to_queue(delivery_day=day+distribution_delay_days, agent_id=order.agent_id, quantity=order.quantity)\n# Update price for tomorrow based on stock tightness\nstock_pressure = max(0.0, 1.0 - inventory_level / (production_rate_per_day * (distribution_delay_days + 1)))\nmask_price = clamp(effective_base * (1.0 + shortage_markup_beta * stock_pressure), min_mask_price, max_mask_price)\nemit(signal.mask_price, mask_price)\nemit(signal.deliveries_due_today)",
                "notes": [
                    "Orders placed today are fulfilled after distribution_delay_days.",
                    "Subsidies reduce the effective base price through subsidy_pass_through.",
                    "Proportional allocation ensures fairness under shortages.",
                ],
            },
            "requirements": ["entities.SupplyMarket"],
            "module_parameters": [
                "supply_initial_inventory",
                "production_rate_per_day",
                "distribution_delay_days",
                "mask_price_base",
                "price_adjustment_alpha",
                "shortage_markup_beta",
                "min_mask_price",
                "max_mask_price",
                "subsidy_pass_through",
            ],
        },
        {
            "name": "AdoptionDecision",
            "description": "Computes adoption intentions, places mask orders, and updates wearing states upon deliveries. Uses logistic adoption with persistence, social influence, information, policy, and cost components.",
            "inputs": [
                "signal.neighbor_mask_fractions",
                "state.agent_info",
                "signal.policy_state",
                "signal.mask_price",
                "signal.deliveries_due_today",
                "time.day",
            ],
            "outputs": ["state.agent_mask", "signal.pending_orders"],
            "dependencies": [
                "NetworkStats",
                "InformationDiffusion",
                "PolicyEnvironment",
                "SupplyMarket",
            ],
            "tick_rate": {"unit": "days", "value": 1},
            "update_rule": {
                "pseudocode": "# 1) Process deliveries arriving today\nfor each delivery in signal.deliveries_due_today:\n  if delivery.quantity > 0:\n    state.agent_mask[delivery.agent_id] = 1\n    agents[delivery.agent_id].last_adoption_change_day = day\n# 2) Compute adoption intentions and place orders\npending_orders = []\nfor each agent i:\n  wear_t = state.agent_mask[i]\n  fF = signal.neighbor_mask_fractions[i].family\n  fW = signal.neighbor_mask_fractions[i].work\n  fC = signal.neighbor_mask_fractions[i].community\n  info = state.agent_info[i]\n  policy = signal.policy_state.mandate_level\n  price = signal.mask_price\n  # Weighted peer norm\n  peer_term = beta_peer_family * w_family * fF + beta_peer_work * w_work * fW + beta_peer_comm * w_comm * fC\n  # Persistence boosts adoption if already wearing\n  persistence_term = beta_persistence * wear_t\n  # Cost term (negative)\n  cost_term = beta_cost * price\n  # Policy term (positive)\n  policy_term = beta_policy * policy\n  # Risk and information effects\n  risk_term = beta_risk * agents[i].risk_perception\n  info_term = beta_info * info\n  # Fatigue reduces propensity over time since last change\n  days_since_change = max(0, day - agents[i].last_adoption_change_day)\n  fatigue_term = -fatigue_decay_rate * days_since_change\n  # Logit score and probability with temperature scaling\n  z = beta0_intercept + peer_term + persistence_term + cost_term + policy_term + risk_term + info_term + fatigue_term\n  p_adopt = 1.0 / (1.0 + exp(-z / max(1e-6, noise_temperature)))\n  # Bound daily change to avoid oscillations\n  if wear_t == 0:\n    adopt_prob = min(max_daily_adoption_change, p_adopt)\n    will_adopt = Bernoulli(adopt_prob)\n    if will_adopt:\n      pending_orders.append({agent_id: i, quantity: 1})\n  else:\n    # Consider disadoption via low propensity (implicit via not placing orders and optionally drop if z very low)\n    p_keep = max(p_adopt, 1.0 - max_daily_adoption_change)\n    drop = Bernoulli(max(0.0, 1.0 - p_keep))\n    if drop:\n      state.agent_mask[i] = 0\n      agents[i].last_adoption_change_day = day\nemit(signal.pending_orders, pending_orders)\nemit(state.agent_mask)",
                "notes": [
                    "Adoption is intention-based with a delivery lag realized via SupplyMarket.",
                    "Peer influence is layer-weighted; parameters can be calibrated.",
                    "Disadoption occurs when overall propensity is low; constrained to avoid unrealistic churn.",
                ],
            },
            "requirements": ["entities.PersonAgent", "environment.network", "randomness"],
            "module_parameters": [
                "beta0_intercept",
                "beta_peer_family",
                "beta_peer_work",
                "beta_peer_comm",
                "beta_info",
                "beta_risk",
                "beta_policy",
                "beta_cost",
                "beta_persistence",
                "fatigue_decay_rate",
                "noise_temperature",
                "w_family",
                "w_work",
                "w_comm",
                "max_daily_adoption_change",
            ],
        },
        {
            "name": "AdoptionAggregator",
            "description": "Aggregates daily adoption rate for evaluation and logging.",
            "inputs": ["state.agent_mask", "state.agent_metadata"],
            "outputs": ["observable.adoption_rate_daily"],
            "dependencies": ["AdoptionDecision"],
            "tick_rate": {"unit": "days", "value": 1},
            "update_rule": {
                "pseudocode": "rate = mean_i(state.agent_mask[i])\nemit(observable.adoption_rate_daily, rate)",
                "notes": [
                    "Can be extended to subgroup aggregation by age or occupation.",
                    "No smoothing by default; controlled via smoothing_window_days if enabled.",
                ],
            },
            "requirements": ["entities.PersonAgent"],
            "module_parameters": ["smoothing_window_days"],
        },
    ],
    "environment": {
        "type": "network",
        "dimensions": None,
        "time_step": 1,
        "time_unit": "days",
    },
    "parameters": [
        {
            "key": "time_horizon_days",
            "dtype": "int",
            "default": 120,
            "bounds": {"low": 40, "high": 365},
            "owner_module": "global",
            "description": "Total simulation days.",
            "frozen": "true",
        },
        {
            "key": "time_step_days",
            "dtype": "int",
            "default": 1,
            "bounds": {"low": 1, "high": 1},
            "owner_module": "global",
            "description": "Time step in days.",
            "frozen": "true",
        },
        {
            "key": "seed",
            "dtype": "int",
            "default": 42,
            "bounds": {"low": 0, "high": 2147483647},
            "owner_module": "global",
            "description": "Random seed for reproducibility.",
            "frozen": "true",
        },
        {
            "key": "initial_adoption_rate",
            "dtype": "float",
            "default": 0.05,
            "bounds": {"low": 0.0, "high": 0.5},
            "owner_module": "global",
            "description": "Fallback initial share of mask wearers if not provided in data.",
            "frozen": "false",
        },
        {
            "key": "policy_active_start_day",
            "dtype": "int",
            "default": 10,
            "bounds": {"low": 10, "high": 10},
            "owner_module": "PolicyEnvironment",
            "description": "Day policy becomes active.",
            "frozen": "true",
        },
        {
            "key": "policy_active_end_day",
            "dtype": "int",
            "default": 120,
            "bounds": {"low": 10, "high": 120},
            "owner_module": "PolicyEnvironment",
            "description": "Day policy ends.",
            "frozen": "true",
        },
        {
            "key": "mandate_level",
            "dtype": "float",
            "default": 0.7,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "PolicyEnvironment",
            "description": "Strength of mandate signal.",
            "frozen": "false",
        },
        {
            "key": "enforcement_probability",
            "dtype": "float",
            "default": 0.2,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "PolicyEnvironment",
            "description": "Daily chance an agent faces enforcement if non-compliant.",
            "frozen": "false",
        },
        {
            "key": "fine_amount",
            "dtype": "float",
            "default": 50.0,
            "bounds": {"low": 0.0, "high": 500.0},
            "owner_module": "PolicyEnvironment",
            "description": "Fine for non-compliance (relative cost units).",
            "frozen": "false",
        },
        {
            "key": "communication_strength",
            "dtype": "float",
            "default": 0.6,
            "bounds": {"low": 0.0, "high": 2.0},
            "owner_module": "PolicyEnvironment",
            "description": "Amplification of public campaigns affecting info diffusion.",
            "frozen": "false",
        },
        {
            "key": "mask_subsidy_level",
            "dtype": "float",
            "default": 0.3,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "PolicyEnvironment",
            "description": "Subsidy as fraction of mask price.",
            "frozen": "false",
        },
        {
            "key": "supply_initial_inventory",
            "dtype": "int",
            "default": 100000,
            "bounds": {"low": 0, "high": 1000000},
            "owner_module": "SupplyMarket",
            "description": "Initial stock of masks in market.",
            "frozen": "false",
        },
        {
            "key": "production_rate_per_day",
            "dtype": "int",
            "default": 10000,
            "bounds": {"low": 0, "high": 100000},
            "owner_module": "SupplyMarket",
            "description": "Daily production inflow to inventory.",
            "frozen": "false",
        },
        {
            "key": "distribution_delay_days",
            "dtype": "int",
            "default": 2,
            "bounds": {"low": 0, "high": 14},
            "owner_module": "SupplyMarket",
            "description": "Days between order and delivery.",
            "frozen": "false",
        },
        {
            "key": "mask_price_base",
            "dtype": "float",
            "default": 1.0,
            "bounds": {"low": 0.1, "high": 5.0},
            "owner_module": "SupplyMarket",
            "description": "Base price absent shortages and subsidies.",
            "frozen": "false",
        },
        {
            "key": "price_adjustment_alpha",
            "dtype": "float",
            "default": 0.05,
            "bounds": {"low": 0.0, "high": 0.2},
            "owner_module": "SupplyMarket",
            "description": "Speed of price adjustment toward target; used implicitly in pricing dynamics.",
            "frozen": "false",
        },
        {
            "key": "shortage_markup_beta",
            "dtype": "float",
            "default": 0.5,
            "bounds": {"low": 0.0, "high": 2.0},
            "owner_module": "SupplyMarket",
            "description": "Price multiplier factor under stock pressure.",
            "frozen": "false",
        },
        {
            "key": "min_mask_price",
            "dtype": "float",
            "default": 0.5,
            "bounds": {"low": 0.1, "high": 1.0},
            "owner_module": "SupplyMarket",
            "description": "Lower bound on price (e.g., production cost).",
            "frozen": "false",
        },
        {
            "key": "max_mask_price",
            "dtype": "float",
            "default": 3.0,
            "bounds": {"low": 1.0, "high": 10.0},
            "owner_module": "SupplyMarket",
            "description": "Upper bound on price (e.g., price caps).",
            "frozen": "false",
        },
        {
            "key": "subsidy_pass_through",
            "dtype": "float",
            "default": 0.8,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "SupplyMarket",
            "description": "Fraction of subsidy that reduces consumer price.",
            "frozen": "false",
        },
        {
            "key": "exogenous_info_rate",
            "dtype": "float",
            "default": 0.02,
            "bounds": {"low": 0.0, "high": 0.2},
            "owner_module": "InformationDiffusion",
            "description": "Daily base probability of receiving info from exogenous sources.",
            "frozen": "false",
        },
        {
            "key": "info_broadcast_credibility",
            "dtype": "float",
            "default": 0.8,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "InformationDiffusion",
            "description": "Effectiveness scaling of broadcast messages.",
            "frozen": "false",
        },
        {
            "key": "info_broadcast_reach",
            "dtype": "float",
            "default": 0.4,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "InformationDiffusion",
            "description": "Audience fraction potentially reached by broadcast per day (used multiplicatively).",
            "frozen": "false",
        },
        {
            "key": "peer_info_rate_family",
            "dtype": "float",
            "default": 0.12,
            "bounds": {"low": 0.0, "high": 0.5},
            "owner_module": "InformationDiffusion",
            "description": "Per-day info transmission rate per masked family neighbor.",
            "frozen": "false",
        },
        {
            "key": "peer_info_rate_work",
            "dtype": "float",
            "default": 0.08,
            "bounds": {"low": 0.0, "high": 0.5},
            "owner_module": "InformationDiffusion",
            "description": "Per-day info transmission rate per masked work/school neighbor.",
            "frozen": "false",
        },
        {
            "key": "peer_info_rate_comm",
            "dtype": "float",
            "default": 0.04,
            "bounds": {"low": 0.0, "high": 0.5},
            "owner_module": "InformationDiffusion",
            "description": "Per-day info transmission rate per masked community neighbor.",
            "frozen": "false",
        },
        {
            "key": "info_decay_rate",
            "dtype": "float",
            "default": 0.05,
            "bounds": {"low": 0.0, "high": 0.5},
            "owner_module": "InformationDiffusion",
            "description": "Daily decay of info exposure level.",
            "frozen": "false",
        },
        {
            "key": "share_probability",
            "dtype": "float",
            "default": 0.5,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "InformationDiffusion",
            "description": "Probability a masked agent shares pro-mask info with a contact per day.",
            "frozen": "false",
        },
        {
            "key": "neighbor_observation_noise_sigma",
            "dtype": "float",
            "default": 0.02,
            "bounds": {"low": 0.0, "high": 0.2},
            "owner_module": "NetworkStats",
            "description": "Std dev of noise in perceived neighbor mask fraction.",
            "frozen": "false",
        },
        {
            "key": "beta0_intercept",
            "dtype": "float",
            "default": -2.944439,
            "bounds": {"low": -5.0, "high": 0.0},
            "owner_module": "AdoptionDecision",
            "description": "Intercept of logistic adoption model (approx 5% baseline).",
            "frozen": "false",
        },
        {
            "key": "beta_peer_family",
            "dtype": "float",
            "default": 1.2,
            "bounds": {"low": 0.0, "high": 3.0},
            "owner_module": "AdoptionDecision",
            "description": "Sensitivity to family neighbors' mask fraction.",
            "frozen": "false",
        },
        {
            "key": "beta_peer_work",
            "dtype": "float",
            "default": 0.8,
            "bounds": {"low": 0.0, "high": 3.0},
            "owner_module": "AdoptionDecision",
            "description": "Sensitivity to work/school neighbors' mask fraction.",
            "frozen": "false",
        },
        {
            "key": "beta_peer_comm",
            "dtype": "float",
            "default": 0.4,
            "bounds": {"low": 0.0, "high": 3.0},
            "owner_module": "AdoptionDecision",
            "description": "Sensitivity to community neighbors' mask fraction.",
            "frozen": "false",
        },
        {
            "key": "beta_info",
            "dtype": "float",
            "default": 1.0,
            "bounds": {"low": 0.0, "high": 3.0},
            "owner_module": "AdoptionDecision",
            "description": "Sensitivity to information exposure.",
            "frozen": "false",
        },
        {
            "key": "beta_risk",
            "dtype": "float",
            "default": 1.0,
            "bounds": {"low": 0.0, "high": 3.0},
            "owner_module": "AdoptionDecision",
            "description": "Sensitivity to personal risk perception.",
            "frozen": "false",
        },
        {
            "key": "beta_policy",
            "dtype": "float",
            "default": 1.5,
            "bounds": {"low": 0.0, "high": 5.0},
            "owner_module": "AdoptionDecision",
            "description": "Sensitivity to mandate level.",
            "frozen": "false",
        },
        {
            "key": "beta_cost",
            "dtype": "float",
            "default": -0.7,
            "bounds": {"low": -5.0, "high": 0.0},
            "owner_module": "AdoptionDecision",
            "description": "Price sensitivity (negative).",
            "frozen": "false",
        },
        {
            "key": "beta_persistence",
            "dtype": "float",
            "default": 1.2,
            "bounds": {"low": 0.0, "high": 5.0},
            "owner_module": "AdoptionDecision",
            "description": "Habit/inertia effect when already wearing.",
            "frozen": "false",
        },
        {
            "key": "fatigue_decay_rate",
            "dtype": "float",
            "default": 0.01,
            "bounds": {"low": 0.0, "high": 0.1},
            "owner_module": "AdoptionDecision",
            "description": "Daily decay in adoption propensity due to fatigue since last change.",
            "frozen": "false",
        },
        {
            "key": "noise_temperature",
            "dtype": "float",
            "default": 1.0,
            "bounds": {"low": 0.1, "high": 5.0},
            "owner_module": "AdoptionDecision",
            "description": "Scale of stochasticity in logistic decision.",
            "frozen": "false",
        },
        {
            "key": "w_family",
            "dtype": "float",
            "default": 0.5,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "AdoptionDecision",
            "description": "Weight for family layer in norm computation.",
            "frozen": "false",
        },
        {
            "key": "w_work",
            "dtype": "float",
            "default": 0.3,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "AdoptionDecision",
            "description": "Weight for work/school layer in norm computation.",
            "frozen": "false",
        },
        {
            "key": "w_comm",
            "dtype": "float",
            "default": 0.2,
            "bounds": {"low": 0.0, "high": 1.0},
            "owner_module": "AdoptionDecision",
            "description": "Weight for community layer in norm computation.",
            "frozen": "false",
        },
        {
            "key": "max_daily_adoption_change",
            "dtype": "float",
            "default": 0.25,
            "bounds": {"low": 0.05, "high": 1.0},
            "owner_module": "AdoptionDecision",
            "description": "Upper bound on daily probability of switching from 0 to 1 to stabilize dynamics.",
            "frozen": "false",
        },
        {
            "key": "smoothing_window_days",
            "dtype": "int",
            "default": 1,
            "bounds": {"low": 1, "high": 7},
            "owner_module": "AdoptionAggregator",
            "description": "Window for smoothing aggregated adoption rate (1 disables smoothing).",
            "frozen": "true",
        },
    ],
    "observables": [
        {
            "id": "observable.adoption_rate_daily",
            "source_module": "AdoptionAggregator",
            "formula_or_field": "mean(PersonAgent.adoption_level)",
            "frequency": "daily",
            "aggregation": "mean",
            "target_data_field": "train_data.csv:adoption_rate",
        }
    ],
    "metrics": [
        {
            "name": "RMSE_of_adoption",
            "inputs": ["observable.adoption_rate_daily"],
            "definition": {"type": "RMSE", "config": {"normalize": True, "window": "train"}},
        },
        {
            "name": "MAE_of_adoption",
            "inputs": ["observable.adoption_rate_daily"],
            "definition": {"type": "MAE", "config": {"normalize": True, "window": "train"}},
        },
        {
            "name": "R2_of_adoption",
            "inputs": ["observable.adoption_rate_daily"],
            "definition": {"type": "R2", "config": {"window": "train"}},
        },
        {
            "name": "Peak_timing_error_days",
            "inputs": ["observable.adoption_rate_daily"],
            "definition": {"type": "PeakTimingError", "config": {"window": "train"}},
        },
        {
            "name": "Final_adoption_gap",
            "inputs": ["observable.adoption_rate_daily"],
            "definition": {"type": "FinalGap", "config": {"window": "train"}},
        },
    ],
    "prediction_period": {"start_day": 30, "end_day": 39},
}


class PersonAgent:
    """
    Represents an individual agent in the simulation with attributes and behavioral state.
    """

    def __init__(
        self,
        agent_id: int,
        age: int = 0,
        occupation: str = "",
        socioeconomic_status: float = 0.0,
        trust_in_authorities: float = 0.5,
        risk_perception: float = 0.5,
        adoption_level: int = 0,
        info_exposure_level: float = 0.0,
        last_adoption_change_day: int = 0,
        household_id: Optional[int] = None,
        workplace_id: Optional[int] = None,
    ) -> None:
        """
        Initialize a PersonAgent with demographic and behavioral attributes.
        """
        self.id = int(agent_id)
        self.age = int(age)
        self.occupation = str(occupation)
        self.socioeconomic_status = float(socioeconomic_status)
        self.trust_in_authorities = clamp(float(trust_in_authorities), 0.0, 1.0)
        self.risk_perception = clamp(float(risk_perception), 0.0, 1.0)
        self.adoption_level = int(adoption_level)
        self.info_exposure_level = clamp(float(info_exposure_level), 0.0, 1.0)
        self.last_adoption_change_day = int(last_adoption_change_day)
        self.household_id = household_id
        self.workplace_id = workplace_id

    def observe_neighbors(self) -> None:
        pass

    def receive_information(self) -> None:
        pass

    def evaluate_adoption_decision(self) -> None:
        pass

    def respond_to_policy(self) -> None:
        pass

    def purchase_masks(self) -> None:
        pass

    def share_opinion(self) -> None:
        pass

    def update_risk_perception(self) -> None:
        pass


class PolicyEnvironment:
    """
    Encapsulates policy-related parameters and active window for mandates and campaigns.
    """

    def __init__(
        self,
        mandate_level: float = 0.7,
        enforcement_probability: float = 0.2,
        fine_amount: float = 50.0,
        communication_strength: float = 0.6,
        mask_subsidy_level: float = 0.3,
        active_start_day: int = 10,
        active_end_day: int = 120,
    ) -> None:
        """
        Initialize PolicyEnvironment with specified values.
        """
        self.mandate_level = clamp(float(mandate_level), 0.0, 1.0)
        self.enforcement_probability = clamp(float(enforcement_probability), 0.0, 1.0)
        self.fine_amount = float(fine_amount)
        self.communication_strength = max(0.0, float(communication_strength))
        self.mask_subsidy_level = clamp(float(mask_subsidy_level), 0.0, 1.0)
        self.active_start_day = int(active_start_day)
        self.active_end_day = int(active_end_day)

    def as_state(self, day: int) -> Dict[str, float]:
        """
        Get the current policy state as a dictionary for the given day.
        """
        if self.active_start_day <= day <= self.active_end_day:
            return {
                "mandate_level": self.mandate_level,
                "enforcement_probability": self.enforcement_probability,
                "fine_amount": self.fine_amount,
                "communication_strength": self.communication_strength,
                "mask_subsidy_level": self.mask_subsidy_level,
            }
        else:
            return {
                "mandate_level": 0.0,
                "enforcement_probability": 0.0,
                "fine_amount": 0.0,
                "communication_strength": 0.0,
                "mask_subsidy_level": 0.0,
            }


class SupplyMarket:
    """
    Represents the supply-side market dynamics for mask inventory, pricing, and order fulfillment.
    """

    def __init__(
        self,
        inventory_level: float = 100000.0,
        production_rate_per_day: float = 10000.0,
        distribution_delay_days: int = 2,
        mask_price: float = 1.0,
        min_mask_price: float = 0.5,
        max_mask_price: float = 3.0,
        mask_price_base: float = 1.0,
        shortage_markup_beta: float = 0.5,
        subsidy_pass_through: float = 0.8,
        price_adjustment_alpha: float = 0.05,
    ) -> None:
        """
        Initialize SupplyMarket with inventory, production, pricing, and delivery parameters.
        """
        self.inventory_level = float(inventory_level)
        self.production_rate_per_day = float(production_rate_per_day)
        self.distribution_delay_days = int(distribution_delay_days)
        self.mask_price = float(mask_price)
        self.min_mask_price = float(min_mask_price)
        self.max_mask_price = float(max_mask_price)
        self.mask_price_base = float(mask_price_base)
        self.shortage_markup_beta = float(shortage_markup_beta)
        self.subsidy_pass_through = clamp(float(subsidy_pass_through), 0.0, 1.0)
        self.price_adjustment_alpha = clamp(float(price_adjustment_alpha), 0.0, 1.0)
        self.pending_orders_queue: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    def push_order(self, agent_id: int, quantity: int, delivery_day: int) -> None:
        """
        Add an order to the delivery queue.
        """
        quantity = max(0, int(quantity))
        if quantity > 0:
            self.pending_orders_queue[int(delivery_day)].append((int(agent_id), quantity))

    def pop_deliveries_due_today(self, day: int) -> List[Tuple[int, int]]:
        """
        Retrieve and remove deliveries scheduled for the current day.
        """
        due = self.pending_orders_queue.pop(int(day), [])
        return due

    def update_day(
        self,
        day: int,
        policy_state: Dict[str, float],
        pending_orders: List[Tuple[int, int]],
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Advance market by one day: produce inventory, deliver due orders proportionally, enqueue new orders, and update price.
        """
        # Production inflow
        self.inventory_level += self.production_rate_per_day

        # Deliveries due today
        deliveries_scheduled = self.pop_deliveries_due_today(day)
        total_due_qty = int(sum(q for _, q in deliveries_scheduled))
        inv_available = int(math.floor(self.inventory_level))
        allocated_today = min(inv_available, total_due_qty)

        deliveries_fulfilled: List[Tuple[int, int]] = []
        if total_due_qty > 0 and allocated_today > 0:
            # Largest remainder method to ensure sum delivered == allocated_today
            ratio = allocated_today / float(total_due_qty)
            raw_allocs = [qty * ratio for _, qty in deliveries_scheduled]
            floors = [int(math.floor(x)) for x in raw_allocs]
            remainder_units = allocated_today - sum(floors)
            fracs = [(i, raw_allocs[i] - floors[i]) for i in range(len(raw_allocs))]
            fracs.sort(key=lambda x: x[1], reverse=True)
            allocs = floors[:]
            for i in range(remainder_units):
                idx = fracs[i][0]
                if deliveries_scheduled[idx][1] > 0:
                    allocs[idx] += 1
            for (agent_id, _), deliver_qty in zip(deliveries_scheduled, allocs):
                if deliver_qty > 0:
                    deliveries_fulfilled.append((agent_id, deliver_qty))
            delivered_sum = sum(q for _, q in deliveries_fulfilled)
        else:
            delivered_sum = 0

        # Reduce inventory by delivered amount
        self.inventory_level -= delivered_sum

        # Price update based on subsidy and stock pressure (target price first)
        subsidy = policy_state.get("mask_subsidy_level", 0.0)
        effective_base = max(self.min_mask_price, self.mask_price_base * (1.0 - self.subsidy_pass_through * subsidy))
        denom = (self.production_rate_per_day * (self.distribution_delay_days + 1))
        stock_pressure = 0.0
        if denom > 0:
            stock_pressure = max(0.0, 1.0 - (self.inventory_level / denom))
        target_price = clamp(effective_base * (1.0 + self.shortage_markup_beta * stock_pressure), self.min_mask_price, self.max_mask_price)
        # Smooth adjustment
        self.mask_price = (1.0 - self.price_adjustment_alpha) * self.mask_price + self.price_adjustment_alpha * target_price

        # Enqueue pending orders for future delivery
        for agent_id, qty in pending_orders:
            self.push_order(agent_id=agent_id, quantity=qty, delivery_day=day + self.distribution_delay_days)

        return float(self.mask_price), deliveries_fulfilled


class InformationSource:
    """
    Represents an information source with broadcast capabilities.
    """

    def __init__(
        self,
        message_frequency: float = 1.0,
        message_slant: str = "pro_mask",
        credibility: float = 0.8,
        reach: float = 0.4,
    ) -> None:
        """
        Initialize the information source.
        """
        self.message_frequency = float(message_frequency)
        self.message_slant = str(message_slant)
        self.credibility = clamp(float(credibility), 0.0, 1.0)
        self.reach = clamp(float(reach), 0.0, 1.0)


class NetworkBuilder:
    """
    Builds and validates multiplex network layers (family, work, community) from JSON data or synthetic generation.
    """

    @staticmethod
    def enforce_symmetry(adj: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Enforce undirected simple graph symmetry on adjacency lists.
        """
        sym: Dict[int, set] = defaultdict(set)
        for u, nbrs in adj.items():
            for v in nbrs:
                if v == u:
                    continue
                sym[u].add(v)
                sym[v].add(u)
        # Ensure all nodes present
        for u in list(adj.keys()):
            sym.setdefault(u, set())
        return {u: sorted(list(vs)) for u, vs in sym.items()}

    def load_or_build_network(self, agent_ids: List[int], path: str) -> Dict[str, Dict[int, List[int]]]:
        """
        Load multiplex network layers from JSON file, or build synthetic layers if file is missing or invalid.
        """
        layers = {"family": defaultdict(list), "work": defaultdict(list), "community": defaultdict(list)}
        agent_id_set = set(agent_ids)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for k, v in raw.items():
                    try:
                        uid = int(k)
                    except Exception:
                        continue
                    if uid not in agent_id_set:
                        continue
                    fam = [int(x) for x in v.get("family", []) if int(x) in agent_id_set and int(x) != uid]
                    work = [int(x) for x in v.get("work_school", []) if int(x) in agent_id_set and int(x) != uid]
                    comm = [int(x) for x in v.get("community", []) if int(x) in agent_id_set and int(x) != uid]
                    layers["family"][uid] = fam
                    layers["work"][uid] = work
                    layers["community"][uid] = comm
                # Enforce symmetry
                for L in ["family", "work", "community"]:
                    layers[L] = self.enforce_symmetry(layers[L])
                # Ensure all agent_ids present
                for aid in agent_ids:
                    for L in ["family", "work", "community"]:
                        layers[L].setdefault(aid, [])
                return layers
            except Exception:
                # Fall back to synthetic
                pass

        # Synthetic small-world-like graphs using networkx Watts-Strogatz for each layer with varying k and p
        n = len(agent_ids)
        if n == 0:
            return {k: {} for k in ["family", "work", "community"]}

        index_to_id = {idx: aid for idx, aid in enumerate(agent_ids)}

        def build_ws(n_nodes: int, k: int, p: float) -> Dict[int, List[int]]:
            adj: Dict[int, List[int]] = defaultdict(list)
            if n_nodes < 3:
                # No edges for very small graphs
                for aid in agent_ids:
                    adj.setdefault(aid, [])
                return adj
            k = min(k, n_nodes - 1)
            if k % 2 == 1:
                k -= 1
            if k < 2:
                k = 2
            g = nx.watts_strogatz_graph(n_nodes, k=k, p=clamp(p, 0.0, 1.0), seed=42)
            for u, v in g.edges():
                adj[index_to_id[u]].append(index_to_id[v])
                adj[index_to_id[v]].append(index_to_id[u])
            # enforce symmetry and fill missing nodes
            adj = self.enforce_symmetry(adj)
            for aid in agent_ids:
                adj.setdefault(aid, [])
            return adj

        layers["family"] = build_ws(n, max(2, min(6, n - 1)), 0.05)
        layers["work"] = build_ws(n, max(2, min(8, n - 1)), 0.1)
        layers["community"] = build_ws(n, max(2, min(4, n - 1)), 0.2)
        return layers


class PolicyModule:
    """
    Module that emits the current policy state signals each day based on the PolicyEnvironment.
    """

    def update(self, day: int, policy_env: PolicyEnvironment) -> Dict[str, float]:
        return policy_env.as_state(day)


# Alias class name to align with plan naming
class PolicyEnvironmentModule(PolicyModule):
    pass


class InformationDiffusionModule:
    """
    Module implementing information diffusion via broadcasts and peer sharing across multiplex network.
    """

    def __init__(
        self,
        exogenous_info_rate: float,
        info_broadcast_credibility: float,
        info_broadcast_reach: float,
        peer_info_rate_family: float,
        peer_info_rate_work: float,
        peer_info_rate_comm: float,
        info_decay_rate: float,
        share_probability: float,
    ) -> None:
        self.exogenous_info_rate = float(exogenous_info_rate)
        self.info_broadcast_credibility = clamp(float(info_broadcast_credibility), 0.0, 1.0)
        self.info_broadcast_reach = clamp(float(info_broadcast_reach), 0.0, 1.0)
        self.peer_info_rate_family = clamp(float(peer_info_rate_family), 0.0, 1.0)
        self.peer_info_rate_work = clamp(float(peer_info_rate_work), 0.0, 1.0)
        self.peer_info_rate_comm = clamp(float(peer_info_rate_comm), 0.0, 1.0)
        self.info_decay_rate = clamp(float(info_decay_rate), 0.0, 1.0)
        self.share_probability = clamp(float(share_probability), 0.0, 1.0)

    def update(
        self,
        agents: Dict[int, PersonAgent],
        info_levels: Dict[int, float],
        mask_state: Dict[int, int],
        policy_state: Dict[str, float],
        layers: Dict[str, Dict[int, List[int]]],
    ) -> Dict[int, float]:
        updated = {}
        broadcast_amp = (1.0 + max(0.0, policy_state.get("communication_strength", 0.0)))
        base_broadcast = self.exogenous_info_rate * self.info_broadcast_reach * self.info_broadcast_credibility * broadcast_amp
        for aid in agents.keys():
            # Peer sharing via complement-of-products
            peer_terms = []
            for layer_name, rate in zip(
                ["family", "work", "community"],
                [self.peer_info_rate_family, self.peer_info_rate_work, self.peer_info_rate_comm],
            ):
                prod = 1.0
                for nbr in layers.get(layer_name, {}).get(aid, []):
                    if mask_state.get(nbr, 0) == 1:
                        prod *= (1.0 - self.share_probability * rate)
                peer_term = 1.0 - prod
                peer_terms.append(peer_term)
            peer_total = 1.0 - (1.0 - peer_terms[0]) * (1.0 - peer_terms[1]) * (1.0 - peer_terms[2])
            new_info = (1.0 - self.info_decay_rate) * float(info_levels.get(aid, 0.0)) + base_broadcast + peer_total
            updated[aid] = clamp(new_info, 0.0, 1.0)
        return updated


class NetworkStatsModule:
    """
    Module that computes neighbor mask fractions per layer with observation noise.
    """

    def __init__(self, neighbor_observation_noise_sigma: float, rng: np.random.Generator) -> None:
        self.noise_sigma = max(0.0, float(neighbor_observation_noise_sigma))
        self.rng = rng

    def update(
        self,
        agents: Dict[int, PersonAgent],
        mask_state: Dict[int, int],
        layers: Dict[str, Dict[int, List[int]]],
    ) -> Dict[int, Dict[str, float]]:
        out: Dict[int, Dict[str, float]] = {}
        for aid in agents.keys():
            per_layer = {}
            for L in ["family", "work", "community"]:
                nbrs = layers.get(L, {}).get(aid, [])
                deg = len(nbrs)
                frac = 0.0
                if deg > 0:
                    frac = sum(mask_state.get(nbr, 0) for nbr in nbrs) / float(deg)
                noisy = clamp(frac + float(self.rng.normal(0.0, self.noise_sigma)), 0.0, 1.0)
                per_layer[L] = noisy
            out[aid] = per_layer
        return out


class AdoptionDecisionModule:
    """
    Module that computes adoption decisions, places orders, and handles deliveries.
    """

    def __init__(
        self,
        beta0_intercept: float,
        beta_peer_family: float,
        beta_peer_work: float,
        beta_peer_comm: float,
        beta_info: float,
        beta_risk: float,
        beta_policy: float,
        beta_cost: float,
        beta_persistence: float,
        fatigue_decay_rate: float,
        noise_temperature: float,
        w_family: float,
        w_work: float,
        w_comm: float,
        max_daily_adoption_change: float,
        rng: np.random.Generator,
    ) -> None:
        self.beta0 = float(beta0_intercept)
        self.beta_peer_family = float(beta_peer_family)
        self.beta_peer_work = float(beta_peer_work)
        self.beta_peer_comm = float(beta_peer_comm)
        self.beta_info = float(beta_info)
        self.beta_risk = float(beta_risk)
        self.beta_policy = float(beta_policy)
        self.beta_cost = float(beta_cost)
        self.beta_persistence = float(beta_persistence)
        self.fatigue_decay_rate = max(0.0, float(fatigue_decay_rate))
        self.noise_temperature = max(1e-6, float(noise_temperature))
        self.w_family = clamp(float(w_family), 0.0, 1.0)
        self.w_work = clamp(float(w_work), 0.0, 1.0)
        self.w_comm = clamp(float(w_comm), 0.0, 1.0)
        self.max_daily_adoption_change = clamp(float(max_daily_adoption_change), 0.0, 1.0)
        self.rng = rng

    def update(
        self,
        day: int,
        agents: Dict[int, PersonAgent],
        mask_state: Dict[int, int],
        info_levels: Dict[int, float],
        neighbor_mask_fractions: Dict[int, Dict[str, float]],
        policy_state: Dict[str, float],
        mask_price: float,
        deliveries_due_today: List[Tuple[int, int]],
    ) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
        # Process deliveries to set mask state to 1
        for agent_id, delivered_qty in deliveries_due_today:
            if delivered_qty > 0:
                mask_state[agent_id] = 1
                agents[agent_id].last_adoption_change_day = day

        pending_orders: List[Tuple[int, int]] = []

        for aid, agent in agents.items():
            wear_t = int(mask_state.get(aid, 0))
            frac_family = neighbor_mask_fractions.get(aid, {}).get("family", 0.0)
            frac_work = neighbor_mask_fractions.get(aid, {}).get("work", 0.0)
            frac_comm = neighbor_mask_fractions.get(aid, {}).get("community", 0.0)
            info = float(info_levels.get(aid, 0.0))
            policy = float(policy_state.get("mandate_level", 0.0))
            price = float(mask_price)

            # Expected penalty for non-compliance (applies when not wearing)
            enforcement_probability = float(policy_state.get("enforcement_probability", 0.0))
            fine_amount = float(policy_state.get("fine_amount", 0.0))
            expected_penalty = enforcement_probability * fine_amount
            # Treat penalty as reducing the "effective cost" of adopting (since not adopting is costly)
            effective_cost = price - (expected_penalty if wear_t == 0 else 0.0)

            peer_term = (
                self.beta_peer_family * self.w_family * frac_family
                + self.beta_peer_work * self.w_work * frac_work
                + self.beta_peer_comm * self.w_comm * frac_comm
            )
            persistence_term = self.beta_persistence * wear_t
            cost_term = self.beta_cost * effective_cost
            policy_term = self.beta_policy * policy
            risk_term = self.beta_risk * agent.risk_perception
            info_term = self.beta_info * info
            days_since_change = max(0, day - agent.last_adoption_change_day)
            fatigue_term = -self.fatigue_decay_rate * days_since_change

            z = self.beta0 + peer_term + persistence_term + cost_term + policy_term + risk_term + info_term + fatigue_term
            p_adopt = logistic(z, self.noise_temperature)

            if wear_t == 0:
                adopt_prob = min(self.max_daily_adoption_change, p_adopt)
                if self.rng.random() < adopt_prob:
                    pending_orders.append((aid, 1))
            else:
                p_keep = max(p_adopt, 1.0 - self.max_daily_adoption_change)
                drop_prob = max(0.0, 1.0 - p_keep)
                if self.rng.random() < drop_prob:
                    mask_state[aid] = 0
                    agent.last_adoption_change_day = day

        return mask_state, pending_orders


class AdoptionAggregatorModule:
    """
    Module that aggregates daily adoption into a single observable timeseries.
    """

    def __init__(self, smoothing_window_days: int = 1) -> None:
        self.window = max(1, int(smoothing_window_days))
        self.history: Deque[float] = deque(maxlen=self.window)

    def update(self, mask_state: Dict[int, int]) -> float:
        total = len(mask_state)
        rate = 0.0
        if total > 0:
            rate = sum(mask_state.values()) / float(total)
        self.history.append(rate)
        return float(np.mean(self.history)) if self.window > 1 else rate


class Metrics:
    """
    Implements evaluation metrics for simulated adoption time series.
    """

    @staticmethod
    def rmse(y_true: List[float], y_pred: List[float]) -> float:
        n = min(len(y_true), len(y_pred))
        if n == 0:
            return float("nan")
        err = np.array(y_true[:n]) - np.array(y_pred[:n])
        return float(np.sqrt(np.mean(err ** 2)))

    @staticmethod
    def mae(y_true: List[float], y_pred: List[float]) -> float:
        n = min(len(y_true), len(y_pred))
        if n == 0:
            return float("nan")
        err = np.abs(np.array(y_true[:n]) - np.array(y_pred[:n]))
        return float(np.mean(err))

    @staticmethod
    def r2(y_true: List[float], y_pred: List[float]) -> float:
        n = min(len(y_true), len(y_pred))
        if n == 0:
            return float("nan")
        yt = np.array(y_true[:n])
        yp = np.array(y_pred[:n])
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        if ss_tot == 0.0:
            return float("nan")
        return float(1.0 - (ss_res / ss_tot))

    @staticmethod
    def peak_timing_error(y_true: List[float], y_pred: List[float]) -> float:
        n = min(len(y_true), len(y_pred))
        if n == 0:
            return float("nan")
        t_peak = int(np.argmax(y_true[:n]))
        p_peak = int(np.argmax(y_pred[:n]))
        return float(abs(t_peak - p_peak))

    @staticmethod
    def final_gap(y_true: List[float], y_pred: List[float]) -> float:
        n = min(len(y_true), len(y_pred))
        if n == 0:
            return float("nan")
        return float(abs(y_true[n - 1] - y_pred[n - 1]))


class IOHelper:
    """
    Handles data loading and saving for the simulation using specified path conventions.
    """

    @staticmethod
    def load_agent_attributes(path: str) -> pd.DataFrame:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                print(f"Warning: Failed to load agent attributes from {path}: {e}")
                return pd.DataFrame()
        print(f"Info: Agent attributes file not found at {path}. Using synthetic agents.")
        return pd.DataFrame()

    @staticmethod
    def load_train_data(path: str) -> pd.DataFrame:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                print(f"Warning: Failed to load train data from {path}: {e}")
                return pd.DataFrame()
        print(f"Info: Train data file not found at {path}. Metrics will be limited.")
        return pd.DataFrame()

    @staticmethod
    def save_results(df: pd.DataFrame, filename: str) -> None:
        try:
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print("Error saving results:", e)


class Simulation:
    """
    Coordinates and runs the multi-agent simulation according to the model plan.
    """

    def __init__(self, model_plan: Dict[str, Any]) -> None:
        self.model_plan = model_plan
        self.config: Dict[str, Any] = self._extract_parameters(model_plan)
        seed = int(self.config.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Containers
        self.agents: Dict[int, PersonAgent] = {}
        self.layers: Dict[str, Dict[int, List[int]]] = {"family": {}, "work": {}, "community": {}}
        self.policy_env: Optional[PolicyEnvironment] = None
        self.market: Optional[SupplyMarket] = None
        self.info_source: Optional[InformationSource] = None
        self.modules: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {
            "agent_mask": {},
            "agent_info": {},
        }
        self.results: Dict[str, List[Any]] = {
            "day": [],
            "observable.adoption_rate_daily": [],
            "mask_price": [],
        }
        self.observed_adoption: List[float] = []

        # Paths to data
        self.agent_file = os.path.join(DATA_DIR, "agent_attributes.csv")
        self.network_file = os.path.join(DATA_DIR, "social_network.json")
        self.train_file = os.path.join(DATA_DIR, "train_data.csv")

        # Initialize environment and state
        self._initialize_entities_and_data()

    def _extract_parameters(self, model_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten parameter list in the model plan into a simple dict of key -> default value.
        """
        params = {}
        for p in model_plan.get("parameters", []):
            key = p.get("key")
            if key is not None:
                params[key] = p.get("default")
        # Include prediction_period if present
        pred = model_plan.get("prediction_period", {})
        if isinstance(pred, dict):
            params["prediction_period_start_day"] = pred.get("start_day")
            params["prediction_period_end_day"] = pred.get("end_day")
        return params

    def _initialize_entities_and_data(self) -> None:
        # Load data
        agent_df = IOHelper.load_agent_attributes(self.agent_file)
        train_df = IOHelper.load_train_data(self.train_file)
        observed = []
        if not train_df.empty and "adoption_rate" in train_df.columns:
            observed = list(pd.to_numeric(train_df["adoption_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0))
        self.observed_adoption = observed

        # Build agents
        if agent_df.empty:
            n_agents = 200
            ids = list(range(n_agents))
            ages = self.rng.integers(18, 70, size=n_agents).tolist()
            occupations = ["other"] * n_agents
            ses = self.rng.normal(0.0, 1.0, size=n_agents).tolist()
            trust = self.rng.beta(2.0, 2.0, size=n_agents).tolist()
            risk = self.rng.uniform(0.0, 1.0, size=n_agents).tolist()
            df = pd.DataFrame(
                {
                    "agent_id": ids,
                    "age": ages,
                    "occupation": occupations,
                    "socioeconomic_status": ses,
                    "trust_in_authorities": trust,
                    "risk_perception": risk,
                }
            )
            agent_df = df
        else:
            # Normalize columns
            if "agent_id" not in agent_df.columns and "id" in agent_df.columns:
                agent_df = agent_df.rename(columns={"id": "agent_id"})
            # Imputations
            if "trust_in_authorities" not in agent_df.columns:
                agent_df["trust_in_authorities"] = np.nan
            if "socioeconomic_status" not in agent_df.columns:
                agent_df["socioeconomic_status"] = np.nan
            if "risk_perception" not in agent_df.columns:
                agent_df["risk_perception"] = np.nan
            agent_df["trust_in_authorities"] = agent_df["trust_in_authorities"].fillna(pd.Series(self.rng.beta(2.0, 2.0, size=len(agent_df))))
            agent_df["socioeconomic_status"] = agent_df["socioeconomic_status"].fillna(pd.Series(self.rng.normal(0.0, 1.0, size=len(agent_df))))
            agent_df["risk_perception"] = agent_df["risk_perception"].fillna(0.5)
            agent_df["risk_perception"] = pd.to_numeric(agent_df["risk_perception"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
            if "occupation" not in agent_df.columns:
                agent_df["occupation"] = "other"
            if "age" not in agent_df.columns:
                agent_df["age"] = pd.Series(self.rng.integers(18, 70, size=len(agent_df)))

        agent_ids = [int(x) for x in agent_df["agent_id"].tolist()]
        init_rate = self.config.get("initial_adoption_rate", 0.05)
        if len(self.observed_adoption) > 0:
            init_rate = float(self.observed_adoption[0])

        mask_state: Dict[int, int] = {}
        info_state: Dict[int, float] = {}

        for _, row in agent_df.iterrows():
            aid = int(row["agent_id"])
            age = int(row.get("age", 0))
            occ = str(row.get("occupation", ""))
            ses = float(row.get("socioeconomic_status", 0.0))
            trust = float(row.get("trust_in_authorities", 0.5))
            risk = float(row.get("risk_perception", 0.5))
            adopt = 1 if self.rng.random() < init_rate else 0

# Execute main for both direct execution and sandbox wrapper invocation
main()