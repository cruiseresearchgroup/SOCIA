import os
import json
import random
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Any, Optional, Tuple, Set
from geopy.distance import geodesic
import numpy as np
import logging
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import jensenshannon
import csv
import sys
import pathlib
import importlib.util
import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/llmob_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Get the directory where the script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# LLM CONFIGURATION
# -----------------------------

# Flag to switch on/off LLM-based inference (fallback to heuristic if disabled)
USE_LLM = True

# Load API key from data_fitting/llmob_data/keys.py (expects variable OPENAI_API_KEY)
KEYS_PATH = os.path.join(DATA_DIR, "keys.py")
if USE_LLM and os.path.exists(KEYS_PATH):
    try:
        spec = importlib.util.spec_from_file_location("keys", KEYS_PATH)
        keys_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keys_module)
        openai.api_key = getattr(keys_module, "OPENAI_API_KEY", None)
        if not openai.api_key:
            logging.warning("OPENAI_API_KEY not found in keys.py – LLM inference disabled.")
            USE_LLM = False
    except Exception as e:
        logging.error(f"Failed loading OpenAI key: {e}. Disabling LLM inference.")
        USE_LLM = False
else:
    if USE_LLM:
        logging.warning("keys.py not found – LLM inference disabled.")
    USE_LLM = False

# Default model to use
LLM_MODEL = "gpt-4o-mini"

# Helper to call the chat completion endpoint with robust error handling
def call_openai_chat(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """Call OpenAI ChatCompletion, return assistant content (or raise)."""
    if not USE_LLM:
        raise RuntimeError("LLM usage disabled or API key missing.")
    try:
        # Check OpenAI client version and use appropriate API call format
        if hasattr(openai, 'ChatCompletion'):
            # Old API format
            response = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return response.choices[0].message["content"].strip()
        else:
            # New API format
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        raise

# -----------------------------------------------------
# LLM-assisted resident profile & schedule generation
# -----------------------------------------------------

def llm_generate_resident_profile(resident_id: str, activities_by_date: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
    """Use LLM to infer pattern, persona and home/work locations from history."""
    if not USE_LLM:
        return None

    # Flatten activities into readable string
    history_lines = []
    for date, acts in activities_by_date.items():
        for act in acts:
            history_lines.append(f"{date}: {act}")
    history_text = "\n".join(history_lines)

    system_msg = (
        "You are an urban mobility analyst. From a resident's historical daily activities you must infer "
        "their overall daily pattern, persona, likely home location and likely workplace. "
        "Return a valid JSON object with keys pattern, persona, home_location, work_location. "
        "The pattern and persona must be dictionaries with scalar values (no lists or dictionaries as values). "
        "\n\nIMPORTANT LOCATION RULES:\n"
        "1. home_location and work_location MUST be EXACT names from the POI database. DO NOT use descriptive locations.\n"
        "2. NEVER use locations like 'near Park', 'suburban', 'Near School', etc.\n"
        "3. ALWAYS use full POI names including ID numbers like 'Park#1234', 'Office#567', 'Cafe#890', etc.\n"
        "4. If you're unsure about a specific POI, use a generic category with ID like 'Home#123' or 'Office#456'.\n"
        "5. DO NOT prefix locations with 'near', 'close to', or other descriptors.\n"
        "Do NOT wrap JSON in markdown."
    )

    user_msg = (
        f"Resident ID: {resident_id}\n"
        f"Historical activities (one per line):\n{history_text}\n"
        "Look at the historical activities and extract specific location names "
        "like 'Park#1234' or 'Cafe#567' to use as home_location and work_location.\n"
        "Provide the JSON now."
    )

    try:
        content = call_openai_chat(system_msg, user_msg)
        # Log the raw response
        logging.info(f"LLM profile response for resident {resident_id}: {content[:500]}...")
        
        json_start = content.find("{")
        if json_start == -1:
            logging.error(f"No JSON found in LLM response for resident {resident_id}")
            return None
            
        json_text = content[json_start:]
        profile = json.loads(json_text)
        
        # Validate profile structure
        if not isinstance(profile, dict):
            logging.warning(f"LLM returned non-dictionary profile for resident {resident_id}")
            return None
            
        # Ensure pattern and persona are dictionaries if present
        if "pattern" in profile and not isinstance(profile["pattern"], dict):
            profile["pattern"] = {}
        if "persona" in profile and not isinstance(profile["persona"], dict):
            profile["persona"] = {}

        # Validate location strings - check for forbidden prefixes
        forbidden_prefixes = ["near ", "close to ", "nearby ", "next to ", "suburb"]
        for loc_key in ["home_location", "work_location"]:
            if loc_key in profile and isinstance(profile[loc_key], str):
                loc = profile[loc_key].lower()
                if any(loc.startswith(prefix) for prefix in forbidden_prefixes):
                    # Remove the location if it has forbidden prefixes
                    logging.warning(f"Removing invalid {loc_key}: '{profile[loc_key]}' that uses descriptive prefix")
                    profile[loc_key] = None
            
        logging.info(f"✓ LLM profile generation completed for resident {resident_id}")
        return profile
    except Exception as e:
        logging.warning(f"LLM profile generation failed for resident {resident_id}: {e}")
        return None


def llm_generate_day_schedule(resident_profile: Dict[str, Any], past7_activities: List[str], simulation_date: str) -> Optional[List[Dict[str, Any]]]:
    """Ask LLM to propose a day schedule. Returns list of dicts with keys type, start_time, duration_minutes, location."""
    if not USE_LLM:
        return None

    resident_id = resident_profile.get("resident_id", "unknown")
    past_text = "\n".join(past7_activities)
    
    # Extract location examples from past activities to guide the LLM
    location_examples = set()
    for activity in past7_activities:
        parts = activity.split(" at ")
        if len(parts) >= 3:
            location = parts[2].strip()
            if "#" in location:  # Only include POI-style locations
                location_examples.add(location)
    
    # Convert to list and limit to at most 10 examples
    location_example_list = list(location_examples)[:10]
    location_example_text = ", ".join(location_example_list)

    system_msg = (
        "You are an AI assistant that creates realistic daily schedules for urban residents. "
        "The schedule must be plausible, match the resident's persona and pattern, use known locations when possible, "
        "and cover from 07:30 to 22:00. Return JSON list; each element must have type, start_time (HH:MM:SS, 24h), "
        "duration_minutes (int <= 240), and location (string)."
        "\n\nIMPORTANT LOCATION RULES:\n"
        "1. The 'location' field MUST be an EXACT POI name from the database, including its ID number.\n"
        "2. Valid location format examples: 'Park#1234', 'Cafe#5678', 'Restaurant#910', etc.\n"
        "3. NEVER use descriptive locations like 'near park', 'downtown', 'suburban', etc.\n"
        "4. DO NOT prefix locations with 'near', 'close to', or other descriptors.\n"
        "5. If unsure about a specific POI, use the resident's home_location or work_location, or reuse locations from their history.\n"
        "Do not include markdown."
    )

    user_msg = (
        f"Date to simulate: {simulation_date}.\n"
        f"Resident profile JSON:\n{json.dumps(resident_profile, ensure_ascii=False)}\n"
        f"Past 7 days activities (one per line):\n{past_text}\n\n"
        f"Examples of valid locations from the resident's history: {location_example_text if location_example_list else 'None available, use POIs with IDs like Restaurant#123'}\n\n"
        "Remember to use ONLY specific POI names with IDs (like 'Park#1234') for locations, never descriptive terms.\n"
        "Provide the schedule JSON list now."
    )

    try:
        content = call_openai_chat(system_msg, user_msg, max_tokens=1500)
        # Log the raw response
        logging.info(f"LLM schedule response for resident {resident_id}: {content[:500]}...")
        
        json_start = content.find("[")
        if json_start == -1:
            logging.error(f"No JSON array found in LLM schedule response for resident {resident_id}")
            return None
            
        schedule = json.loads(content[json_start:])
        
        # Validate location formats in the schedule
        forbidden_prefixes = ["near ", "close to ", "nearby ", "next to ", "suburb"]
        for i, activity in enumerate(schedule):
            if "location" in activity and isinstance(activity["location"], str):
                loc = activity["location"].lower()
                if any(loc.startswith(prefix) for prefix in forbidden_prefixes):
                    # Fix locations with forbidden prefixes by using home location as fallback
                    logging.warning(f"Invalid location in schedule: '{activity['location']}' - replacing with fallback")
                    home_loc = resident_profile.get("home_location")
                    if home_loc and "#" in home_loc:
                        activity["location"] = home_loc
                    else:
                        # Last resort fallback
                        activity["location"] = "Convenience Store#1"
        
        logging.info(f"✓ LLM schedule generation completed for resident {resident_id} on {simulation_date}")
        return schedule
    except Exception as e:
        logging.warning(f"LLM schedule generation failed: {e}")
        return None

def load_json(file_name: str) -> Any:
    """Load a JSON file from the data directory with error handling."""
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        logging.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if not os.access(file_path, os.R_OK):
        logging.error(f"No read permission for the file: {file_path}")
        raise PermissionError(f"No read permission for the file: {file_path}")
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if not data:
                logging.error(f"Loaded data from {file_name} is empty.")
                raise ValueError(f"Loaded data from {file_name} is empty.")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_name}: {e}")
        raise
    except IOError as e:
        logging.error(f"IO error loading {file_name}: {e}")
        raise

class Resident:
    """Represents an urban resident with attributes and behaviors."""
    
    def __init__(self, resident_id: str, attributes: Dict[str, Any], historical_activities: Dict[str, List[str]]):
        self.resident_id = resident_id
        self.age = attributes.get("age")
        self.gender = attributes.get("gender")
        self.occupation = attributes.get("occupation")
        self.home_location = attributes.get("home_location")
        self.work_location = attributes.get("work_location")
        self.historical_activities = historical_activities  # Activities indexed by date
        self.transport_mode = attributes.get("transport_mode", "walking")
        self.trajectory = []
        self.current_location = self.home_location
        self.daily_pattern = self.extract_pattern()
        self.persona = self.derive_persona()
        
    def extract_pattern(self) -> Dict[str, Any]:
        """Extract activity patterns from historical data."""
        # Pattern includes typical start times, duration, frequency of activities
        pattern = {
            "typical_wake_time": "07:00:00",
            "typical_sleep_time": "22:00:00",
            "frequent_locations": set(),
            "activity_durations": {}
        }
        
        for day_activities in self.historical_activities.values():
            for activity in day_activities:
                if " at " in activity:
                    parts = activity.split(" at ")
                    if len(parts) >= 2:
                        act_type = parts[0]
                        if act_type not in pattern["activity_durations"]:
                            pattern["activity_durations"][act_type] = []
                        pattern["activity_durations"][act_type].append(60)  # Default 60 min duration
                        
                        # Extract location if present
                        if len(parts) > 2:
                            location = parts[2]
                            pattern["frequent_locations"].add(location)
        
        # Calculate average durations
        for act_type, durations in pattern["activity_durations"].items():
            pattern["activity_durations"][act_type] = sum(durations) / len(durations)
            
        return pattern
    
    def derive_persona(self) -> Dict[str, Any]:
        """Derive persona based on attributes and historical activities."""
        persona = {
            "lifestyle": "active" if self.age < 45 else "moderate",
            "preferences": {}
        }
        
        # Count activity types
        activity_counts = {}
        total_activities = 0
        
        for day_activities in self.historical_activities.values():
            for activity in day_activities:
                act_type = activity.split(" at ")[0]
                activity_counts[act_type] = activity_counts.get(act_type, 0) + 1
                total_activities += 1
        
        # Calculate preferences based on activity frequency
        if total_activities > 0:
            for act_type, count in activity_counts.items():
                persona["preferences"][act_type] = count / total_activities
        
        return persona

    def commute(self, current_time: datetime) -> None:
        """Simulate commuting behavior for the resident."""
        work_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
        day_of_week = current_time.strftime("%A")
        
        # 如果工作地点是随机生成的（以Workplace_开头），则不进行通勤
        if self.work_location.startswith("Workplace_"):
            return
        
        if day_of_week in work_days:
            commute_start_time = dt_time(8, 0)
            commute_end_time = dt_time(9, 0)
            return_start_time = dt_time(17, 0)
            return_end_time = dt_time(18, 0)
            
            curr_time = current_time.time()
            
            if commute_start_time <= curr_time <= commute_end_time and self.current_location == self.home_location:
                self.current_location = self.work_location
                self.trajectory.append((self.current_location, current_time, "commuting to work"))
            elif return_start_time <= curr_time <= return_end_time and self.current_location == self.work_location:
                self.current_location = self.home_location
                self.trajectory.append((self.current_location, current_time, "returning home"))
    
    def participate_in_activity(self, motivation: str, current_time: datetime, duration_minutes: int) -> None:
        """Simulate activity participation based on motivation."""
        activities = {
            "shopping": self.shop,
            "exercising": self.exercise,
            "socializing": self.socialize,
            "working": self.work,
            "studying": self.study,
            "dining": self.dine,
            "resting": self.rest
        }
        
        activity_fn = activities.get(motivation)
        if activity_fn:
            end_time = current_time + timedelta(minutes=duration_minutes)
            activity_fn(current_time, end_time)
            return end_time
        
        return current_time + timedelta(minutes=duration_minutes)

    def shop(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate shopping activity."""
        self.trajectory.append((self.current_location, start_time, "shopping"))
        self.trajectory.append((self.current_location, end_time, "finished shopping"))

    def exercise(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate exercising activity."""
        self.trajectory.append((self.current_location, start_time, "exercising"))
        self.trajectory.append((self.current_location, end_time, "finished exercising"))

    def socialize(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate socializing activity."""
        self.trajectory.append((self.current_location, start_time, "socializing"))
        self.trajectory.append((self.current_location, end_time, "finished socializing"))

    def work(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate working activity."""
        if not self.work_location.startswith("Workplace_"):
            self.current_location = self.work_location
        self.trajectory.append((self.current_location, start_time, "working"))
        self.trajectory.append((self.current_location, end_time, "finished working"))

    def study(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate studying activity."""
        self.trajectory.append((self.current_location, start_time, "studying"))
        self.trajectory.append((self.current_location, end_time, "finished studying"))

    def dine(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate dining activity."""
        self.trajectory.append((self.current_location, start_time, "dining"))
        self.trajectory.append((self.current_location, end_time, "finished dining"))

    def rest(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate resting activity."""
        self.current_location = self.home_location
        self.trajectory.append((self.current_location, start_time, "resting"))
        self.trajectory.append((self.current_location, end_time, "finished resting"))

class Location:
    """Represents a location with attributes and behaviors."""

    def __init__(self, location_type: str, address: str, capacity: int, operating_hours: str, coordinates: Tuple[float, float]):
        self.type = location_type
        self.address = address
        self.capacity = capacity
        self.operating_hours = operating_hours
        self.current_occupancy = 0
        self.coordinates = coordinates
        self.popularity = 0
        self.visitors = set()  # Store unique visitors

    def host_activity(self, resident_id: str, activity_type: str) -> bool:
        """Host an activity at the location if capacity allows."""
        if self.current_occupancy < self.capacity:
            self.current_occupancy += 1
            self.popularity += 1
            self.visitors.add(resident_id)
            logging.info(f"Hosting {activity_type} for {resident_id} at {self.address}. Current occupancy: {self.current_occupancy}")
            return True
        logging.warning(f"Cannot host {activity_type} at {self.address}. Capacity full.")
        return False

    def is_open(self, current_time: datetime) -> bool:
        """Check if the location is open at the current time."""
        if "-" not in self.operating_hours:
            return True  # Always open if no specific hours
            
        open_time_str, close_time_str = self.operating_hours.split("-")
        open_time = datetime.strptime(open_time_str, "%H:%M").time()
        close_time = datetime.strptime(close_time_str, "%H:%M").time()
        
        return open_time <= current_time.time() <= close_time

class SimulationEnvironment:
    """Manages the simulation environment and timing."""

    def __init__(self, simulation_date: datetime, time_step_minutes: int = 10):
        self.simulation_date = simulation_date
        self.current_time = datetime.combine(simulation_date.date(), dt_time(7, 0))  # Start at 7 AM
        self.end_time = datetime.combine(simulation_date.date(), dt_time(22, 0))  # End at 10 PM
        self.time_step = timedelta(minutes=time_step_minutes)

    def advance_time(self) -> None:
        """Advance simulation time by one time step."""
        self.current_time += self.time_step

class UrbanSimulation:
    """Main class to coordinate the urban simulation."""
    
    def __init__(self, random_seed: int = 42):
        self.residents = {}  # Dictionary of resident_id -> Resident
        self.locations = []
        self.poi_data = {}
        self.category_data = {}  # 存储类别数据
        self.poi_by_category = {}  # 按类别存储POI数据
        self.poi_by_name = {}  # 按名称存储POI数据
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.resident_data = {}
        self.load_data()
        
    def load_data(self) -> None:
        """Load and initialize data for the simulation."""
        try:
            self.resident_data = load_json("2019Y.json")
            self.poi_data = load_json("poi_category_192021_longitude_latitude_complement_alignment_clean.json")
            self.category_data = load_json("catto.json")
            
            # 处理POI数据，建立查找索引
            for category, pois in self.poi_data.items():
                self.poi_by_category[category] = []
                for poi in pois:
                    if len(poi) > 2:
                        poi_name = poi[2]
                        coordinates = (float(poi[0]), float(poi[1]))
                        
                        # 存储不含编号的POI名称与完整POI信息的映射
                        base_name = poi_name.split("#")[0] if "#" in poi_name else poi_name
                        if base_name not in self.poi_by_name:
                            self.poi_by_name[base_name] = []
                        self.poi_by_name[base_name].append({
                            "full_name": poi_name,
                            "category": category,
                            "coordinates": coordinates
                        })
                        
                        # 存储POI类别与POI信息的映射
                        self.poi_by_category[category].append({
                            "name": poi_name,
                            "coordinates": coordinates
                        })
            
            # Initialize locations
            location_id = 0
            for location_type, details in self.category_data.items():
                for poi in self.poi_data.get(location_type, []):
                    coordinates = (float(poi[0]), float(poi[1]))
                    self.locations.append(Location(
                        location_type=location_type,
                        address=poi[2] if len(poi) > 2 else f"Location_{location_id}",
                        capacity=random.randint(10, 100),
                        operating_hours="09:00-18:00",  # Default operating hours
                        coordinates=coordinates
                    ))
                    location_id += 1
            
            logging.info(f"Loaded {len(self.resident_data)} residents and {len(self.locations)} locations.")
            logging.info(f"Processed {len(self.poi_by_name)} unique location types and {len(self.poi_by_category)} location categories.")
            
        except FileNotFoundError as e:
            logging.error(f"Error loading data: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            raise

    def initialize_resident(self, resident_id: str) -> Resident:
        """Initialize a resident with the given ID for simulation."""
        if resident_id not in self.resident_data:
            logging.error(f"Resident ID {resident_id} not found in data.")
            raise ValueError(f"Resident ID {resident_id} not found in data.")
            
        # Parse historical activities dictionary
        historical_activities = self.parse_activities(self.resident_data[resident_id])
        
        # Optionally ask LLM to infer pattern/persona/home/work
        logging.info(f"Generating profile for resident {resident_id}...")
        llm_profile = llm_generate_resident_profile(resident_id, historical_activities)

        if llm_profile:
            home_location = llm_profile.get("home_location")
            work_location = llm_profile.get("work_location")
            
            # Ensure pattern and persona are properly formatted dictionaries
            pattern_overrides = {}
            if "pattern" in llm_profile and isinstance(llm_profile["pattern"], dict):
                pattern_overrides = llm_profile["pattern"]
                
            persona_overrides = {}
            if "persona" in llm_profile and isinstance(llm_profile["persona"], dict):
                persona_overrides = llm_profile["persona"]
        else:
            home_location = None
            work_location = None
            pattern_overrides = {}
            persona_overrides = {}

        # 从历史活动中推断家庭和工作地点（如果LLM未给出或给出的不完整）
        if not home_location or not work_location:
            inferred_home, inferred_work = self.infer_home_work_locations(historical_activities)
            home_location = home_location or inferred_home
            work_location = work_location or inferred_work

        # 如果仍然无法推断出真实地点，则从POI数据库中选择合适的位置
        if not home_location or home_location.startswith("Home_"):
            # 尝试从住宅类别中选择一个POI
            home_categories = ["Home", "Apartment", "Residential Area", "Housing"]
            home_location = self.select_poi_from_categories(home_categories)
            
            # 如果以上类别找不到，使用便利店或地标性位置作为备选
            if not home_location:
                fallback_categories = ["Convenience Store", "Park", "Shopping Mall"]
                home_location = self.select_poi_from_categories(fallback_categories)
                
            # 最后才使用默认值
            if not home_location:
                home_location = f"Home_{resident_id}"
        
        if not work_location or work_location.startswith("Workplace_"):
            # 尝试从工作类别中选择一个POI
            work_categories = ["Office", "Company", "School", "University", "Hospital"]
            work_location = self.select_poi_from_categories(work_categories)
            
            # 如果以上类别找不到，使用商业区域作为备选
            if not work_location:
                fallback_categories = ["Shopping Mall", "Department Store", "Building"]
                work_location = self.select_poi_from_categories(fallback_categories)
                
            # 最后才使用默认值
            if not work_location:
                work_location = f"Workplace_{resident_id}"
            
        # Create resident attributes
        attributes = {
            "age": random.randint(18, 65),
            "gender": random.choice(["male", "female"]),
            "occupation": random.choice(["worker", "student"]),
            "home_location": home_location,
            "work_location": work_location,
            "transport_mode": random.choice(["car", "bike", "walking"])
        }
        
        resident_obj = Resident(resident_id, attributes, historical_activities)

        # Apply LLM-derived pattern / persona overrides
        if pattern_overrides:
            resident_obj.daily_pattern.update(pattern_overrides)
        if persona_overrides:
            resident_obj.persona.update(persona_overrides)

        return resident_obj
    
    def select_poi_from_categories(self, categories: List[str]) -> Optional[str]:
        """
        从指定类别中选择一个POI
        
        Args:
            categories: POI类别列表
        
        Returns:
            选中的POI名称，如果没有找到则返回None
        """
        for category in categories:
            if category in self.poi_by_category and self.poi_by_category[category]:
                poi_options = self.poi_by_category[category]
                if poi_options:
                    chosen_poi = random.choice(poi_options)
                    return chosen_poi["name"]
        
        # 如果找不到指定类别，尝试从任何类别中选择
        all_categories = list(self.poi_by_category.keys())
        if all_categories:
            random_category = random.choice(all_categories)
            if self.poi_by_category[random_category]:
                chosen_poi = random.choice(self.poi_by_category[random_category])
                return chosen_poi["name"]
        
        return None

    def infer_home_work_locations(self, historical_activities: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[str]]:
        """
        推断居民的家庭和工作地点，基于他们的历史活动
        
        Args:
            historical_activities: 按日期索引的历史活动记录
        
        Returns:
            家庭位置和工作位置的元组(home_location, work_location)
        """
        # 记录与"home"相关活动和"work"相关活动的地点频率
        home_locations = {}
        work_locations = {}
        
        # 与家相关的活动类型
        home_activities = ["resting", "sleep", "sleeping", "home", "staying at home", "relaxing"]
        # 与工作相关的活动类型
        work_activities = ["working", "work", "office", "at office", "at company", "at workplace", "business"]
        
        # 分析历史活动
        for day_activities in historical_activities.values():
            for activity in day_activities:
                parts = activity.split(" at ")
                if len(parts) >= 3:  # 活动格式为 "活动类型 at 时间 at 地点"
                    activity_type = parts[0].lower().strip()
                    location = parts[2].strip()
                    
                    # 提取基本地点名称（去掉#后的数字）
                    base_location = location.split("#")[0] if "#" in location else location
                    
                    # 检查是否是家庭相关活动
                    if any(home_act in activity_type for home_act in home_activities):
                        home_locations[base_location] = home_locations.get(base_location, 0) + 1
                    
                    # 检查是否是工作相关活动
                    if any(work_act in activity_type for work_act in work_activities):
                        work_locations[base_location] = work_locations.get(base_location, 0) + 1
        
        # 找到最常见的家庭和工作地点
        home_location = None
        work_location = None
        
        if home_locations:
            # 获取出现频率最高的地点作为家
            home_location = max(home_locations.items(), key=lambda x: x[1])[0]
            
            # 如果找到的地点在POI数据中存在，使用完整名称
            if home_location in self.poi_by_name:
                poi_options = self.poi_by_name[home_location]
                if poi_options:
                    home_location = poi_options[0]["full_name"]
        
        if work_locations:
            # 获取出现频率最高的地点作为工作地点
            work_location = max(work_locations.items(), key=lambda x: x[1])[0]
            
            # 如果找到的地点在POI数据中存在，使用完整名称
            if work_location in self.poi_by_name:
                poi_options = self.poi_by_name[work_location]
                if poi_options:
                    work_location = poi_options[0]["full_name"]
        
        # 如果家庭和工作地点相同，尝试找到第二常见的工作地点
        if home_location and work_location and home_location == work_location and len(work_locations) > 1:
            # 将最常见地点移除
            del work_locations[work_location]
            if work_locations:
                work_location = max(work_locations.items(), key=lambda x: x[1])[0]
                # 检查POI数据
                if work_location in self.poi_by_name:
                    poi_options = self.poi_by_name[work_location]
                    if poi_options:
                        work_location = poi_options[0]["full_name"]
                        
        return home_location, work_location

    def parse_activities(self, activities: List[str]) -> Dict[str, List[str]]:
        """
        Parse activities and organize them by date.
        
        Extract actual dates from activity descriptions in the format "Activities at YYYY-MM-DD:"
        
        Args:
            activities: List of activity descriptions
            
        Returns:
            Dictionary mapping dates to activities
        """
        historical_activities = {}
        
        # Extract dates from the format "Activities at YYYY-MM-DD:"
        for activity in activities:
            if activity.startswith("Activities at "):
                # Extract date part from "Activities at 2019-01-02: ..."
                date_part = activity.split("Activities at ")[1].split(":")[0].strip()
                
                try:
                    # Attempt to parse the date
                    date_obj = datetime.strptime(date_part, "%Y-%m-%d")
                    date_str = date_obj.strftime("%Y-%m-%d")
                    
                    if date_str not in historical_activities:
                        historical_activities[date_str] = []
                    
                    # Extract the individual activities from the day
                    activities_part = activity.split(": ")[1]
                    # Split by commas, but handle cases where commas are inside location names
                    daily_activities = []
                    current_activity = ""
                    
                    for item in activities_part.split(", "):
                        if " at " in item and not current_activity:
                            daily_activities.append(item)
                        elif " at " in item and current_activity:
                            daily_activities.append(current_activity)
                            current_activity = item
                        else:
                            if current_activity:
                                current_activity += ", " + item
                            else:
                                current_activity = item
                    
                    if current_activity:
                        daily_activities.append(current_activity)
                    
                    # Add each individual activity to this date
                    for act in daily_activities:
                        historical_activities[date_str].append(act)
                        
                except (ValueError, IndexError) as e:
                    logging.warning(f"Failed to parse date from '{activity}': {e}")
                    continue
        
        # If no valid dates were found in the activities (should not happen with proper data)
        if not historical_activities:
            logging.warning("No valid dates found in activities. Using fallback date assignment.")
            # Use a fixed date range in 2019 for fallback
            base_date = datetime(2019, 1, 1)
            
            # Group activities for fallback
            for i, activity in enumerate(activities):
                date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
                historical_activities[date_str] = [activity]
        
        return historical_activities

    def run(self, resident_id: str, simulation_date: Optional[datetime] = None, 
            use_only_previous_days: bool = False) -> List[Tuple[str, datetime, str]]:
        """
        Run the simulation for a single resident on a specific date.
        
        This process follows these steps:
        1. Gather activities from the 7 days before the simulation date
        2. Use these activities to predict motivations for the simulation date
        3. Generate a day schedule based on the resident's pattern and predicted motivations
        4. Execute the schedule to create a trajectory of activities
        
        Args:
            resident_id: The ID of the resident to simulate
            simulation_date: The date to simulate (default: current date)
            use_only_previous_days: If True, only use data from days prior to simulation_date
            
        Returns:
            The resident's trajectory for the simulated day
        """
        if simulation_date is None:
            simulation_date = datetime.now()
            
        # Initialize the resident if not already done
        if resident_id not in self.residents:
            self.residents[resident_id] = self.initialize_resident(resident_id)
        
        resident = self.residents[resident_id]
        
        # Set up the simulation environment
        environment = SimulationEnvironment(simulation_date)
        simulation_date_str = simulation_date.strftime("%Y-%m-%d")
        logging.info(f"Step 3: Running simulation for resident {resident_id} on {simulation_date_str}")
        
        # Get historical activities for prediction
        history_days = 7
        historical_activities = []
        
        # Sort available days and find those before simulation_date
        available_days = list(resident.historical_activities.keys())
        available_days.sort()
        
        if use_only_previous_days:
            # Find all activities from previous 7 days (not necessarily consecutive)
            previous_activities = []
            previous_days_found = []
            
            for i in range(1, 8):
                prev_date = simulation_date - timedelta(days=i)
                prev_date_str = prev_date.strftime("%Y-%m-%d")
                
                if prev_date_str in resident.historical_activities:
                    previous_days_found.append(prev_date_str)
                    day_activities = resident.historical_activities[prev_date_str]
                    previous_activities.extend(day_activities)
                    logging.info(f"Step 2: Found {len(day_activities)} activities on {prev_date_str} for resident {resident_id}")
            
            # Log how many days we actually found
            logging.info(f"Step 2: Using {len(previous_activities)} total activities from {len(previous_days_found)} days for prediction")
            
            if previous_activities:
                historical_activities = previous_activities
            else:
                # Fallback: use any available previous days if no activities in the past 7 days
                logging.warning(f"No activities found in previous 7 days for resident {resident_id}. Using fallback.")
                previous_days = [day for day in available_days if day < simulation_date_str]
                previous_days = previous_days[-history_days:] if previous_days else []
                
                for day in previous_days:
                    historical_activities.extend(resident.historical_activities[day])
                logging.info(f"Step 2: Using {len(historical_activities)} activities from fallback days")
        else:
            # Use a mix of all available historical data except the simulation date
            for day in available_days:
                if day in resident.historical_activities and day != simulation_date_str:
                    historical_activities.extend(resident.historical_activities[day])
            logging.info(f"Step 2: Using {len(historical_activities)} activities from all historical days")
        
        # Clear the resident's trajectory for the new simulation
        resident.trajectory = []
        resident.current_location = resident.home_location
        
        # Start at home in the morning
        resident.trajectory.append((resident.home_location, environment.current_time, "waking up"))
        
        # Morning routine (breakfast, preparation)
        current_time = environment.current_time + timedelta(minutes=30)  # 30 min morning routine
        resident.trajectory.append((resident.home_location, current_time, "morning routine"))
        
        # Generate the day's activity schedule (LLM first, fallback to heuristic)
        logging.info(f"Step 3: Generating day schedule for resident {resident_id} on {simulation_date_str}...")

        # Convert pattern and sets to JSON-serializable format
        resident_profile = {
            "resident_id": resident_id,
            "pattern": {k: list(v) if isinstance(v, set) else v for k, v in resident.daily_pattern.items()},
            "persona": resident.persona,
            "home_location": resident.home_location,
            "work_location": resident.work_location
        }

        llm_schedule = llm_generate_day_schedule(resident_profile, 
                                               historical_activities[-50:], 
                                               simulation_date.strftime("%Y-%m-%d")) if USE_LLM else None

        if llm_schedule:
            day_schedule = llm_schedule
            logging.info(f"Step 3: LLM provided schedule with {len(day_schedule)} activities")
        else:
            day_schedule = self.generate_day_schedule(resident, historical_activities, simulation_date)
            logging.info(f"Step 3: Heuristic schedule with {len(day_schedule)} activities")
        
        logging.info(f"Step 4: Generated schedule with {len(day_schedule)} activities for {simulation_date_str}")
        
        # Execute the day's schedule
        for i, activity in enumerate(day_schedule):
            motivation = activity["type"]
            start_time = datetime.combine(simulation_date.date(), 
                                         datetime.strptime(activity["start_time"], "%H:%M:%S").time())
            duration_minutes = activity["duration_minutes"]
            location = activity["location"]
            
            logging.info(f"Step 4: Scheduled activity {i+1}/{len(day_schedule)}: {motivation} at {start_time.strftime('%H:%M:%S')} for {duration_minutes} minutes at {location}")
            
            # Skip activities that would start after the end time
            if start_time >= environment.end_time:
                logging.info(f"Step 4: Skipping activity {motivation} as it starts after end time")
                continue
                
            # If there's a gap in the schedule, add commuting or resting
            if current_time < start_time:
                gap_minutes = int((start_time - current_time).total_seconds() / 60)
                if gap_minutes > 15:  # Only consider gaps > 15 minutes
                    logging.info(f"Step 4: Adding resting activity to fill {gap_minutes} minute gap")
                    resident.rest(current_time, start_time)
            
            # Update current location before performing the activity
            if motivation != "resting":  # For resting, we already set home_location in the rest method
                resident.current_location = location
            
            # Perform the scheduled activity
            logging.info(f"Step 4: Executing {motivation} activity at {location}")
            end_time = resident.participate_in_activity(motivation, start_time, duration_minutes)
            current_time = end_time
        
        # Return home at the end of the day if not already there
        if resident.current_location != resident.home_location:
            final_time = min(environment.end_time, current_time + timedelta(minutes=30))
            logging.info(f"Step 4: Returning home at {final_time.strftime('%H:%M:%S')}")
            resident.trajectory.append((resident.home_location, final_time, "returned home"))
            
        # Add final resting/sleeping activity
        if current_time < environment.end_time:
            logging.info(f"Step 4: Adding final resting activity from {current_time.strftime('%H:%M:%S')} to {environment.end_time.strftime('%H:%M:%S')}")
            resident.rest(current_time, environment.end_time)
            
        logging.info(f"Step 4: Completed trajectory generation with {len(resident.trajectory)} total activities")
        return resident.trajectory

    def generate_day_schedule(self, resident: Resident, historical_activities: List[str], 
                              simulation_date: datetime) -> List[Dict[str, Any]]:
        """
        Generate a daily activity schedule based on historical data, patterns, and persona.
        
        Args:
            resident: The resident to generate schedule for
            historical_activities: List of historical activities
            simulation_date: The date to generate schedule for
            
        Returns:
            List of scheduled activities with timing information
        """
        day_of_week = simulation_date.strftime("%A")
        is_weekday = day_of_week in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
        
        # Create schedule structure
        schedule = []
        
        # Extract activity patterns from historical data
        activity_times = {}
        activity_types = []
        activity_locations = {}  # 记录活动类型和地点的映射
        
        for activity in historical_activities:
            if " at " in activity:
                parts = activity.split(" at ")
                if len(parts) >= 2:
                    activity_type = parts[0].strip()
                    time_str = parts[1].strip()
                    
                    # 提取位置（如果有）
                    location = parts[2].strip() if len(parts) >= 3 else None
                    
                    # Standardize activity types
                    std_type = self.standardize_activity_type(activity_type)
                    
                    if std_type:
                        if std_type not in activity_times:
                            activity_times[std_type] = []
                            activity_locations[std_type] = {}
                            
                        try:
                            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
                            activity_times[std_type].append(time_obj)
                            
                            if std_type not in activity_types:
                                activity_types.append(std_type)
                                
                            # 记录活动类型与地点的对应关系
                            if location:
                                if location not in activity_locations[std_type]:
                                    activity_locations[std_type][location] = 0
                                activity_locations[std_type][location] += 1
                        except ValueError:
                            continue
        
        # If we have historical data, prioritize that
        if activity_types:
            # For each extracted activity type, add it to the schedule
            for act_type in activity_types:
                if act_type in activity_times and activity_times[act_type]:
                    # Calculate average time for this activity
                    times = activity_times[act_type]
                    avg_hour = sum(t.hour for t in times) / len(times)
                    avg_minute = sum(t.minute for t in times) / len(times)
                    
                    # Create time string with some randomness
                    random_offset = random.randint(-30, 30)  # ±30 minutes
                    time_minutes = int(avg_hour * 60 + avg_minute + random_offset)
                    hour = (time_minutes // 60) % 24  # Ensure within 24-hour range
                    minute = time_minutes % 60
                    
                    time_str = f"{hour:02d}:{minute:02d}:00"
                    
                    # Skip invalid times (should be rare with the modulo operation)
                    if hour >= 24 or hour < 0:
                        continue
                    
                    # 根据活动类型选择合适的地点
                    location = None
                    
                    # 对于特定活动类型，使用推断的家庭和工作地点
                    if act_type == "resting" or act_type == "sleeping":
                        location = resident.home_location
                    elif act_type == "working":
                        location = resident.work_location
                    else:
                        # 对于其他活动，尝试从历史数据中找到最常见的地点
                        if act_type in activity_locations and activity_locations[act_type]:
                            location_counts = activity_locations[act_type]
                            if location_counts:
                                location = max(location_counts.items(), key=lambda x: x[1])[0]
                    
                    # 如果没有找到合适的地点，使用get_location_for_activity方法
                    if not location or (location.startswith("Home_") or location.startswith("Workplace_")):
                        location = self.get_location_for_activity(act_type, resident, historical_activities)
                    
                    # Add activity to schedule with reasonable duration
                    schedule.append({
                        "type": act_type,
                        "start_time": time_str,
                        "duration_minutes": random.randint(45, 120),  # 45 mins to 2 hours
                        "location": location
                    })
        
        # If we don't have enough historical data, fall back to generic schedule
        if len(schedule) < 3:
            if is_weekday and resident.occupation == "worker":
                schedule.append({
                    "type": "working",
                    "start_time": "09:00:00",
                    "duration_minutes": 240,  # 4 hours before lunch
                    "location": resident.work_location
                })
                schedule.append({
                    "type": "dining",
                    "start_time": "13:00:00",
                    "duration_minutes": 60,
                    "location": "Restaurant"
                })
                schedule.append({
                    "type": "working",
                    "start_time": "14:00:00",
                    "duration_minutes": 180,  # 3 hours afternoon work
                    "location": resident.work_location
                })
            elif is_weekday and resident.occupation == "student":
                schedule.append({
                    "type": "studying",
                    "start_time": "09:00:00",
                    "duration_minutes": 240,
                    "location": "Library"
                })
                schedule.append({
                    "type": "dining",
                    "start_time": "13:00:00",
                    "duration_minutes": 60,
                    "location": "Restaurant"
                })
                schedule.append({
                    "type": "studying",
                    "start_time": "14:00:00",
                    "duration_minutes": 180,
                    "location": "Library"
                })
            else:
                # More flexible schedule for weekends or non-working/studying people
                # Morning activity
                morning_activities = ["shopping", "exercising", "socializing"]
                morning_act = random.choice(morning_activities)
                schedule.append({
                    "type": morning_act,
                    "start_time": "10:00:00",
                    "duration_minutes": 120,
                    "location": self.get_location_for_activity(morning_act, resident, historical_activities)
                })
                
                # Lunch
                schedule.append({
                    "type": "dining",
                    "start_time": "13:00:00",
                    "duration_minutes": 60,
                    "location": "Restaurant"
                })
                
                # Afternoon activity
                afternoon_activities = ["shopping", "socializing", "resting"]
                afternoon_act = random.choice(afternoon_activities)
                location = resident.home_location if afternoon_act == "resting" else self.get_location_for_activity(afternoon_act, resident, historical_activities)
                schedule.append({
                    "type": afternoon_act,
                    "start_time": "15:00:00",
                    "duration_minutes": 120,
                    "location": location
                })
            
            # Add evening activities for everyone
            evening_act = random.choice(["dining", "socializing", "resting"])
            location = resident.home_location if evening_act == "resting" else self.get_location_for_activity(evening_act, resident, historical_activities)
            schedule.append({
                "type": evening_act,
                "start_time": "18:00:00",
                "duration_minutes": 120,
                "location": location
            })
        
        # Ensure we have dining activities
        has_dining = any(activity["type"] == "dining" for activity in schedule)
        if not has_dining:
            schedule.append({
                "type": "dining",
                "start_time": "13:00:00",
                "duration_minutes": 60,
                "location": "Restaurant"
            })
        
        # Adjust schedule with some randomness for more realism
        for activity in schedule:
            # Add some randomness to start times (±15 min)
            start_time = datetime.strptime(activity["start_time"], "%H:%M:%S")
            random_minutes = random.randint(-15, 15)
            adjusted_start_time = start_time + timedelta(minutes=random_minutes)
            activity["start_time"] = adjusted_start_time.strftime("%H:%M:%S")
            
            # Add some randomness to durations (±15%)
            duration = activity["duration_minutes"]
            adjustment_factor = random.uniform(0.85, 1.15)
            activity["duration_minutes"] = int(duration * adjustment_factor)
        
        # Sort schedule by start time
        schedule.sort(key=lambda x: x["start_time"])
        
        return schedule
        
    def standardize_activity_type(self, activity_type: str) -> str:
        """Map various activity descriptions to standard types."""
        activity_type = activity_type.lower().strip()
        
        # Define mappings of raw activity types to standard categories
        mappings = {
            "shopping": ["shopping", "shop", "buying", "purchase", "mall"],
            "exercising": ["exercising", "exercise", "gym", "workout", "fitness", "running", "jogging"],
            "socializing": ["socializing", "socialize", "meeting", "meet", "party", "gathering", "visiting"],
            "working": ["working", "work", "job", "office", "business", "employment"],
            "studying": ["studying", "study", "learning", "research", "library", "reading", "education"],
            "dining": ["dining", "dinner", "lunch", "breakfast", "eating", "restaurant", "cafe", "food"],
            "resting": ["resting", "rest", "relaxing", "relax", "sleep", "nap", "break", "home"]
        }
        
        # Find the best match
        for std_type, variants in mappings.items():
            for variant in variants:
                if variant in activity_type:
                    return std_type
        
        # Default to resting if no match found
        return "resting"
        
    def get_location_for_activity(self, activity_type: str, resident: Resident, 
                                previous_activities: List[str] = None) -> str:
        """
        Get a suitable location for the given activity.
        
        This function selects a location from resident's history or from POI database that matches
        the activity type. All locations must exist in the POI database and have a valid category.
        
        Args:
            activity_type: The type of activity
            resident: The resident who will engage in the activity
            previous_activities: List of previous activities (optional)
            
        Returns:
            A valid location name from the POI database
        """
        # 优先考虑居民的特定地点
        if activity_type in ["resting", "sleeping"] and not resident.home_location.startswith("Home_"):
            return resident.home_location
            
        if activity_type == "working" and not resident.work_location.startswith("Workplace_"):
            return resident.work_location
        
        # 首先根据活动类型映射到可能的地点类别
        category_mapping = {
            "shopping": ["Shopping Mall", "Convenience Store", "Supermarket", "Department Store"],
            "exercising": ["Gym", "Park", "Sports Club", "Stadium"],
            "socializing": ["Cafe", "Restaurant", "Bar", "Movie Theater", "Pub"],
            "working": ["Office", "Company"],
            "studying": ["Library", "University", "School"],
            "dining": ["Restaurant", "Cafe", "Fast Food Restaurant"],
            "resting": ["Home"]
        }
        
        # 从居民历史活动中提取可能的地点
        historical_locations = []
        if previous_activities:
            for activity in previous_activities:
                parts = activity.split(" at ")
                if len(parts) >= 3:  # 活动格式为 "活动类型 at 时间 at 地点"
                    hist_activity_type = parts[0]
                    location = parts[2].strip()
                    
                    # 如果历史活动类型与当前活动类型相同，优先考虑该地点
                    if self.standardize_activity_type(hist_activity_type) == self.standardize_activity_type(activity_type):
                        # 提取基本地点名称（去掉#后的数字）
                        base_location = location.split("#")[0] if "#" in location else location
                        historical_locations.append(base_location)
        
        # 如果历史活动中有匹配的地点，优先从中随机选择
        if historical_locations:
            # 过滤出在POI数据中存在的地点
            valid_locations = [loc for loc in historical_locations if loc in self.poi_by_name]
            if valid_locations:
                chosen_base = random.choice(valid_locations)
                # 从该基本地点名称对应的所有POI中随机选择一个
                poi_options = self.poi_by_name[chosen_base]
                chosen_poi = random.choice(poi_options)
                return chosen_poi["full_name"]
        
        # 如果没有合适的历史地点，从活动类型对应的类别中选择
        possible_categories = category_mapping.get(activity_type, ["Convenience Store"])  # 默认使用便利店
        
        # 过滤出在POI数据中存在的类别
        valid_categories = [cat for cat in possible_categories if cat in self.poi_by_category]
        
        if not valid_categories:
            # 如果没有匹配的类别，使用任意一个有效类别
            valid_categories = [cat for cat in self.poi_by_category.keys() if len(self.poi_by_category[cat]) > 0]
        
        if valid_categories:
            chosen_category = random.choice(valid_categories)
            poi_options = self.poi_by_category[chosen_category]
            if poi_options:
                chosen_poi = random.choice(poi_options)
                return chosen_poi["name"]
        
        # 以防所有方法都失败，返回一个默认的地点 (首选"Convenience Store")
        for default_category in ["Convenience Store", "Park", "Restaurant"]:
            if default_category in self.poi_by_category and self.poi_by_category[default_category]:
                chosen_poi = random.choice(self.poi_by_category[default_category])
                return chosen_poi["name"]
        
        # 最后的后备方案
        return "Convenience Store#1"

    def evaluate(self, resident_id: str, ground_truth_day: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the simulation by comparing predicted activities with ground truth.
        
        The evaluation follows this process:
        1. For a resident, randomly select a day from their 2019Y.JSON activities
        2. Get activities from the 7 days preceding this day
        3. Use these 7 days of activities to predict today's motivations
        4. Generate a day schedule based on the resident's pattern and predicted motivations
        5. Compare the predicted activities with the actual activities for evaluation
        
        Args:
            resident_id: The ID of the resident to evaluate
            ground_truth_day: The specific day to evaluate against (optional)
                              If None, a random day will be selected
            
        Returns:
            Dictionary with evaluation metrics
        """
        if resident_id not in self.residents:
            # Initialize the resident if not already done
            try:
                self.residents[resident_id] = self.initialize_resident(resident_id)
                logging.info(f"Step 1: Initialized resident {resident_id} with historical activities")
            except Exception as e:
                logging.error(f"Failed to initialize resident {resident_id}: {e}")
                return {"error": f"Failed to initialize resident: {e}"}
                
        resident = self.residents[resident_id]
        
        # Get available days with activity data
        available_days = list(resident.historical_activities.keys())
        if not available_days:
            logging.error(f"No historical data for resident {resident_id}.")
            return {"error": "No historical data"}
        
        logging.info(f"Step 1: Resident {resident_id} has {len(available_days)} days of activity data available")
        
        # Sort days chronologically
        available_days.sort()
        
        # Randomly select a day if not specified
        if ground_truth_day is None:
            # Find days that have some history in the previous 7 days
            eligible_days = []
            
            for i, day in enumerate(available_days):
                day_date = datetime.strptime(day, "%Y-%m-%d")
                
                # Check if there are any activities in the previous 7 days
                has_history = False
                prev_days_count = 0
                
                for j in range(1, 8):
                    prev_date = day_date - timedelta(days=j)
                    prev_date_str = prev_date.strftime("%Y-%m-%d")
                    if prev_date_str in resident.historical_activities:
                        has_history = True
                        prev_days_count += 1
                
                # Consider eligible if at least one day of history exists
                if has_history:
                    eligible_days.append((day, prev_days_count))
            
            if not eligible_days:
                logging.warning(f"No days with history found for resident {resident_id}.")
                return {"error": "Insufficient historical data"}
                
            # Prefer days with more history by sorting
            eligible_days.sort(key=lambda x: x[1], reverse=True)
            ground_truth_day = eligible_days[0][0]
            history_days_count = eligible_days[0][1]
            
            logging.info(f"Step 1: Selected day {ground_truth_day} with {history_days_count} days of history for resident {resident_id}")
        else:
            # Calculate history days for a specified ground truth day
            day_date = datetime.strptime(ground_truth_day, "%Y-%m-%d")
            history_days_count = 0
            for j in range(1, 8):
                prev_date = day_date - timedelta(days=j)
                prev_date_str = prev_date.strftime("%Y-%m-%d")
                if prev_date_str in resident.historical_activities:
                    history_days_count += 1
        
        # Get ground truth activities for the specified day
        ground_truth = resident.historical_activities.get(ground_truth_day, [])
        
        if not ground_truth:
            logging.error(f"No ground truth data for resident {resident_id} on day {ground_truth_day}.")
            return {"error": "No ground truth data for selected day"}
            
        logging.info(f"Step 1: Retrieved {len(ground_truth)} ground truth activities for resident {resident_id} on {ground_truth_day}")
            
        # Extract standardized activity types from ground truth
        ground_truth_types = set()
        for activity in ground_truth:
            if " at " in activity:
                activity_type = activity.split(" at ")[0].strip()
                std_type = self.standardize_activity_type(activity_type)
                ground_truth_types.add(std_type)
        
        # If no valid activity types found, use the raw text
        if not ground_truth_types:
            ground_truth_types = set(act.split(" at ")[0].strip() if " at " in act else act 
                                   for act in ground_truth)
            
        logging.info(f"Step 2: Will gather activities from the 7 days before {ground_truth_day} to use as history")
            
        # Run simulation for this day, using only data from previous 7 days
        simulation_date = datetime.strptime(ground_truth_day, "%Y-%m-%d")
        predicted_trajectory = self.run(resident_id, simulation_date, use_only_previous_days=True)
        
        logging.info(f"Step 5: Generated trajectory with {len(predicted_trajectory)} activities for evaluation")
        
        # Extract predicted activities
        predicted_activities = [activity for _, _, activity in predicted_trajectory 
                               if not activity.startswith("finished ")]
        
        # Extract standardized predicted activity types
        predicted_types = set(self.standardize_activity_type(act.split(" ")[0]) 
                             for act in predicted_activities 
                             if act not in ["waking up", "morning routine", "returned home"])
        
        logging.info(f"Step 5: Comparing {len(predicted_types)} predicted activity types with {len(ground_truth_types)} ground truth types")
        
        # Calculate metrics
        # 1. Activity Type Accuracy
        activity_type_accuracy = 0
        if ground_truth_types:
            # Calculate intersection of sets
            matched_types = ground_truth_types.intersection(predicted_types)
            activity_type_accuracy = len(matched_types) / len(ground_truth_types)
        
        # 2. Activity Count Accuracy
        # How close is the number of predicted activities to the ground truth?
        activity_count = len(ground_truth)
        predicted_count = sum(1 for act in predicted_activities 
                              if act not in ["waking up", "morning routine", "returned home", 
                                            "commuting to work", "returning home"])
        
        count_ratio = min(predicted_count / max(1, activity_count), 
                           activity_count / max(1, predicted_count))
        
        # 3. Time Pattern Similarity
        # Extract time patterns from ground truth and predictions
        ground_truth_times = []
        for activity in ground_truth:
            parts = activity.split(" at ")
            if len(parts) >= 2:
                try:
                    time_obj = datetime.strptime(parts[1][:8], "%H:%M:%S").time()
                    ground_truth_times.append(time_obj)
                except (ValueError, IndexError):
                    pass
        
        predicted_times = [t.time() for _, t, _ in predicted_trajectory]
        
        time_similarity = 0
        if ground_truth_times and predicted_times:
            # Convert times to minutes since midnight for comparison
            gt_minutes = [t.hour * 60 + t.minute for t in ground_truth_times]
            pred_minutes = [t.hour * 60 + t.minute for t in predicted_times]
            
            # Compute simple distribution (Morning, Afternoon, Evening)
            periods = {"morning": (5, 12), "afternoon": (12, 18), "evening": (18, 23)}
            
            gt_period_counts = {period: 0 for period in periods}
            pred_period_counts = {period: 0 for period in periods}
            
            for minutes in gt_minutes:
                hour = minutes // 60
                for period, (start, end) in periods.items():
                    if start <= hour < end:
                        gt_period_counts[period] += 1
                        break
            
            for minutes in pred_minutes:
                hour = minutes // 60
                for period, (start, end) in periods.items():
                    if start <= hour < end:
                        pred_period_counts[period] += 1
                        break
            
            # Normalize the counts
            total_gt = sum(gt_period_counts.values())
            total_pred = sum(pred_period_counts.values())
            
            gt_distribution = {k: v/max(1, total_gt) for k, v in gt_period_counts.items()}
            pred_distribution = {k: v/max(1, total_pred) for k, v in pred_period_counts.items()}
            
            # Calculate time similarity based on period distributions
            diffs = [abs(gt_distribution[p] - pred_distribution[p]) for p in periods]
            time_similarity = 1 - (sum(diffs) / len(periods))
        
        # Combine into overall accuracy score
        # This is a simple weighted average - adjust weights as needed
        overall_accuracy = 0.5 * activity_type_accuracy + 0.3 * count_ratio + 0.2 * time_similarity
        
        logging.info(f"Step 5: Evaluation completed - Type Accuracy: {activity_type_accuracy:.2f}, Count Accuracy: {count_ratio:.2f}, Time Similarity: {time_similarity:.2f}")
        
        return {
            "resident_id": resident_id,
            "ground_truth_day": ground_truth_day,
            "history_days_count": history_days_count,
            "ground_truth_activities": list(ground_truth_types),
            "predicted_activities": list(predicted_types),
            "activity_type_accuracy": activity_type_accuracy,
            "activity_count_accuracy": count_ratio,
            "time_pattern_similarity": time_similarity,
            "overall_accuracy": overall_accuracy
        }

    def save_results(self, filename: str) -> None:
        """Save the simulation results to a file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Resident ID", "Simulation Date", "Location", "Duration (hours)", "Visit Time", "Activity"])
            
            # 用于跟踪已保存的居民和日期组合
            saved_simulations = set()
            
            # For each resident, save their trajectory
            for resident_id, resident in self.residents.items():
                # 如果居民没有轨迹，跳过
                if not resident.trajectory:
                    continue
                    
                # Determine simulation date from trajectory
                simulation_date = "Unknown"
                if resident.trajectory:
                    _, time_point, _ = resident.trajectory[0]
                    simulation_date = time_point.strftime("%Y-%m-%d")
                
                # 检查这个居民和日期组合是否已经保存过
                simulation_key = f"{resident_id}_{simulation_date}"
                if simulation_key in saved_simulations:
                    continue
                    
                saved_simulations.add(simulation_key)
                
                for i in range(len(resident.trajectory) - 1):
                    location, start_time, activity = resident.trajectory[i]
                    _, end_time, _ = resident.trajectory[i + 1]
                    
                    duration_hours = (end_time - start_time).total_seconds() / 3600
                    writer.writerow([
                        resident_id,
                        simulation_date,
                        location,
                        duration_hours,
                        start_time.strftime("%H:%M:%S"),
                        activity
                    ])
            
            # Add summary statistics
            writer.writerow([])
            writer.writerow(["--- Summary Statistics ---"])
            writer.writerow(["Total Residents", len(self.residents)])
            writer.writerow(["Total Locations", len(self.locations)])
            writer.writerow(["Simulation Duration", "1 day per resident"])
        
        logging.info(f"Results saved to {filename}")

    def _plot_timeline(self, output_dir: str) -> None:
        """Plot resident activity timelines."""
        plt.figure(figsize=(12, 8))
        
        for i, (resident_id, resident) in enumerate(self.residents.items()):
            # Sort trajectory by time
            trajectory = sorted(resident.trajectory, key=lambda x: x[1])
            
            locations = [loc for loc, _, _ in trajectory]
            times = [t for _, t, _ in trajectory]
            activities = [act for _, _, act in trajectory]
            
            # Plot timeline
            plt.plot(times, [i] * len(times), 'o-', label=f"Resident {resident_id}")
            
            # Add activity labels
            for j, (time, activity) in enumerate(zip(times, activities)):
                plt.text(time, i + 0.1, activity, fontsize=8, rotation=45)
        
        plt.yticks(range(len(self.residents)), [f"Resident {id}" for id in self.residents.keys()])
        plt.xlabel("Time of Day")
        plt.title("Resident Activity Timelines")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "timelines.png"))
        
    def _plot_activity_distribution(self, output_dir: str) -> None:
        """Plot distribution of activities across residents."""
        activity_counts = {}
        
        for resident in self.residents.values():
            for _, _, activity in resident.trajectory:
                if activity.startswith("finished "):
                    continue
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        activities = list(activity_counts.keys())
        counts = list(activity_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(activities, counts)
        plt.xlabel("Activity Type")
        plt.ylabel("Frequency")
        plt.title("Activity Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "activity_distribution.png"))
        
    def _plot_location_network(self, output_dir: str) -> None:
        """Plot network of resident movements between locations."""
        G = nx.DiGraph()
        
        # Add nodes for all locations with their coordinates
        location_set = set()
        location_coords = {}
        
        for resident in self.residents.values():
            for loc, _, _ in resident.trajectory:
                location_set.add(loc)
                
                # 尝试获取坐标
                if loc not in location_coords:
                    coords = self.get_location_coordinates(loc)
                    if coords:
                        location_coords[loc] = coords
                
        # 添加节点，使用真实坐标
        for loc in location_set:
            if loc in location_coords:
                G.add_node(loc, pos=location_coords[loc])
            else:
                G.add_node(loc)
                
        # Add edges for movements between locations
        for resident in self.residents.values():
            for i in range(len(resident.trajectory) - 1):
                loc1, _, _ = resident.trajectory[i]
                loc2, _, _ = resident.trajectory[i + 1]
                
                if loc1 != loc2:
                    if G.has_edge(loc1, loc2):
                        G[loc1][loc2]['weight'] += 1
                    else:
                        G.add_edge(loc1, loc2, weight=1)
        
        plt.figure(figsize=(12, 10))
        
        # 使用真实坐标而不是spring_layout
        pos = {}
        for node in G.nodes():
            if node in location_coords:
                # 调整坐标使其适合可视化
                lon, lat = location_coords[node]
                pos[node] = (lon, lat)
        
        # 对于没有坐标的节点，使用spring_layout
        if not pos or len(pos) < len(G.nodes()):
            remaining_nodes = [n for n in G.nodes() if n not in pos]
            if remaining_nodes:
                temp_pos = nx.spring_layout(G.subgraph(remaining_nodes))
                pos.update(temp_pos)
        
        # Node sizes based on visit frequency
        node_sizes = []
        for node in G.nodes():
            size = sum(1 for resident in self.residents.values() 
                      for loc, _, _ in resident.trajectory if loc == node)
            node_sizes.append(size * 50)  # 缩小节点尺寸以适应真实坐标
        
        # Edge widths based on transition frequency
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]  # 减小边宽度
        
        nx.draw_networkx(G, pos, with_labels=True, node_size=node_sizes, 
                         width=edge_widths, font_size=6, arrows=True, 
                         node_color='skyblue', edge_color='gray')
        
        plt.title("Location Transition Network (Real Coordinates)")
        plt.axis('on')  # 显示坐标轴以便查看真实坐标
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "location_network_real.png"))

    def get_location_coordinates(self, location_name: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a location from POI database.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Tuple of (longitude, latitude) if found, None otherwise
        """
        # 处理带编号的地点名称
        base_name = location_name.split("#")[0] if "#" in location_name else location_name
        
        # 1. 直接查找完整地点名
        for category, pois in self.poi_data.items():
            for poi in pois:
                if len(poi) > 2 and poi[2] == location_name:
                    return (float(poi[0]), float(poi[1]))
        
        # 2. 如果是居民的家或工作地点，使用默认值
        if location_name.startswith("Home_") or location_name.startswith("Workplace_"):
            # 使用东京的中心坐标作为默认值
            return (35.6762, 139.6503)
        
        # 3. 如果不是完整匹配，尝试基本名称匹配
        if base_name in self.poi_by_name:
            poi_options = self.poi_by_name[base_name]
            if poi_options:
                return poi_options[0]["coordinates"]
        
        # 4. 最后尝试类别匹配
        for category in self.poi_by_category:
            if base_name == category or base_name in category:
                poi_options = self.poi_by_category[category]
                if poi_options:
                    return poi_options[0]["coordinates"]
        
        # 如果所有方法都失败，返回None
        logging.warning(f"Could not find coordinates for location: {location_name}")
        return None

    def _plot_trajectory_map(self, output_dir: str) -> None:
        """
        Plot resident trajectories on a map using real coordinates.
        """
        plt.figure(figsize=(15, 10))
        
        # 为每个居民选择一个颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.residents)))
        
        for i, (resident_id, resident) in enumerate(self.residents.items()):
            # 提取轨迹中的位置
            locations = [loc for loc, _, _ in resident.trajectory]
            
            # 获取坐标
            coordinates = []
            for loc in locations:
                coords = self.get_location_coordinates(loc)
                if coords:
                    coordinates.append(coords)
            
            if len(coordinates) >= 2:  # 至少需要两个点才能绘制一条线
                # 分解坐标到纬度和经度列表
                lats, lons = zip(*coordinates)
                
                # 绘制轨迹线
                plt.plot(lons, lats, 'o-', color=colors[i % len(colors)], 
                         linewidth=1.5, markersize=4, alpha=0.7,
                         label=f"Resident {resident_id}")
                
                # 在第一个和最后一个位置添加标记
                plt.plot(lons[0], lats[0], 'o', color=colors[i % len(colors)], 
                         markersize=8, markeredgecolor='black')
                plt.plot(lons[-1], lats[-1], 's', color=colors[i % len(colors)], 
                         markersize=8, markeredgecolor='black')
        
        # 添加地图标题和标签
        plt.title("Resident Trajectories on Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例，但限制数量
        if len(self.residents) > 10:
            # 只显示前10个居民的图例
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        else:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "trajectory_map.png"), dpi=300)

    def visualize(self, output_dir: str = None) -> None:
        """Visualize the simulation results with multiple plots."""
        if output_dir is None:
            output_dir = os.path.join(SCRIPT_DIR, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Trajectory timeline plot
        self._plot_timeline(output_dir)
        
        # 2. Activity distribution plot
        self._plot_activity_distribution(output_dir)
        
        # 3. Network graph of locations
        self._plot_location_network(output_dir)
        
        # 4. Trajectory map plot
        self._plot_trajectory_map(output_dir)

def main() -> None:
    """Main function to set up and run the urban simulation."""
    # Set up simulation
    simulation = UrbanSimulation(random_seed=42)
    
    # Get list of available resident IDs
    resident_ids = list(simulation.resident_data.keys())
    
    if not resident_ids:
        logging.error("No resident data found. Cannot run simulation.")
        return
    
    # Randomly select 10 residents instead of using all residents
    if len(resident_ids) > 10:
        selected_resident_ids = random.sample(resident_ids, 10)
        logging.info(f"Randomly selected 10 residents from {len(resident_ids)} available residents")
    else:
        selected_resident_ids = resident_ids
        logging.info(f"Using all {len(resident_ids)} residents (fewer than 10 available)")
    
    # Collect evaluation results
    evaluation_results = []
    
    # 创建结果目录
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建多日轨迹文件
    multi_day_results_path = os.path.join(results_dir, "multi_day_results.csv")
    with open(multi_day_results_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Resident ID", "Simulation Date", "Location", "Duration (hours)", "Visit Time", "Activity"])
    
    # Run simulation and evaluation for each selected resident
    for resident_id in selected_resident_ids:
        logging.info(f"Simulating and evaluating resident {resident_id}...")
        
        # 初始化居民
        if resident_id not in simulation.residents:
            try:
                simulation.residents[resident_id] = simulation.initialize_resident(resident_id)
            except Exception as e:
                logging.error(f"Failed to initialize resident {resident_id}: {e}")
                continue
        
        resident = simulation.residents[resident_id]
        
        # 获取此居民的所有活动日期
        available_days = list(resident.historical_activities.keys())
        if not available_days or len(available_days) < 5:
            logging.warning(f"Resident {resident_id} has insufficient activity data (< 5 days). Skipping.")
            continue
        
        # 随机选择5个不同的日期，而不是10个
        if len(available_days) > 5:
            selected_days = random.sample(available_days, 5)
        else:
            selected_days = available_days
        
        logging.info(f"Selected {len(selected_days)} days for resident {resident_id}: {', '.join(selected_days)}")
        
        # 对每个选定的日期进行评估
        resident_evaluations = []
        for day in selected_days:
            # 评估该日期的活动
            evaluation_result = simulation.evaluate(resident_id, day)
            
            if "error" not in evaluation_result:
                # 只保留成功的评估
                resident_evaluations.append(evaluation_result)
                
                # 同时运行模拟以生成可视化数据
                simulation_date = datetime.strptime(evaluation_result["ground_truth_day"], "%Y-%m-%d")
                simulation.run(resident_id, simulation_date)
                
                # 将这一天的轨迹保存到多日轨迹文件中
                with open(multi_day_results_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    for i in range(len(resident.trajectory) - 1):
                        location, start_time, activity = resident.trajectory[i]
                        _, end_time, _ = resident.trajectory[i + 1]
                        
                        duration_hours = (end_time - start_time).total_seconds() / 3600
                        writer.writerow([
                            resident_id,
                            simulation_date.strftime("%Y-%m-%d"),
                            location,
                            duration_hours,
                            start_time.strftime("%H:%M:%S"),
                            activity
                        ])
        
        # 计算此居民的平均指标
        if resident_evaluations:
            avg_type_accuracy = sum(r["activity_type_accuracy"] for r in resident_evaluations) / len(resident_evaluations)
            avg_count_accuracy = sum(r["activity_count_accuracy"] for r in resident_evaluations) / len(resident_evaluations)
            avg_time_similarity = sum(r["time_pattern_similarity"] for r in resident_evaluations) / len(resident_evaluations)
            avg_overall = sum(r["overall_accuracy"] for r in resident_evaluations) / len(resident_evaluations)
            
            # 添加居民平均评估结果
            evaluation_results.append({
                "resident_id": resident_id,
                "days_evaluated": len(resident_evaluations),
                "activity_type_accuracy": avg_type_accuracy,
                "activity_count_accuracy": avg_count_accuracy,
                "time_pattern_similarity": avg_time_similarity, 
                "overall_accuracy": avg_overall
            })
            
            # 添加每日评估详情
            evaluation_results.extend(resident_evaluations)
    
    # Print summary of evaluation results
    if evaluation_results:
        # 计算仅包含居民平均指标的结果
        resident_avg_results = [r for r in evaluation_results if "ground_truth_day" not in r]
        
        logging.info(f"Successfully evaluated {len(resident_avg_results)} residents.")
        
        # Calculate average metrics across all residents
        avg_type_accuracy = sum(r["activity_type_accuracy"] for r in resident_avg_results) / len(resident_avg_results)
        avg_count_accuracy = sum(r["activity_count_accuracy"] for r in resident_avg_results) / len(resident_avg_results)
        avg_time_similarity = sum(r["time_pattern_similarity"] for r in resident_avg_results) / len(resident_avg_results)
        avg_overall = sum(r["overall_accuracy"] for r in resident_avg_results) / len(resident_avg_results)
        
        logging.info(f"Average activity type accuracy: {avg_type_accuracy:.4f}")
        logging.info(f"Average activity count accuracy: {avg_count_accuracy:.4f}")
        logging.info(f"Average time pattern similarity: {avg_time_similarity:.4f}")
        logging.info(f"Average overall accuracy: {avg_overall:.4f}")
        
        # 分别保存居民平均结果和每日详细结果
        # 1. 居民平均结果
        with open(os.path.join(results_dir, "resident_avg_evaluation.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Resident ID", "Days Evaluated",
                "Activity Type Accuracy", "Activity Count Accuracy", 
                "Time Pattern Similarity", "Overall Accuracy"
            ])
            
            for result in resident_avg_results:
                writer.writerow([
                    result["resident_id"],
                    result["days_evaluated"],
                    result["activity_type_accuracy"],
                    result["activity_count_accuracy"],
                    result["time_pattern_similarity"],
                    result["overall_accuracy"]
                ])
        
        # 2. 每日详细结果
        daily_results = [r for r in evaluation_results if "ground_truth_day" in r]
        with open(os.path.join(results_dir, "daily_evaluation_results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Resident ID", "Ground Truth Day", "History Days Count",
                "Activity Type Accuracy", "Activity Count Accuracy", 
                "Time Pattern Similarity", "Overall Accuracy"
            ])
            
            for result in daily_results:
                writer.writerow([
                    result["resident_id"],
                    result["ground_truth_day"],
                    result.get("history_days_count", 0),
                    result["activity_type_accuracy"],
                    result["activity_count_accuracy"],
                    result["time_pattern_similarity"],
                    result["overall_accuracy"]
                ])
                
        logging.info(f"Detailed evaluation results saved to {results_dir}")
    else:
        logging.warning("No successful evaluations were performed.")
    
    # Save and visualize results - 这仍然会保存每个居民的最后一次轨迹(单日)
    results_csv_path = os.path.join(SCRIPT_DIR, "results", "results.csv")
    simulation.save_results(results_csv_path)
    simulation.visualize(os.path.join(SCRIPT_DIR, "results"))
    
    logging.info(f"Simulation complete. Results saved to {os.path.join(SCRIPT_DIR, 'results')}")

# Execute main for both direct execution and sandbox wrapper invocation
main()