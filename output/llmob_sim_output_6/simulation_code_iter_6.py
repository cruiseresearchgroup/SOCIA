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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/llmob_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

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
        location = "Shopping Mall"  # This could be selected more intelligently
        self.current_location = location
        self.trajectory.append((self.current_location, start_time, "shopping"))
        self.trajectory.append((self.current_location, end_time, "finished shopping"))

    def exercise(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate exercising activity."""
        location = "Gym"  # This could be selected more intelligently
        self.current_location = location
        self.trajectory.append((self.current_location, start_time, "exercising"))
        self.trajectory.append((self.current_location, end_time, "finished exercising"))

    def socialize(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate socializing activity."""
        location = "Cafe"  # This could be selected more intelligently
        self.current_location = location
        self.trajectory.append((self.current_location, start_time, "socializing"))
        self.trajectory.append((self.current_location, end_time, "finished socializing"))

    def work(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate working activity."""
        self.current_location = self.work_location
        self.trajectory.append((self.current_location, start_time, "working"))
        self.trajectory.append((self.current_location, end_time, "finished working"))

    def study(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate studying activity."""
        location = "Library"
        self.current_location = location
        self.trajectory.append((self.current_location, start_time, "studying"))
        self.trajectory.append((self.current_location, end_time, "finished studying"))

    def dine(self, start_time: datetime, end_time: datetime) -> None:
        """Simulate dining activity."""
        location = "Restaurant"
        self.current_location = location
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
            category_data = load_json("catto.json")
            
            # Initialize locations
            location_id = 0
            for location_type, details in category_data.items():
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
        
        # Create resident attributes
        attributes = {
            "age": random.randint(18, 65),
            "gender": random.choice(["male", "female"]),
            "occupation": random.choice(["worker", "student"]),
            "home_location": f"Home_{resident_id}",
            "work_location": f"Workplace_{resident_id}",
            "transport_mode": random.choice(["car", "bike", "walking"])
        }
        
        return Resident(resident_id, attributes, historical_activities)
    
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
        
        # Generate the day's activity schedule based on history, pattern, and persona
        logging.info(f"Step 3: Generating day schedule based on {len(historical_activities)} historical activities")
        day_schedule = self.generate_day_schedule(resident, historical_activities, simulation_date)
        logging.info(f"Step 4: Generated schedule with {len(day_schedule)} activities for {simulation_date_str}")
        
        # Execute the day's schedule
        for i, activity in enumerate(day_schedule):
            motivation = activity["type"]
            start_time = datetime.combine(simulation_date.date(), 
                                         datetime.strptime(activity["start_time"], "%H:%M:%S").time())
            duration_minutes = activity["duration_minutes"]
            
            logging.info(f"Step 4: Scheduled activity {i+1}/{len(day_schedule)}: {motivation} at {start_time.strftime('%H:%M:%S')} for {duration_minutes} minutes")
            
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
            
            # Perform the scheduled activity
            logging.info(f"Step 4: Executing {motivation} activity")
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
        
        for activity in historical_activities:
            if " at " in activity:
                parts = activity.split(" at ")
                if len(parts) >= 2:
                    activity_type = parts[0].strip()
                    time_str = parts[1].strip()
                    
                    # Standardize activity types
                    std_type = self.standardize_activity_type(activity_type)
                    
                    if std_type:
                        if std_type not in activity_times:
                            activity_times[std_type] = []
                        try:
                            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
                            activity_times[std_type].append(time_obj)
                            if std_type not in activity_types:
                                activity_types.append(std_type)
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
                    
                    # Add activity to schedule with reasonable duration
                    schedule.append({
                        "type": act_type,
                        "start_time": time_str,
                        "duration_minutes": random.randint(45, 120),  # 45 mins to 2 hours
                        "location": self.get_location_for_activity(act_type, resident)
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
                    "location": self.get_location_for_activity(morning_act, resident)
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
                schedule.append({
                    "type": afternoon_act,
                    "start_time": "15:00:00",
                    "duration_minutes": 120,
                    "location": self.get_location_for_activity(afternoon_act, resident)
                })
            
            # Add evening activities for everyone
            evening_act = random.choice(["dining", "socializing", "resting"])
            schedule.append({
                "type": evening_act,
                "start_time": "18:00:00",
                "duration_minutes": 120,
                "location": self.get_location_for_activity(evening_act, resident)
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
        
    def get_location_for_activity(self, activity_type: str, resident: Resident) -> str:
        """Get appropriate location for a given activity type."""
        locations = {
            "shopping": "Shopping Mall",
            "exercising": "Gym",
            "socializing": "Cafe",
            "working": resident.work_location,
            "studying": "Library",
            "dining": "Restaurant",
            "resting": resident.home_location
        }
        
        return locations.get(activity_type, resident.home_location)

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
            
            # For each resident, save their trajectory
            for resident_id, resident in self.residents.items():
                # Determine simulation date from trajectory
                simulation_date = "Unknown"
                if resident.trajectory:
                    _, time_point, _ = resident.trajectory[0]
                    simulation_date = time_point.strftime("%Y-%m-%d")
                
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
    
    def visualize(self, output_dir: str = "results") -> None:
        """Visualize the simulation results with multiple plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Trajectory timeline plot
        self._plot_timeline(output_dir)
        
        # 2. Activity distribution plot
        self._plot_activity_distribution(output_dir)
        
        # 3. Network graph of locations
        self._plot_location_network(output_dir)

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
        
        # Add nodes for all locations
        location_set = set()
        for resident in self.residents.values():
            for loc, _, _ in resident.trajectory:
                location_set.add(loc)
                
        for loc in location_set:
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
        pos = nx.spring_layout(G)
        
        # Node sizes based on visit frequency
        node_sizes = []
        for node in G.nodes():
            size = sum(1 for resident in self.residents.values() 
                      for loc, _, _ in resident.trajectory if loc == node)
            node_sizes.append(size * 100)
        
        # Edge widths based on transition frequency
        edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
        
        nx.draw_networkx(G, pos, with_labels=True, node_size=node_sizes, 
                         width=edge_widths, font_size=8, arrows=True)
        
        plt.title("Location Transition Network")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "location_network.png"))

def main() -> None:
    """Main function to set up and run the urban simulation."""
    # Set up simulation
    simulation = UrbanSimulation(random_seed=42)
    
    # Get list of available resident IDs
    resident_ids = list(simulation.resident_data.keys())
    
    if not resident_ids:
        logging.error("No resident data found. Cannot run simulation.")
        return
    
    # Determine how many residents to evaluate
    total_residents = len(resident_ids)
    num_to_evaluate = min(total_residents, 50)  # Increase to 50 residents for evaluation
    
    # Randomly select resident IDs to simulate
    selected_residents = random.sample(resident_ids, num_to_evaluate)
    logging.info(f"Selected {num_to_evaluate} residents for simulation and evaluation.")
    
    # Collect evaluation results
    evaluation_results = []
    
    # Run simulation and evaluation for each selected resident
    for resident_id in selected_residents:
        logging.info(f"Simulating and evaluating resident {resident_id}...")
        
        # Evaluate the resident
        evaluation_result = simulation.evaluate(resident_id)
        
        if "error" not in evaluation_result:
            # Only keep successful evaluations
            evaluation_results.append(evaluation_result)
            
            # Also run the simulation to generate visualization data
            simulation_date = datetime.strptime(evaluation_result["ground_truth_day"], "%Y-%m-%d")
            simulation.run(resident_id, simulation_date)
    
    # Print summary of evaluation results
    if evaluation_results:
        logging.info(f"Successfully evaluated {len(evaluation_results)} residents.")
        
        # Calculate average metrics
        avg_type_accuracy = sum(r["activity_type_accuracy"] for r in evaluation_results) / len(evaluation_results)
        avg_count_accuracy = sum(r["activity_count_accuracy"] for r in evaluation_results) / len(evaluation_results)
        avg_time_similarity = sum(r["time_pattern_similarity"] for r in evaluation_results) / len(evaluation_results)
        avg_overall = sum(r["overall_accuracy"] for r in evaluation_results) / len(evaluation_results)
        
        logging.info(f"Average activity type accuracy: {avg_type_accuracy:.4f}")
        logging.info(f"Average activity count accuracy: {avg_count_accuracy:.4f}")
        logging.info(f"Average time pattern similarity: {avg_time_similarity:.4f}")
        logging.info(f"Average overall accuracy: {avg_overall:.4f}")
        
        # Save detailed evaluation results
        os.makedirs("results", exist_ok=True)
        with open("results/evaluation_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Resident ID", "Ground Truth Day", "History Days Count",
                "Activity Type Accuracy", "Activity Count Accuracy", 
                "Time Pattern Similarity", "Overall Accuracy"
            ])
            
            for result in evaluation_results:
                writer.writerow([
                    result["resident_id"],
                    result["ground_truth_day"],
                    result.get("history_days_count", 0),
                    result["activity_type_accuracy"],
                    result["activity_count_accuracy"],
                    result["time_pattern_similarity"],
                    result["overall_accuracy"]
                ])
                
        logging.info("Detailed evaluation results saved to results/evaluation_results.csv")
    else:
        logging.warning("No successful evaluations were performed.")
    
    # Save and visualize results
    simulation.save_results("results/results.csv")
    simulation.visualize()
    
    logging.info("Simulation complete. Results saved to results/")

# Execute main for both direct execution and sandbox wrapper invocation
main()