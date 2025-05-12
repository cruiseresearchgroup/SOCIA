import os
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from geopy.distance import geodesic
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/llmob_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Load data files
def load_json(file_name: str) -> Any:
    """Load a JSON file from the data directory."""
    try:
        with open(os.path.join(DATA_DIR, file_name), 'r') as file:
            return json.load(file)
    except (FileNotFoundError, IOError) as e:
        logging.error(f"Error loading {file_name}: {e}")
        return {}

# Entity: Resident
class Resident:
    """Represents an urban resident with attributes and behaviors."""
    
    def __init__(self, resident_id: str, attributes: Dict[str, Any], daily_activities: List[str]):
        self.resident_id = resident_id
        self.age = attributes.get("age")
        self.gender = attributes.get("gender")
        self.occupation = attributes.get("occupation")
        self.home_location = attributes.get("home_location")
        self.work_location = attributes.get("work_location")
        self.daily_schedule = daily_activities
        self.current_location = self.home_location

    def commute(self, time: datetime):
        """Simulate commuting behavior for the resident."""
        if self.current_location == self.home_location:
            self.current_location = self.work_location
        else:
            self.current_location = self.home_location

    def participate_in_activity(self, motivation: str, current_time: datetime):
        """Simulate activity participation based on daily schedule."""
        activities = {
            "shopping": self.shop,
            "exercising": self.exercise,
            "socializing": self.socialize,
            "working": self.work,
            "resting": self.rest
        }
        activity = activities.get(motivation)
        if activity:
            activity(current_time)

    def shop(self, current_time: datetime):
        """Simulate shopping activity."""
        # Example: Update resident state or interact with locations
        logging.info(f"{self.resident_id} is shopping at {self.current_location}.")

    def exercise(self, current_time: datetime):
        """Simulate exercising activity."""
        # Example: Update resident state or interact with locations
        logging.info(f"{self.resident_id} is exercising at {self.current_location}.")

    def socialize(self, current_time: datetime):
        """Simulate socializing activity."""
        # Example: Update resident state or interact with locations
        logging.info(f"{self.resident_id} is socializing at {self.current_location}.")

    def work(self, current_time: datetime):
        """Simulate working activity."""
        # Example: Update resident state or interact with locations
        logging.info(f"{self.resident_id} is working at {self.current_location}.")

    def rest(self, current_time: datetime):
        """Simulate resting activity."""
        # Example: Update resident state or interact with locations
        logging.info(f"{self.resident_id} is resting at {self.current_location}.")

# Entity: Location
class Location:
    """Represents a location with attributes and behaviors."""

    def __init__(self, location_type: str, address: str, capacity: int, operating_hours: str):
        self.type = location_type
        self.address = address
        self.capacity = capacity
        self.operating_hours = operating_hours
        self.current_occupancy = 0

    def host_activity(self) -> bool:
        """Host an activity at the location."""
        if self.current_occupancy < self.capacity:
            self.current_occupancy += 1
            return True
        return False

# Simulation Environment
class SimulationEnvironment:
    """Manages the simulation environment and timing."""

    def __init__(self, duration_days: int, time_step: int):
        self.current_time = datetime.now()
        self.end_time = self.current_time + timedelta(days=duration_days)
        self.time_step = timedelta(minutes=time_step)

    def advance_time(self):
        """Advance simulation time by one time step."""
        self.current_time += self.time_step

# Simulation Class
class UrbanSimulation:
    """Main class to coordinate the urban simulation."""

    def __init__(self, population_size: int, simulation_duration_days: int):
        self.population_size = population_size
        self.simulation_duration_days = simulation_duration_days
        self.residents = []
        self.locations = []
        self.environment = SimulationEnvironment(simulation_duration_days, 60)

    def load_data(self):
        """Load and initialize data for the simulation."""
        resident_data = load_json("2019Y.json")
        poi_data = load_json("poi_category_192021_longitude_latitude_complement_alignment_clean.json")
        category_data = load_json("catto.json")
        
        # Initialize residents
        for resident_id, activities in resident_data.items():
            attributes = {
                "age": random.randint(18, 65),
                "gender": random.choice(["male", "female"]),
                "occupation": random.choice(["worker", "student"]),
                "home_location": random.choice(list(poi_data.keys())),
                "work_location": random.choice(list(poi_data.keys()))
            }
            self.residents.append(Resident(resident_id, attributes, activities))
        
        # Initialize locations
        for location_type, details in category_data.items():
            for poi in poi_data.get(location_type, []):
                self.locations.append(Location(location_type, poi[2], random.randint(10, 100), "09:00-18:00"))

    def run(self):
        """Run the simulation."""
        while self.environment.current_time < self.environment.end_time:
            for resident in self.residents:
                resident.commute(self.environment.current_time)
                motivation = self.generate_motivation(resident)
                resident.participate_in_activity(motivation, self.environment.current_time)
            self.environment.advance_time()

    def evaluate(self):
        """Evaluate the simulation using specified metrics."""
        # Implement evaluation logic using metrics like Step distance, Step interval, and Jensen-Shannon divergence
        logging.info("Evaluating simulation...")

    def visualize(self):
        """Visualize the simulation results."""
        # Implement visualization logic
        logging.info("Visualizing results...")

    def save_results(self, filename: str):
        """Save the simulation results to a file."""
        # Implement logic to save results
        logging.info(f"Results saved to {filename}")

    def generate_motivation(self, resident: Resident) -> str:
        """Generate a daily motivation for the resident."""
        # Implement logic to generate motivation based on resident data
        return random.choice(["shopping", "exercising", "socializing", "working", "resting"])

def main():
    """Entry point for the simulation."""
    simulation = UrbanSimulation(population_size=1000, simulation_duration_days=30)
    simulation.load_data()
    simulation.run()
    simulation.evaluate()
    simulation.visualize()
    simulation.save_results("results.csv")

# Execute main
main()