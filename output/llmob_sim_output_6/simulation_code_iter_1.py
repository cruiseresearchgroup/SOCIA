import os
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from geopy.distance import geodesic
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/llmob_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Load data files
def load_json(file_name: str) -> Any:
    """Load a JSON file from the data directory."""
    if not os.path.isdir(DATA_DIR):
        logging.error(f"Data directory does not exist: {DATA_DIR}")
        return {}
    try:
        with open(os.path.join(DATA_DIR, file_name), 'r') as file:
            return json.load(file)
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_name}: {e}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_name}: {e}")
        return {}
    except IOError as e:
        logging.error(f"IO error loading {file_name}: {e}")
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

    def commute(self, time: datetime) -> None:
        """Simulate commuting behavior for the resident."""
        if self.current_location == self.home_location:
            self.current_location = self.work_location
        else:
            self.current_location = self.home_location

    def participate_in_activity(self, motivation: str, current_time: datetime) -> None:
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

    def shop(self, current_time: datetime) -> None:
        """Simulate shopping activity."""
        logging.info(f"{self.resident_id} is shopping at {self.current_location}.")

    def exercise(self, current_time: datetime) -> None:
        """Simulate exercising activity."""
        logging.info(f"{self.resident_id} is exercising at {self.current_location}.")

    def socialize(self, current_time: datetime) -> None:
        """Simulate socializing activity."""
        logging.info(f"{self.resident_id} is socializing at {self.current_location}.")

    def work(self, current_time: datetime) -> None:
        """Simulate working activity."""
        logging.info(f"{self.resident_id} is working at {self.current_location}.")

    def rest(self, current_time: datetime) -> None:
        """Simulate resting activity."""
        logging.info(f"{self.resident_id} is resting at {self.current_location}.")

# Entity: Location
class Location:
    """Represents a location with attributes and behaviors."""

    def __init__(self, location_type: str, address: str, capacity: int, operating_hours: str, coordinates: Tuple[float, float]):
        self.type = location_type
        self.address = address
        self.capacity = capacity
        self.operating_hours = operating_hours
        self.current_occupancy = 0
        self.coordinates = coordinates

    def host_activity(self, activity_type: str) -> bool:
        """Host an activity at the location if capacity allows."""
        if self.current_occupancy < self.capacity:
            self.current_occupancy += 1
            logging.info(f"Hosting {activity_type} at {self.address}. Current occupancy: {self.current_occupancy}")
            return True
        logging.warning(f"Cannot host {activity_type} at {self.address}. Capacity full.")
        return False

# Entity: Organization
class Organization:
    """Represents an organization with attributes and behaviors."""

    def __init__(self, org_type: str, employees: List[str], operating_hours: str):
        self.org_type = org_type
        self.employees = employees
        self.operating_hours = operating_hours

    def operate(self) -> None:
        """Operate the organization."""
        logging.info(f"Operating {self.org_type} with employees: {self.employees}")

    def schedule_events(self) -> None:
        """Schedule events within the organization."""
        logging.info(f"Scheduling events for {self.org_type}")

# Simulation Environment
class SimulationEnvironment:
    """Manages the simulation environment and timing."""

    def __init__(self, duration_days: int, time_step: int):
        self.current_time = datetime.now()
        self.end_time = self.current_time + timedelta(days=duration_days)
        self.time_step = timedelta(minutes=time_step)

    def advance_time(self) -> None:
        """Advance simulation time by one time step."""
        self.current_time += self.time_step

# Simulation Class
class UrbanSimulation:
    """Main class to coordinate the urban simulation."""

    def __init__(self, population_size: int, simulation_duration_days: int):
        self.population_size = population_size
        self.simulation_duration_days = simulation_duration_days
        self.residents: List[Resident] = []
        self.locations: List[Location] = []
        self.organizations: List[Organization] = []
        self.environment = SimulationEnvironment(simulation_duration_days, 10)  # Adjusted for 10-minute time step

    def load_data(self) -> None:
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
                coordinates = (float(poi[0]), float(poi[1]))
                self.locations.append(Location(location_type, poi[2], random.randint(10, 100), "09:00-18:00", coordinates))

        # Initialize organizations
        self.organizations.append(Organization("office", ["emp1", "emp2"], "09:00-17:00"))

    def run(self) -> None:
        """Run the simulation."""
        while self.environment.current_time < self.environment.end_time:
            for resident in self.residents:
                resident.commute(self.environment.current_time)
                motivation = self.generate_motivation(resident)
                resident.participate_in_activity(motivation, self.environment.current_time)
            self.environment.advance_time()

    def evaluate(self) -> None:
        """Evaluate the simulation using specified metrics."""
        logging.info("Evaluating simulation...")

    def visualize(self) -> None:
        """Visualize the simulation results."""
        logging.info("Visualizing results...")
        plt.figure()
        plt.title("Simulation Results")
        plt.show()

    def save_results(self, filename: str) -> None:
        """Save the simulation results to a file."""
        logging.info(f"Results saved to {filename}")
        with open(filename, 'w') as f:
            f.write("Sample results data")

    def generate_motivation(self, resident: Resident) -> str:
        """Generate a daily motivation for the resident."""
        context_motivations = self.derive_motivations_from_context(resident)
        return random.choices(context_motivations, weights=[0.2, 0.3, 0.1, 0.3, 0.1])[0]

    def derive_motivations_from_context(self, resident: Resident) -> List[str]:
        """Derive possible motivations based on context and historical data."""
        # Implement logic to analyze historical trajectory data
        historical_patterns = ["shopping", "exercising", "socializing", "working", "resting"]
        # Example enhancement: analyze trajectories to refine motivations
        # This needs access to real trajectory data and meaningful analysis
        return historical_patterns

    def calculate_shortest_path(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Calculate shortest path between two locations using Dijkstra's algorithm."""
        G = nx.Graph()
        # Assume nodes are created based on some logic, e.g., POIs data
        for location in self.locations:
            G.add_node(location.coordinates)
            # Add edges based on hypothetical distances, replace with real logic
            for other_location in self.locations:
                if location != other_location:
                    distance = geodesic(location.coordinates, other_location.coordinates).meters
                    G.add_edge(location.coordinates, other_location.coordinates, weight=distance)
        path = nx.dijkstra_path(G, source=start, target=end)
        return path

def main() -> None:
    """Entry point for the simulation."""
    simulation = UrbanSimulation(population_size=1000, simulation_duration_days=30)
    simulation.load_data()
    simulation.run()
    simulation.evaluate()
    simulation.visualize()
    simulation.save_results("results.csv")

# Execute main for both direct execution and sandbox wrapper invocation
main()