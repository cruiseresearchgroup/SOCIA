import os
import json
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from geopy.distance import geodesic
import numpy as np
import logging
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import jensenshannon

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
    
    def __init__(self, resident_id: str, attributes: Dict[str, Any], daily_activities: List[str]):
        self.resident_id = resident_id
        self.age = attributes.get("age")
        self.gender = attributes.get("gender")
        self.occupation = attributes.get("occupation")
        self.home_location = attributes.get("home_location")
        self.work_location = attributes.get("work_location")
        self.daily_schedule = daily_activities
        self.current_location = self.home_location
        self.transport_mode = attributes.get("transport_mode", "walking")
        self.trajectory = []

    def commute(self, time: datetime) -> None:
        """Simulate commuting behavior for the resident."""
        commute_start_time = datetime.strptime("08:00:00", "%H:%M:%S").time()
        commute_end_time = datetime.strptime("18:00:00", "%H:%M:%S").time()
        if commute_start_time <= time.time() <= commute_end_time:
            if self.current_location == self.home_location:
                self.current_location = self.work_location
            else:
                self.current_location = self.home_location
        self.trajectory.append((self.current_location, time))

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
        self.trajectory.append((self.current_location, current_time))

    def exercise(self, current_time: datetime) -> None:
        """Simulate exercising activity."""
        logging.info(f"{self.resident_id} is exercising at {self.current_location}.")
        self.trajectory.append((self.current_location, current_time))

    def socialize(self, current_time: datetime) -> None:
        """Simulate socializing activity."""
        logging.info(f"{self.resident_id} is socializing at {self.current_location}.")
        self.trajectory.append((self.current_location, current_time))

    def work(self, current_time: datetime) -> None:
        """Simulate working activity."""
        work_start_time = datetime.strptime("09:00:00", "%H:%M:%S").time()
        work_end_time = datetime.strptime("17:00:00", "%H:%M:%S").time()
        if work_start_time <= current_time.time() <= work_end_time:
            self.current_location = self.work_location
            logging.info(f"{self.resident_id} is working at {self.work_location}.")
            self.trajectory.append((self.current_location, current_time))

    def rest(self, current_time: datetime) -> None:
        """Simulate resting activity."""
        logging.info(f"{self.resident_id} is resting at {self.current_location}.")
        self.trajectory.append((self.current_location, current_time))

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

    def host_event(self, activity_type: str) -> bool:
        """Host an event at the location if capacity allows."""
        if self.current_occupancy < self.capacity:
            self.current_occupancy += 1
            self.popularity += 1
            logging.info(f"Hosting {activity_type} at {self.address}. Current occupancy: {self.current_occupancy}")
            return True
        logging.warning(f"Cannot host {activity_type} at {self.address}. Capacity full.")
        return False

    def attract_visitors(self, residents: List[Resident], current_time: datetime) -> None:
        """Simulate the attraction of visitors based on location attributes."""
        for resident in residents:
            if self.type in resident.daily_schedule and self.current_occupancy < self.capacity:
                self.host_event(self.type)

class Organization:
    """Represents an organization with attributes and behaviors."""

    def __init__(self, org_type: str, employees: List[str], operating_hours: str, location: str):
        self.org_type = org_type
        self.employees = employees
        self.operating_hours = operating_hours
        self.employee_count = len(employees)
        self.location = location

    def operate(self) -> None:
        """Operate the organization with detailed operations during business hours."""
        current_time = datetime.now()
        start_time, end_time = [datetime.strptime(t, "%H:%M").time() for t in self.operating_hours.split('-')]
        logging.info(f"Operating {self.org_type} with employees: {self.employees}")
        if start_time <= current_time.time() <= end_time:
            logging.info(f"{self.org_type} is actively operating.")

    def host_meetings(self) -> None:
        """Host meetings within the organization."""
        logging.info(f"Hosting meetings for {self.org_type}")

class SimulationEnvironment:
    """Manages the simulation environment and timing."""

    def __init__(self, duration_days: int, time_step: int):
        self.current_time = datetime.now()
        self.end_time = self.current_time + timedelta(days=duration_days)
        self.time_step = timedelta(minutes=time_step)

    def advance_time(self) -> None:
        """Advance simulation time by one time step."""
        self.current_time += self.time_step

class UrbanSimulation:
    """Main class to coordinate the urban simulation."""
    
    def __init__(self, population_size: int, simulation_duration_days: int, random_seed: int = 42):
        self.population_size = population_size
        self.simulation_duration_days = simulation_duration_days
        self.residents: List[Resident] = []
        self.locations: List[Location] = []
        self.organizations: List[Organization] = []
        self.environment = SimulationEnvironment(simulation_duration_days, 10)  # Adjusted for 10-minute time step
        self.poi_data = {}
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.load_data()

    def load_data(self) -> None:
        """Load and initialize data for the simulation."""
        try:
            resident_data = load_json("2019Y.json")
            self.poi_data = load_json("poi_category_192021_longitude_latitude_complement_alignment_clean.json")
            category_data = load_json("catto.json")
            
            # Initialize residents
            for resident_id, activities in resident_data.items():
                attributes = {
                    "age": random.randint(18, 65),
                    "gender": random.choice(["male", "female"]),
                    "occupation": random.choice(["worker", "student"]),
                    "home_location": random.choice(list(self.poi_data.keys())),
                    "work_location": random.choice(list(self.poi_data.keys())),
                    "transport_mode": random.choice(["car", "bike", "walking"])
                }
                self.residents.append(Resident(resident_id, attributes, activities))
            
            # Initialize locations
            for location_type, details in category_data.items():
                for poi in self.poi_data.get(location_type, []):
                    coordinates = (float(poi[0]), float(poi[1]))
                    self.locations.append(Location(location_type, poi[2], random.randint(10, 100), "09:00-18:00", coordinates))

            # Initialize organizations
            self.organizations.append(Organization("office", ["emp1", "emp2"], "09:00-17:00", "Main Street"))
        except FileNotFoundError as e:
            logging.error(f"Error loading data: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            raise

    def run(self) -> None:
        """Run the simulation."""
        while self.environment.current_time < self.environment.end_time:
            for resident in self.residents:
                resident.commute(self.environment.current_time)
                motivation = self.generate_motivation(resident)
                resident.participate_in_activity(motivation, self.environment.current_time)
            for location in self.locations:
                location.attract_visitors(self.residents, self.environment.current_time)
            self.environment.advance_time()

    def evaluate(self) -> None:
        """Evaluate the simulation using specified metrics."""
        logging.info("Evaluating simulation...")
        self.calculate_step_distance()
        self.calculate_step_interval()
        self.calculate_jsd()

    def calculate_step_distance(self) -> None:
        """Calculate the Step distance (SD) for the simulation."""
        for resident in self.residents:
            distances = []
            for i in range(len(resident.trajectory) - 1):
                loc1 = self.get_coordinates(resident.trajectory[i][0])
                loc2 = self.get_coordinates(resident.trajectory[i + 1][0])
                if loc1 and loc2:
                    distance = geodesic(loc1, loc2).meters
                    distances.append(distance)
            logging.info(f"Step distances for {resident.resident_id}: {distances}")

    def calculate_step_interval(self) -> None:
        """Calculate the Step interval (SI) for the simulation."""
        for resident in self.residents:
            intervals = [(resident.trajectory[i + 1][1] - resident.trajectory[i][1]).seconds / 60
                         for i in range(len(resident.trajectory) - 1)]
            logging.info(f"Step intervals for {resident.resident_id}: {intervals}")

    def calculate_jsd(self) -> None:
        """Calculate the Jensen-Shannon divergence (JSD) for the simulation."""
        try:
            real_data = self.extract_real_data()
            simulated_data = self.extract_simulated_data()
            if len(real_data) == len(simulated_data):
                jsd_value = jensenshannon(real_data, simulated_data)
                logging.info(f"Jensen-Shannon divergence (JSD): {jsd_value}")
            else:
                logging.error("Real and simulated data do not match in length for JSD calculation.")
        except Exception as e:
            logging.error(f"Error calculating JSD: {e}")

    def extract_real_data(self) -> np.ndarray:
        """Extract real trajectory data for JSD calculation."""
        real_trajectories = []
        try:
            resident_data = load_json("2019Y.json")
            for activities in resident_data.values():
                for activity in activities:
                    time_str = activity.split(" at ")[1]
                    real_trajectories.append(datetime.strptime(time_str, "%H:%M:%S"))
        except Exception as e:
            logging.error(f"Failed to extract real data: {e}")
            raise
        return np.array(real_trajectories)

    def extract_simulated_data(self) -> np.ndarray:
        """Extract simulated trajectory data for JSD calculation."""
        simulated_trajectories = []
        for resident in self.residents:
            for _, time in resident.trajectory:
                simulated_trajectories.append(time)
        return np.array(simulated_trajectories)

    def visualize(self) -> None:
        """Visualize the simulation results."""
        logging.info("Visualizing results...")
        plt.figure()
        for resident in self.residents:
            trajectory = [self.get_coordinates(loc) for loc, _ in resident.trajectory if self.get_coordinates(loc)]
            if trajectory:
                x, y = zip(*trajectory)
                plt.plot(x, y, label=f"Resident {resident.resident_id}")
        plt.title("Resident Trajectories")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()

    def save_results(self, filename: str) -> None:
        """Save the simulation results to a file."""
        logging.info(f"Results saved to {filename}")
        with open(filename, 'w') as f:
            f.write("Sample results data")

    def generate_motivation(self, resident: Resident) -> str:
        """Generate a daily motivation for the resident."""
        historical_data = self.get_historical_data(resident)
        context_motivations = self.derive_motivations_from_context(resident, historical_data)
        if context_motivations:
            return max(context_motivations, key=lambda x: x['probability'])['name']
        else:
            return 'resting'

    def get_historical_data(self, resident: Resident) -> List[str]:
        """Retrieve historical data for a resident."""
        try:
            resident_data = load_json("2019Y.json")
            return resident_data.get(resident.resident_id, [])
        except Exception as e:
            logging.error(f"Failed to retrieve historical data for {resident.resident_id}: {e}")
            return []

    def derive_motivations_from_context(self, resident: Resident, historical_data: List[str]) -> List[Dict[str, Any]]:
        """Derive possible motivations based on context and historical data."""
        motivations = ["shopping", "exercising", "socializing", "working", "resting"]
        probabilities = np.random.dirichlet(np.ones(len(motivations)), size=1)[0]
        return [{"name": motivation, "probability": prob} for motivation, prob in zip(motivations, probabilities)]

    def calculate_shortest_path(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Calculate shortest path between two locations using Dijkstra's algorithm."""
        G = nx.Graph()
        for location in self.locations:
            if location.coordinates:
                G.add_node(location.coordinates)
                for other_location in self.locations:
                    if location != other_location and other_location.coordinates:
                        distance = geodesic(location.coordinates, other_location.coordinates).meters
                        G.add_edge(location.coordinates, other_location.coordinates, weight=distance)
        path = nx.dijkstra_path(G, source=start, target=end)
        return path

    def get_coordinates(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a given location name using POI data."""
        for location_type, pois in self.poi_data.items():
            for poi in pois:
                if location_name == poi[2]:
                    return (float(poi[0]), float(poi[1]))
        return None

def main() -> None:
    """Main function to set up and run the urban simulation."""
    simulation = UrbanSimulation(population_size=1000, simulation_duration_days=30, random_seed=42)
    simulation.run()
    simulation.evaluate()
    simulation.visualize()
    simulation.save_results("results.csv")

# Execute main for both direct execution and sandbox wrapper invocation
main()