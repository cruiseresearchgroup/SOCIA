# main.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from entities import Person
from environment import Environment
from typing import Dict

# Path setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class Simulation:
    """
    Manages the entire simulation process, including initialization, execution, and result collection.
    """
    def __init__(self, model_plan: Dict):
        self.environment = Environment(
            size=model_plan["environment"]["dimensions"],
            population=model_plan["parameters"]["population_size"]
        )
        self.environment.initialize_population(
            initial_infected=model_plan["parameters"]["initial_infected"],
            infection_probability=model_plan["parameters"]["infection_probability"],
            recovery_time=model_plan["parameters"]["recovery_time"]
        )
        self.prediction_period = model_plan["prediction_period"]
        self.evaluation_metrics = model_plan["evaluation_metrics"]
        self.results = defaultdict(list)

    def run(self) -> None:
        """
        Executes the simulation over the specified prediction period.
        """
        start_day = self.prediction_period.get("start_day", 0)
        end_day = self.prediction_period.get("end_day", 100)

        for day in range(start_day, end_day + 1):
            self.environment.step()
            self.record_metrics(day)

    def record_metrics(self, day: int) -> None:
        """
        Records the metrics for the current day.
        """
        counts = {"susceptible": 0, "infected": 0, "recovered": 0}
        for person in self.environment.people:
            counts[person.health_status] += 1

        self.results["day"].append(day)
        self.results["susceptible"].append(counts["susceptible"])
        self.results["infected"].append(counts["infected"])
        self.results["recovered"].append(counts["recovered"])

    def evaluate(self) -> dict:
        """
        Computes the evaluation metrics based on the simulation results.
        """
        metrics_results = {}
        total_days = len(self.results["day"])
        total_population = self.environment.population

        # Infection rate calculation
        total_infected = sum(self.results["infected"])
        metrics_results["infection_rate"] = total_infected / (total_population * total_days)

        # Recovery rate calculation
        total_recovered = sum(self.results["recovered"])
        metrics_results["recovery_rate"] = total_recovered / total_infected if total_infected > 0 else 0

        # Peak infection day calculation
        peak_infection_day = self.results["infected"].index(max(self.results["infected"]))
        metrics_results["peak_infection_day"] = self.results["day"][peak_infection_day]

        return metrics_results

    def visualize(self) -> None:
        """
        Visualizes the simulation results.
        """
        plt.plot(self.results["day"], self.results["susceptible"], label="Susceptible")
        plt.plot(self.results["day"], self.results["infected"], label="Infected")
        plt.plot(self.results["day"], self.results["recovered"], label="Recovered")
        plt.xlabel('Day')
        plt.ylabel('Number of People')
        plt.title('Epidemic Simulation Results')
        plt.legend()
        plt.show()

    def save_results(self, filename: str) -> None:
        """
        Saves the simulation results to a CSV file.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            with open(filename, 'w') as file:
                file.write("day,susceptible,infected,recovered\n")
                for day, s, i, r in zip(self.results["day"], self.results["susceptible"], self.results["infected"], self.results["recovered"]):
                    file.write(f"{day},{s},{i},{r}\n")
        except (IOError, FileNotFoundError, Exception) as e:
            print(f"An error occurred while writing to the file: {e}")

def main() -> None:
    """
    Main function to execute the simulation. It sets up the model plan, runs the simulation, visualizes the results,
    and saves them to a file. The model parameters define the environment size, population characteristics, and 
    simulation timeframe.
    """
    model_plan = {
        "environment": {"dimensions": (50, 50)},
        "parameters": {
            "population_size": 1000,
            "initial_infected": 1,
            "infection_probability": 0.05,
            "recovery_time": 14
        },
        "prediction_period": {"start_day": 0, "end_day": 100},
        "evaluation_metrics": ["infection_rate", "recovery_rate", "peak_infection_day"]
    }

    simulation = Simulation(model_plan)
    simulation.run()
    simulation.visualize()
    simulation.save_results(os.path.join(DATA_DIR, "results.csv"))


# Execute main for both direct execution and sandbox wrapper invocation
main()