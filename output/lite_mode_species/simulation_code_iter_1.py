import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend suitable for servers
import matplotlib.pyplot as plt
from typing import List, Optional, Any

# Path setup with error handling
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")

def get_data_dir() -> str:
    """
    Returns the full data directory path. Raises a clear error if environment variables are missing.
    """
    if PROJECT_ROOT is None:
        raise EnvironmentError("Environment variable PROJECT_ROOT is not set.")
    if DATA_PATH is None:
        raise EnvironmentError("Environment variable DATA_PATH is not set.")
    return os.path.join(PROJECT_ROOT, DATA_PATH)

try:
    DATA_DIR = get_data_dir()
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception as e:
    # Let the exception propagate for sandbox and wrapper compatibility
    raise

result_path = os.path.join(DATA_DIR, "results.csv")
picture_path = os.path.join(DATA_DIR, "figure.png")

class PlantSpecies:
    """
    Represents a plant species in the simulation.

    Parameters
    ----------
    name : str
        Name of the species.
    intrinsic_growth_rate : float
        The intrinsic growth rate (r).
    carrying_capacity : float
        The maximum population sustainable (K).
    water_need : float
        Proportion of total water the species would optimally use (relative scale).
    sunlight_need : float
        Proportion of total sunlight the species would optimally use (relative scale).

    Attributes
    ----------
    population : float
        Current population of the species (set during simulation initialization).
    """
    def __init__(
        self,
        name: str,
        intrinsic_growth_rate: float,
        carrying_capacity: float,
        water_need: float,
        sunlight_need: float
    ) -> None:
        self.name: str = name
        self.r: float = intrinsic_growth_rate
        self.K: float = carrying_capacity
        self.water_need: float = water_need
        self.sunlight_need: float = sunlight_need
        self.population: float = 0.0  # Always a float, set during simulation

class DesertCompetitionSimulation:
    """
    Simulates competition among multiple plant species for water and sunlight in a desert environment.

    Parameters
    ----------
    species_list : List[PlantSpecies]
        List of PlantSpecies instances.
    initial_populations : List[float]
        Initial populations for each species.
    total_water : float
        Total water resource available.
    total_sunlight : float
        Total sunlight resource available.
    competition_matrix : np.ndarray
        Inter-species competition coefficients (species x species).
    timesteps : int
        Number of time steps to simulate.

    Attributes
    ----------
    history : np.ndarray
        Stores populations at each timestep as a NumPy array (timesteps+1, n_species).
    """
    def __init__(
        self,
        species_list: List[PlantSpecies],
        initial_populations: List[float],
        total_water: float,
        total_sunlight: float,
        competition_matrix: np.ndarray,
        timesteps: int = 100
    ) -> None:
        self.species: List[PlantSpecies] = species_list
        self.total_water: float = total_water
        self.total_sunlight: float = total_sunlight
        self.timesteps: int = timesteps
        self.n_species: int = len(species_list)
        self.competition_matrix: np.ndarray = competition_matrix
        for i, pop in enumerate(initial_populations):
            self.species[i].population = float(pop)
        # Preallocate history as a NumPy array for efficiency
        self.history: np.ndarray = np.zeros((self.timesteps + 1, self.n_species), dtype=float)

    def resource_limitation_factor(self, populations: np.ndarray) -> np.ndarray:
        """
        Computes per-species limitation factors based on water and sunlight availability.

        Parameters
        ----------
        populations : np.ndarray
            Current population sizes for all species.

        Returns
        -------
        np.ndarray
            An array of limitation factors (0-1) for each species.
        """
        # Calculate demand for each species
        water_demand = np.array([sp.water_need * pop for sp, pop in zip(self.species, populations)])
        sunlight_demand = np.array([sp.sunlight_need * pop for sp, pop in zip(self.species, populations)])

        total_water_demand = water_demand.sum()
        total_sunlight_demand = sunlight_demand.sum()

        # Avoid division by zero
        if total_water_demand > 0:
            water_alloc = np.minimum(water_demand, self.total_water * (water_demand / total_water_demand))
            water_factor = np.divide(water_alloc, water_demand, out=np.ones_like(water_demand), where=water_demand!=0)
        else:
            water_factor = np.ones_like(water_demand)

        if total_sunlight_demand > 0:
            sunlight_alloc = np.minimum(sunlight_demand, self.total_sunlight * (sunlight_demand / total_sunlight_demand))
            sunlight_factor = np.divide(sunlight_alloc, sunlight_demand, out=np.ones_like(sunlight_demand), where=sunlight_demand!=0)
        else:
            sunlight_factor = np.ones_like(sunlight_demand)

        # Each species limited by their most limiting resource
        limitation_factors = np.minimum(water_factor, sunlight_factor)
        limitation_factors = np.clip(limitation_factors, 0.0, 1.0)
        return limitation_factors

    def step(self, populations: np.ndarray) -> np.ndarray:
        """
        Advances the simulation by one timestep using the generalized Lotka-Volterra competition model,
        incorporating resource limitation.

        Parameters
        ----------
        populations : np.ndarray
            Current populations.

        Returns
        -------
        np.ndarray
            Updated populations after one timestep.
        """
        limitation_factors = self.resource_limitation_factor(populations)
        # Vectorized competition sum: matrix multiply competition_matrix (n_species x n_species) by populations (n_species,)
        competition_sums = np.dot(self.competition_matrix, populations)
        r_vec = np.array([sp.r for sp in self.species])
        K_vec = np.array([sp.K for sp in self.species])

        growth = r_vec * populations * (1 - competition_sums / K_vec)
        growth = growth * limitation_factors
        new_populations = populations + growth
        new_populations = np.maximum(new_populations, 0.0)
        return new_populations

    def run(self) -> pd.DataFrame:
        """
        Runs the simulation for the set number of timesteps.

        Returns
        -------
        pd.DataFrame
            DataFrame with population history, each column is a species.
        """
        populations = np.array([sp.population for sp in self.species], dtype=float)
        self.history[0, :] = populations
        for t in range(1, self.timesteps + 1):
            populations = self.step(populations)
            self.history[t, :] = populations
        columns = [sp.name for sp in self.species]
        df = pd.DataFrame(self.history, columns=columns)
        return df

    def save_results(self, result_path: str) -> None:
        """
        Saves the simulation results to a CSV file.

        Parameters
        ----------
        result_path : str
            Path to save the CSV file.
        """
        columns = [sp.name for sp in self.species]
        df = pd.DataFrame(self.history, columns=columns)
        try:
            df.to_csv(result_path, index=False)
        except Exception as e:
            print(f"Error saving results to '{result_path}': {e}", file=sys.stderr)

    def plot(self, picture_path: str) -> None:
        """
        Plots the population dynamics and saves the figure.

        Parameters
        ----------
        picture_path : str
            Path to save the figure.
        """
        try:
            plt.figure(figsize=(8, 5))
            time_points = np.arange(self.timesteps + 1)
            for i, sp in enumerate(self.species):
                plt.plot(time_points, self.history[:, i], label=sp.name)
            plt.xlabel("Time Step")
            plt.ylabel("Population")
            plt.title("Plant Species Competition in a Desert")
            plt.legend()
            plt.tight_layout()
            plt.savefig(picture_path)
            plt.close()
        except Exception as e:
            print(f"Error saving plot to '{picture_path}': {e}", file=sys.stderr)

def main() -> None:
    """
    Sets up and executes the desert plant competition simulation.

    Simulation details:
    - Three species (Cactus, DesertGrass, Shrub) are initialized with specific growth parameters.
    - Each species starts with a given initial population.
    - The simulation runs for a specified number of timesteps, tracking population changes due to logistic growth,
      inter-species competition, and competition for limited water and sunlight.
    - Results are saved to CSV and a population dynamics plot is generated.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Example parameterization for three species
    species_data = [
        # name, r, K, water_need, sunlight_need
        ("Cactus", 0.11, 120, 1.0, 0.6),
        ("DesertGrass", 0.16, 90, 0.8, 1.0),
        ("Shrub", 0.09, 70, 1.2, 0.9)
    ]
    species_list = [PlantSpecies(*params) for params in species_data]
    initial_populations = [30, 25, 15]

    # Total available resources (arbitrary units)
    total_water = 100.0
    total_sunlight = 100.0

    # Competition coefficients: diagonal=1 (intraspecific), off-diagonal <1 (interspecific)
    competition_matrix = np.array([
        [1.0, 0.45, 0.35],
        [0.42, 1.0, 0.40],
        [0.30, 0.38, 1.0]
    ], dtype=float)

    timesteps = 80

    sim = DesertCompetitionSimulation(
        species_list=species_list,
        initial_populations=initial_populations,
        total_water=total_water,
        total_sunlight=total_sunlight,
        competition_matrix=competition_matrix,
        timesteps=timesteps
    )
    result_df = sim.run()
    sim.save_results(result_path)
    sim.plot(picture_path)

# Always execute main for direct or wrapper invocation
main()