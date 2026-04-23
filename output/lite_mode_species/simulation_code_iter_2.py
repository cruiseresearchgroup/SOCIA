"""
Spatial Desert Plant Competition Simulation

This module simulates competition among multiple plant species for water and sunlight
in a spatially explicit desert environment. It features logistic population growth,
inter-species competition, explicit resource accounting, and spatial dispersal on a
grid. Results are saved as CSV and visualized in a PNG.

Key classes:
- PlantSpecies: Defines species' ecological parameters.
- DesertGridCell: Represents a grid cell with resources and resident populations.
- DesertCompetitionSimulation: Orchestrates the simulation, resource allocation,
  competition, and dispersal.

Usage:
- Run this file directly to execute the default simulation scenario.
- Adjust configuration via environment variables or main() arguments.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend suitable for servers
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy.ndimage import convolve  # For vectorized dispersal

# Improved path setup with better error message and sensible defaults
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data")
if not os.path.isabs(DATA_PATH):
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
else:
    DATA_DIR = DATA_PATH
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception as e:
    print(f"Error creating data directory '{DATA_DIR}': {e}", file=sys.stderr)
    sys.exit(1)
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
        self.name = name
        self.r = intrinsic_growth_rate
        self.K = carrying_capacity
        self.water_need = water_need
        self.sunlight_need = sunlight_need
        self.population = 0.0

class DesertGridCell:
    """
    Represents a cell in the spatial desert grid.

    Parameters
    ----------
    i : int
        Row index.
    j : int
        Column index.
    water : float
        Water resource available in this cell.
    sunlight : float
        Sunlight resource available in this cell.
    n_species : int
        Number of plant species in the simulation.

    Attributes
    ----------
    i : int
    j : int
    water : float
    sunlight : float
    populations : np.ndarray
        Array of species populations in this cell.
    """
    def __init__(self, i: int, j: int, water: float, sunlight: float, n_species: int) -> None:
        self.i = i
        self.j = j
        self.water = water
        self.sunlight = sunlight
        self.populations = np.zeros(n_species, dtype=float)

class DesertCompetitionSimulation:
    """
    Simulates competition among multiple plant species for water and sunlight in a spatially explicit desert environment.

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
    grid_size : Tuple[int, int]
        Size of the desert grid (rows, columns).

    Attributes
    ----------
    grid : List[List[DesertGridCell]]
        2D list representing the spatial grid.
    history : dict
        Stores population and resource histories for analysis and plotting.
    """
    def __init__(
        self,
        species_list: List[PlantSpecies],
        initial_populations: List[float],
        total_water: float,
        total_sunlight: float,
        competition_matrix: np.ndarray,
        timesteps: int = 100,
        grid_size: Tuple[int, int] = (6, 6)
    ) -> None:
        self.species = species_list
        self.n_species = len(species_list)
        self.timesteps = timesteps
        self.competition_matrix = competition_matrix
        self.grid_rows, self.grid_cols = grid_size

        # Divide resources spatially (add heterogeneity)
        rng = np.random.default_rng(seed=42)
        water_grid = rng.uniform(0.7, 1.3, (self.grid_rows, self.grid_cols))
        sunlight_grid = rng.uniform(0.8, 1.2, (self.grid_rows, self.grid_cols))
        water_grid = water_grid / water_grid.sum() * total_water
        sunlight_grid = sunlight_grid / sunlight_grid.sum() * total_sunlight

        self.grid = []
        for i in range(self.grid_rows):
            row = []
            for j in range(self.grid_cols):
                cell = DesertGridCell(
                    i, j,
                    water=float(water_grid[i, j]),
                    sunlight=float(sunlight_grid[i, j]),
                    n_species=self.n_species
                )
                row.append(cell)
            self.grid.append(row)

        # Place initial populations: cluster each species in a random quadrant
        placed = np.zeros((self.grid_rows, self.grid_cols, self.n_species), dtype=float)
        for s_index, count in enumerate(initial_populations):
            # Pick a cluster center for each species
            center_i = rng.integers(1, self.grid_rows-1)
            center_j = rng.integers(1, self.grid_cols-1)
            cluster_cells = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni = min(max(center_i + di, 0), self.grid_rows-1)
                    nj = min(max(center_j + dj, 0), self.grid_cols-1)
                    cluster_cells.append((ni, nj))
            # Distribute initial population randomly within cluster
            splits = rng.dirichlet(np.ones(len(cluster_cells)))
            for idx, (ci, cj) in enumerate(cluster_cells):
                placed[ci, cj, s_index] += splits[idx] * count
        # Assign initial populations to cells
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                self.grid[i][j].populations = placed[i, j, :]

        # For tracking total and per-cell resource usage
        self.history = {
            'populations': np.zeros((timesteps+1, self.n_species)),
            'water_used': np.zeros((timesteps+1,)),
            'sunlight_used': np.zeros((timesteps+1,)),
            'water_available': np.zeros((timesteps+1,)),
            'sunlight_available': np.zeros((timesteps+1,))
        }
        self._update_species_pop_history(0)
        self._update_resource_history(0)

    def _update_species_pop_history(self, t: int) -> None:
        """
        Update population history for all species at time t.

        Parameters
        ----------
        t : int
            The current time step.

        Returns
        -------
        None
        """
        total_pops = np.zeros(self.n_species)
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                total_pops += self.grid[i][j].populations
        self.history['populations'][t, :] = total_pops

    def _update_resource_history(self, t: int) -> None:
        """
        Update total water/sunlight used and available at time t.

        Parameters
        ----------
        t : int
            The current time step.

        Returns
        -------
        None
        """
        total_water = 0.0
        total_sun = 0.0
        used_water = 0.0
        used_sun = 0.0
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                cell = self.grid[i][j]
                total_water += cell.water
                total_sun += cell.sunlight
                # Calculate demand
                water_demand = np.sum([sp.water_need * cell.populations[s_idx] for s_idx, sp in enumerate(self.species)])
                sunlight_demand = np.sum([sp.sunlight_need * cell.populations[s_idx] for s_idx, sp in enumerate(self.species)])
                used_water += min(water_demand, cell.water)
                used_sun += min(sunlight_demand, cell.sunlight)
        self.history['water_used'][t] = used_water
        self.history['sunlight_used'][t] = used_sun
        self.history['water_available'][t] = total_water
        self.history['sunlight_available'][t] = total_sun

    def resource_limitation_factor(self, cell: 'DesertGridCell') -> np.ndarray:
        """
        Computes per-species limitation factors based on water and sunlight availability within a cell.

        Parameters
        ----------
        cell : DesertGridCell
            The grid cell to compute limitation factors for.

        Returns
        -------
        np.ndarray
            Array of limitation factors (0-1) for each species in the cell.
        """
        water_demand = np.array([sp.water_need * cell.populations[s_idx] for s_idx, sp in enumerate(self.species)])
        sunlight_demand = np.array([sp.sunlight_need * cell.populations[s_idx] for s_idx, sp in enumerate(self.species)])
        total_water_demand = water_demand.sum()
        total_sunlight_demand = sunlight_demand.sum()
        water_alloc = np.zeros_like(water_demand)
        sunlight_alloc = np.zeros_like(sunlight_demand)
        if total_water_demand > 0:
            water_alloc = np.minimum(water_demand, cell.water * (water_demand / total_water_demand))
            water_factor = np.divide(water_alloc, water_demand, out=np.ones_like(water_demand), where=water_demand!=0)
        else:
            water_factor = np.ones_like(water_demand)
        if total_sunlight_demand > 0:
            sunlight_alloc = np.minimum(sunlight_demand, cell.sunlight * (sunlight_demand / total_sunlight_demand))
            sunlight_factor = np.divide(sunlight_alloc, sunlight_demand, out=np.ones_like(sunlight_demand), where=sunlight_demand!=0)
        else:
            sunlight_factor = np.ones_like(sunlight_demand)
        limitation_factors = np.minimum(water_factor, sunlight_factor)
        limitation_factors = np.clip(limitation_factors, 0.0, 1.0)
        return limitation_factors

    def step(self) -> None:
        """
        Advances the simulation by one timestep for all grid cells.

        Returns
        -------
        None
        """
        new_grid_populations: np.ndarray = np.zeros((self.grid_rows, self.grid_cols, self.n_species), dtype=float)
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                cell: DesertGridCell = self.grid[i][j]
                pops: np.ndarray = cell.populations
                limitation_factors: np.ndarray = self.resource_limitation_factor(cell)
                competition_sums: np.ndarray = np.dot(self.competition_matrix, pops)
                r_vec: np.ndarray = np.array([sp.r for sp in self.species])
                K_vec: np.ndarray = np.array([sp.K for sp in self.species])
                growth: np.ndarray = r_vec * pops * (1 - competition_sums / K_vec)
                growth = growth * limitation_factors
                new_pops: np.ndarray = pops + growth
                new_pops = np.maximum(new_pops, 0.0)
                new_grid_populations[i, j, :] = new_pops

        # Vectorized dispersal using scipy.ndimage.convolve
        dispersal_rate = 0.1
        kernel = np.ones((3, 3), dtype=float)
        kernel[1, 1] = 0.0  # Exclude center
        kernel /= kernel.sum()  # Each neighbor gets 1/8

        for s_idx in range(self.n_species):
            dispersal: np.ndarray = dispersal_rate * new_grid_populations[:, :, s_idx]
            retained: np.ndarray = new_grid_populations[:, :, s_idx] - dispersal
            distributed: np.ndarray = convolve(dispersal, kernel, mode='constant', cval=0.0)
            new_grid_populations[:, :, s_idx] = retained + distributed

        # Update cell populations
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                self.grid[i][j].populations = new_grid_populations[i, j, :]

    def run(self) -> pd.DataFrame:
        """
        Runs the simulation for the set number of timesteps.

        Returns
        -------
        pd.DataFrame
            DataFrame with total population history, each column is a species.
        """
        for t in range(1, self.timesteps + 1):
            self.step()
            self._update_species_pop_history(t)
            self._update_resource_history(t)
        columns = [sp.name for sp in self.species]
        df = pd.DataFrame(self.history['populations'], columns=columns)
        return df

    def save_results(self, result_path: str) -> None:
        """
        Saves the simulation results to a CSV file, including resources.

        Parameters
        ----------
        result_path : str
            Path to save the CSV file.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If saving the file fails.
        """
        columns = [sp.name for sp in self.species]
        df = pd.DataFrame(self.history['populations'], columns=columns)
        df['WaterUsed'] = self.history['water_used']
        df['SunlightUsed'] = self.history['sunlight_used']
        df['WaterAvailable'] = self.history['water_available']
        df['SunlightAvailable'] = self.history['sunlight_available']
        try:
            df.to_csv(result_path, index=False)
        except Exception as e:
            print(f"Error saving results to '{result_path}': {e}", file=sys.stderr)
            raise

    def plot(self, picture_path: str) -> None:
        """
        Plots the population and resource dynamics and saves the figure.

        Parameters
        ----------
        picture_path : str
            Path to save the figure.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If saving the plot fails.
        """
        try:
            plt.figure(figsize=(10, 7))
            time_points = np.arange(self.timesteps + 1)
            # Population plot
            ax1 = plt.subplot(2, 1, 1)
            for i, sp in enumerate(self.species):
                plt.plot(time_points, self.history['populations'][:, i], label=sp.name)
            plt.ylabel("Total Population")
            plt.title("Plant Species Competition in a Desert (Spatial Model)")
            plt.legend()
            # Resource plot
            ax2 = plt.subplot(2, 1, 2)
            plt.plot(time_points, self.history['water_available'], label='Water Available', color='blue', linestyle='dashed')
            plt.plot(time_points, self.history['water_used'], label='Water Used', color='blue')
            plt.plot(time_points, self.history['sunlight_available'], label='Sunlight Available', color='orange', linestyle='dashed')
            plt.plot(time_points, self.history['sunlight_used'], label='Sunlight Used', color='orange')
            plt.xlabel("Time Step")
            plt.ylabel("Resource (units)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(picture_path)
            plt.close()
        except Exception as e:
            print(f"Error saving plot to '{picture_path}': {e}", file=sys.stderr)
            raise

def main(
    project_root: Optional[str] = None,
    data_path: Optional[str] = None,
    result_csv: Optional[str] = None,
    figure_png: Optional[str] = None
) -> None:
    """
    Sets up and executes the spatial desert plant competition simulation.

    Simulation details:
    - Three species (Cactus, DesertGrass, Shrub) are initialized with specific growth parameters.
    - Each species starts with a given initial population, distributed spatially in clusters.
    - The simulation runs for a specified number of timesteps, tracking population and explicit resource usage per time.
    - Results are saved to CSV and a population/resource dynamics plot is generated.

    Parameters
    ----------
    project_root : Optional[str]
        Optional override for the project root directory.
    data_path : Optional[str]
        Optional override for the data directory.
    result_csv : Optional[str]
        Optional override for the result CSV output path.
    figure_png : Optional[str]
        Optional override for the plot image output path.

    Returns
    -------
    None
    """
    # Setup output paths
    root = project_root or os.environ.get("PROJECT_ROOT", os.getcwd())
    dpath = data_path or os.environ.get("DATA_PATH", "data")
    if not os.path.isabs(dpath):
        data_dir = os.path.join(root, dpath)
    else:
        data_dir = dpath
    try:
        os.makedirs(data_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating data directory '{data_dir}': {e}", file=sys.stderr)
        sys.exit(1)
    results_file = result_csv or os.path.join(data_dir, "results.csv")
    figure_file = figure_png or os.path.join(data_dir, "figure.png")

    species_data = [
        # name, r, K, water_need, sunlight_need
        ("Cactus", 0.11, 120, 1.0, 0.6),
        ("DesertGrass", 0.16, 90, 0.8, 1.0),
        ("Shrub", 0.09, 70, 1.2, 0.9)
    ]
    species_list = [PlantSpecies(*params) for params in species_data]
    initial_populations = [30, 25, 15]
    total_water = 100.0
    total_sunlight = 100.0
    competition_matrix = np.array([
        [1.0, 0.45, 0.35],
        [0.42, 1.0, 0.40],
        [0.30, 0.38, 1.0]
    ], dtype=float)
    timesteps = 80
    grid_size = (6, 6)

    sim = DesertCompetitionSimulation(
        species_list=species_list,
        initial_populations=initial_populations,
        total_water=total_water,
        total_sunlight=total_sunlight,
        competition_matrix=competition_matrix,
        timesteps=timesteps,
        grid_size=grid_size
    )
    result_df = sim.run()
    sim.save_results(results_file)
    sim.plot(figure_file)

# Execute main for both direct execution and sandbox wrapper invocation
main()