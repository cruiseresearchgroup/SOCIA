import random
import matplotlib.pyplot as plt
import pandas as pd
import math

class Person:
    """Represents a single person (agent) in the simulation."""

    def __init__(self, health_status: str, days_infected: int = 0, ever_infected: bool = False):
        """
        Initializes a Person agent.

        Args:
            health_status (str): The initial health status ('Susceptible', 'Infected', 'Recovered').
            days_infected (int): The number of days the person has been infected (0 if not Infected).
            ever_infected (bool): True if the person has ever been infected, False otherwise.
        """
        valid_statuses = ['Susceptible', 'Infected', 'Recovered']
        if health_status not in valid_statuses:
            raise ValueError(f"Invalid initial health_status: {health_status}. Must be one of {valid_statuses}")

        self.health_status = health_status
        self.days_infected = days_infected
        self.ever_infected = ever_infected # Tracks cumulative cases

    def __repr__(self):
        return f"Person(status='{self.health_status}', days_infected={self.days_infected}, ever_infected={self.ever_infected})"

    def attempt_to_infect_others(self, population: list, transmission_probability: float, average_contacts_per_day: int) -> set:
        """
        If infected, attempts to transmit the virus to a random sample of the population.

        Args:
            population (list[Person]): The list of all agents in the simulation.
            transmission_probability (float): The probability of transmission per contact [0, 1].
            average_contacts_per_day (int): The average number of random contacts per day.

        Returns:
            set[Person]: A set of susceptible agents who were potentially infected during this agent's interactions.
        """
        newly_infected_candidates = set()
        if self.health_status == 'Infected':
            # Ensure we don't try to contact self, and handle populations smaller than contacts
            num_contacts = min(average_contacts_per_day, len(population) - 1)
            if num_contacts <= 0:
                 return newly_infected_candidates # Cannot contact anyone

            # Select random contacts from the population, excluding self
            # Using random.sample is efficient for selecting unique contacts
            try:
                contacts = random.sample([p for p in population if p is not self], num_contacts)
            except ValueError:
                 # This can happen if population size is 1 and num_contacts is 1,
                 # or other edge cases where sample size > population size.
                 # Given num_contacts is min(avg_contacts, pop_size-1), this should
                 # only happen if pop_size is 1 and avg_contacts > 0, or pop_size <= avg_contacts
                 # and trying to sample pop_size-1 from less than pop_size-1.
                 # The min(..., len(population)-1) should prevent this if len(population) > 1.
                 # If len(population) == 1, num_contacts is min(avg_contacts, 0) which is 0.
                 # So this except block might be overly cautious but doesn't hurt.
                 return newly_infected_candidates


            for contact in contacts:
                if contact.health_status == 'Susceptible':
                    if random.random() < transmission_probability:
                        newly_infected_candidates.add(contact)

        return newly_infected_candidates

    def recover_over_time(self, recovery_duration: int):
        """
        Increments days infected and changes status to Recovered if duration is met.

        Args:
            recovery_duration (int): The number of days until recovery.
        """
        if self.health_status == 'Infected':
            self.days_infected += 1
            if self.days_infected >= recovery_duration:
                self.health_status = 'Recovered'
                self.days_infected = 0 # Reset days_infected after recovery


class Simulation:
    """Manages the simulation state, agents, parameters, and steps."""

    def __init__(self, parameters: dict):
        """
        Initializes the simulation.

        Args:
            parameters (dict): A dictionary of simulation parameters.
        """
        self.parameters = parameters
        self.population: list[Person] = []
        self.metrics: dict = {
            'day': [],
            'susceptible_count': [],
            'infected_count': [],
            'recovered_count': [],
            'total_cases': [] # Cumulative count of ever infected
        }
        self._initialize_population()
        self._record_metrics(0) # Record initial state

    def _initialize_population(self):
        """Creates and initializes the population of agents based on parameters."""
        pop_size = self.parameters.get('population_size', 1000)
        initial_infected = self.parameters.get('initial_infected_count', 1)
        random_seed = self.parameters.get('random_seed', None)

        if random_seed is not None:
            random.seed(random_seed)

        if not (0 <= initial_infected <= pop_size):
            raise ValueError(f"initial_infected_count ({initial_infected}) must be between 0 and population_size ({pop_size})")

        # Create all agents as Susceptible initially
        self.population = [Person(health_status='Susceptible') for _ in range(pop_size)]

        # Randomly select and infect initial individuals
        if initial_infected > 0:
            infected_indices = random.sample(range(pop_size), initial_infected)
            for i in infected_indices:
                self.population[i].health_status = 'Infected'
                self.population[i].days_infected = 0 # Start day count at 0
                self.population[i].ever_infected = True # Mark as ever infected

        print(f"Population initialized: {pop_size} individuals, {initial_infected} initially infected.")


    def _record_metrics(self, day: int):
        """Records the current state of the simulation metrics."""
        susceptible = sum(1 for p in self.population if p.health_status == 'Susceptible')
        infected = sum(1 for p in self.population if p.health_status == 'Infected')
        recovered = sum(1 for p in self.population if p.health_status == 'Recovered')
        total_cases = sum(1 for p in self.population if p.ever_infected) # Cumulative count

        self.metrics['day'].append(day)
        self.metrics['susceptible_count'].append(susceptible)
        self.metrics['infected_count'].append(infected)
        self.metrics['recovered_count'].append(recovered)
        self.metrics['total_cases'].append(total_cases)

        # Basic Validation Check: Population stability
        if susceptible + infected + recovered != len(self.population):
             print(f"Warning: Population size mismatch on day {day}! Total agents: {len(self.population)}, Sum of states: {susceptible + infected + recovered}")

    def simulation_step(self, day: int):
        """Executes one time step (one day) of the simulation."""
        transmission_prob = self.parameters.get('transmission_probability', 0.1)
        recovery_dur = self.parameters.get('recovery_duration', 10)
        avg_contacts = self.parameters.get('average_contacts_per_day', 5)

        # 1. Identify agents currently Infected at the START of the day
        # This snapshot is needed because recovery happens *before* infection attempts
        # in this implementation logic, and only agents *currently* infected attempt spread.
        infected_at_start_of_step = [p for p in self.population if p.health_status == 'Infected']

        # 2. Process Recoveries for all agents
        # Iterate through all agents and handle recovery for those currently infected
        for person in self.population:
             person.recover_over_time(recovery_dur)

        # 3. Process Infections
        # Collect potential new infections from agents who were infected at the start
        newly_infected_candidates = set()
        # Only agents who are *still* Infected after the recovery step attempt to infect
        currently_infected_after_recovery = [p for p in self.population if p.health_status == 'Infected']

        for person in currently_infected_after_recovery:
             candidates = person.attempt_to_infect_others(self.population, transmission_prob, avg_contacts)
             newly_infected_candidates.update(candidates)

        # 4. Apply New Infections
        # Update status for agents who were marked as candidates and are still susceptible
        # Use a list comprehension to avoid modifying the set while iterating if that were the case,
        # though modifying the set is okay here since we only read from it.
        # Check status again here because a person might have been added to the set multiple times
        # but should only transition once.
        for person in newly_infected_candidates:
            if person.health_status == 'Susceptible':
                person.health_status = 'Infected'
                person.days_infected = 0
                person.ever_infected = True # Mark as ever infected

        # 5. Record Metrics at the end of the day
        self._record_metrics(day + 1)

    def run(self):
        """Runs the simulation for the specified duration."""
        duration = self.parameters.get('simulation_duration_days', 100)
        if not isinstance(duration, int) or duration < 0:
             raise ValueError(f"simulation_duration_days must be a non-negative integer, got {duration}")

        print(f"Running simulation for {duration} days...")
        for day in range(duration):
            # print(f"Simulating Day {day + 1}...") # Optional: Print progress
            self.simulation_step(day)
        print("Simulation finished.")

    def get_metrics(self) -> pd.DataFrame:
        """Returns the collected simulation metrics as a pandas DataFrame."""
        return pd.DataFrame(self.metrics)


def plot_simulation_results(metrics_df: pd.DataFrame):
    """
    Plots the simulation results (counts of statuses) over time.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing simulation metrics.
    """
    if metrics_df.empty:
        print("No metrics data to plot.")
        return

    plt.figure(figsize=(12, 8))
    plt.plot(metrics_df['day'], metrics_df['susceptible_count'], label='Susceptible', color='blue')
    plt.plot(metrics_df['day'], metrics_df['infected_count'], label='Infected', color='red')
    plt.plot(metrics_df['day'], metrics_df['recovered_count'], label='Recovered', color='green')
    plt.plot(metrics_df['day'], metrics_df['total_cases'], label='Total Cases (Cumulative)', color='purple', linestyle='--')

    plt.xlabel('Day')
    plt.ylabel('Number of People')
    plt.title('Simple Epidemic Spread Simulation (SIR-like)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Define simulation parameters
    simulation_parameters = {
        "population_size": 1000,
        "initial_infected_count": 5,
        "transmission_probability": 0.15, # Increased slightly for a clearer curve
        "recovery_duration": 10, # Days until recovery
        "average_contacts_per_day": 5,
        "simulation_duration_days": 150, # Run longer to see full recovery
        "random_seed": 42 # Use a fixed seed for reproducibility
    }

    # Create and run the simulation
    try:
        sim = Simulation(simulation_parameters)
        sim.run()

        # Get and plot results
        results_df = sim.get_metrics()
        print("\nSimulation Metrics (last 5 days):")
        print(results_df.tail())

        # Basic validation checks from criteria
        print("\nValidation Checks:")
        # Population Stability
        initial_pop = simulation_parameters.get('population_size')
        final_pop_sum = results_df[['susceptible_count', 'infected_count', 'recovered_count']].iloc[-1].sum()
        print(f"- Population Stability: Initial={initial_pop}, Final Sum={final_pop_sum}. {'PASS' if initial_pop == final_pop_sum else 'FAIL'}")

        # Status Transitions (Manual check via logic: S->I, I->R are the only transitions coded)
        print("- Status Transitions: Logic implements S->I and I->R transitions only. (Requires code review)")

        # Typical Curve Shape (Visual check)
        print("- Typical Curve Shape: Check plot for characteristic Infected curve shape (peak, decline). (Requires visual inspection)")


        plot_simulation_results(results_df)

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except ImportError:
        print("Required libraries (pandas, matplotlib) not found. Please install them:")
        print("pip install pandas matplotlib")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")