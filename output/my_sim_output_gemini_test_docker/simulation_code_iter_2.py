import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- utils.py ---

def set_random_seed(seed):
    """Sets the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        # Note: numpy seed is not needed as numpy is not used for random choices here.
        # If numpy is used for random operations later, add np.random.seed(seed)

# --- person.py ---

class Person:
    """
    Represents an individual agent in the simulation.

    Attributes:
        id (int): Unique identifier for the person.
        health_status (str): The current health status ('Susceptible', 'Infected', 'Recovered').
        days_infected (int): The number of simulation steps the person has been 'Infected'.
    """

    def __init__(self, id, initial_status, days_infected=0):
        """
        Initializes a new Person agent.

        Args:
            id (int): The unique ID for the agent.
            initial_status (str): The starting health status ('Susceptible', 'Infected', 'Recovered').
            days_infected (int): The initial number of days infected (default is 0).
        """
        if initial_status not in ['Susceptible', 'Infected', 'Recovered']:
            raise ValueError(f"Invalid initial status: {initial_status}")

        self.id = id
        self.health_status = initial_status
        self.days_infected = days_infected

    def update_status(self, mean_infectious_duration):
        """
        Updates the health status of the person based on rules.
        Handles the transition from 'Infected' to 'Recovered'.

        Args:
            mean_infectious_duration (int): The average duration a person is infectious.
        """
        if self.health_status == 'Infected':
            self.days_infected += 1
            # Simple transition: recover after mean_infectious_duration days
            if self.days_infected > mean_infectious_duration:
                self.health_status = 'Recovered'
                # Reset days_infected after recovery, not strictly necessary but good practice
                self.days_infected = 0

    def __repr__(self):
        return f"Person(id={self.id}, status='{self.health_status}', days_infected={self.days_infected})"

# --- interaction.py ---

def attempt_infection(person1, person2, infection_probability_per_contact):
    """
    Attempts to transmit the virus between two people during a contact.

    Args:
        person1 (Person): The first person involved in the contact.
        person2 (Person): The second person involved in the contact.
        infection_probability_per_contact (float): The probability of transmission
                                                   from an Infected to a Susceptible
                                                   person during a single contact.

    Returns:
        list[int]: A list of IDs of persons who became newly infected as a result
                   of this specific interaction. Returns an empty list if no
                   new infections occurred. Note: The caller is responsible for
                   actually changing the status *after* all interactions for
                   the step are processed, to ensure synchronous updates.
    """
    newly_infected_ids = []

    # Check if person1 can infect person2
    if person1.health_status == 'Infected' and person2.health_status == 'Susceptible':
        if random.random() < infection_probability_per_contact:
            newly_infected_ids.append(person2.id)

    # Check if person2 can infect person1 (symmetric interaction)
    if person2.health_status == 'Infected' and person1.health_status == 'Susceptible':
        if random.random() < infection_probability_per_contact:
            newly_infected_ids.append(person1.id)

    # Return unique IDs in case the same person was somehow added twice (unlikely here)
    return list(set(newly_infected_ids))


# --- metrics.py ---

class MetricsCollector:
    """
    Collects and stores simulation metrics at each step.
    """

    def __init__(self):
        """Initializes the metrics storage."""
        self.data = {
            'step': [],
            'susceptible_count': [],
            'infected_count': [],
            'recovered_count': [],
            'total_infections': [], # Cumulative
            'peak_infected_count': 0 # Maximum value seen so far
        }
        self._cumulative_infections = 0 # Internal counter for total infections

    def collect(self, step, population):
        """
        Collects metrics for the current simulation step.

        Args:
            step (int): The current simulation step number.
            population (list[Person]): The list of all agents in the simulation.
        """
        susceptible = sum(1 for p in population if p.health_status == 'Susceptible')
        infected = sum(1 for p in population if p.health_status == 'Infected')
        recovered = sum(1 for p in population if p.health_status == 'Recovered')

        # Calculate newly infected in this step to update cumulative count
        # This requires comparing state to the previous step, or tracking state changes.
        # A simpler approach for *cumulative* infections is to count transitions
        # from Susceptible to Infected *as they happen*.
        # However, the model plan just asks for total_infections as a metric,
        # typically meaning cumulative cases. Let's assume it means cumulative
        # count of people who have *ever* been infected.
        # A more robust way to track cumulative infections is needed.
        # Let's refine the definition: total_infections is the count of people
        # who are currently Infected OR have Recovered. This is a common proxy
        # for cumulative cases in simple SIR models without births/deaths.
        cumulative_infections = infected + recovered

        self.data['step'].append(step)
        self.data['susceptible_count'].append(susceptible)
        self.data['infected_count'].append(infected)
        self.data['recovered_count'].append(recovered)
        self.data['total_infections'].append(cumulative_infections)

        # Update peak infected count
        self.data['peak_infected_count'] = max(self.data['peak_infected_count'], infected)

        # Basic validation: check population conservation
        total_population = susceptible + infected + recovered
        if step > 0 and total_population != self.data['susceptible_count'][0] + self.data['infected_count'][0] + self.data['recovered_count'][0]:
             print(f"Warning: Population size changed at step {step}! Expected: {self.data['susceptible_count'][0] + self.data['infected_count'][0] + self.data['recovered_count'][0]}, Got: {total_population}")
        elif step == 0 and total_population != len(population):
             print(f"Warning: Initial population size mismatch! Expected: {len(population)}, Got: {total_population}")


    def get_dataframe(self):
        """Returns the collected metrics as a pandas DataFrame."""
        return pd.DataFrame(self.data)

    def get_peak_infected(self):
        """Returns the peak number of simultaneously infected individuals."""
        return self.data['peak_infected_count']

# --- simulation.py ---

class Simulation:
    """
    Coordinates the agent-based epidemic simulation.
    """

    def __init__(self, parameters):
        """
        Initializes the simulation with parameters.

        Args:
            parameters (dict): A dictionary containing simulation parameters.
        """
        self.parameters = parameters
        self.population = []
        self.metrics_collector = MetricsCollector()

        # Set random seed if provided
        set_random_seed(self.parameters.get("random_seed"))

        self.initialize_population()

    def initialize_population(self):
        """
        Creates and initializes the population of Person agents.
        """
        pop_size = self.parameters['population_size']
        initial_infected = self.parameters['initial_infected_count']

        if initial_infected > pop_size:
            raise ValueError("Initial infected count cannot exceed population size.")

        # Create all agents as Susceptible initially
        self.population = [Person(id=i, initial_status='Susceptible') for i in range(pop_size)]

        # Randomly select initial infected individuals
        infected_indices = random.sample(range(pop_size), initial_infected)
        for i in infected_indices:
            self.population[i].health_status = 'Infected'
            self.population[i].days_infected = 0 # Start days_infected at 0

        print(f"Initialized population: {pop_size} total, {initial_infected} infected.")

    def run_step(self, step):
        """
        Executes a single step of the simulation.

        Args:
            step (int): The current step number.
        """
        # --- Phase 1: Interactions ---
        # Identify agents who become infected in this step based on interactions.
        # Collect IDs first to apply changes synchronously after all interactions.
        newly_infected_ids_in_this_step = set()
        pop_size = len(self.population)
        contacts_per_person = self.parameters['contacts_per_person_per_step']
        infection_prob = self.parameters['infection_probability_per_contact']

        # To efficiently pick random partners, especially in non-spatial models,
        # we can shuffle indices or directly sample from the population list.
        # A common way for 'random mixing' is to pair randomly or pick partners randomly.
        # The plan says "randomly selecting another Person agent".
        # Let's iterate through each person and give them N contacts.
        for person in self.population:
            for _ in range(contacts_per_person):
                # Pick a random partner (can be self, model doesn't exclude it)
                partner = random.choice(self.population)
                # Attempt infection and collect IDs of those who *would* become infected
                infected_ids = attempt_infection(person, partner, infection_prob)
                newly_infected_ids_in_this_step.update(infected_ids)

        # Now, apply the state changes from interactions *synchronously*
        # Ensure we only infect those who are currently Susceptible
        for person_id in newly_infected_ids_in_this_step:
            person = self.population[person_id] # Assuming ID matches index
            if person.health_status == 'Susceptible':
                person.health_status = 'Infected'
                person.days_infected = 0 # Newly infected start at day 0

        # --- Phase 2: Status Updates ---
        # Apply time-based status updates (e.g., recovery)
        mean_infectious_duration = self.parameters['mean_infectious_duration']
        for person in self.population:
            person.update_status(mean_infectious_duration)

        # --- Phase 3: Collect Metrics ---
        self.metrics_collector.collect(step, self.population)


    def run_simulation(self):
        """
        Runs the simulation for the specified number of steps.
        """
        simulation_steps = self.parameters['simulation_steps']
        print(f"Starting simulation for {simulation_steps} steps...")

        # Collect initial metrics (step 0)
        self.metrics_collector.collect(0, self.population)

        for step in range(1, simulation_steps + 1):
            # print(f"Running step {step}...") # Uncomment for verbose output
            self.run_step(step)

        print("Simulation finished.")
        print(f"Peak infected count: {self.metrics_collector.get_peak_infected()}")


    def get_results(self):
        """
        Returns the simulation results (metrics) as a pandas DataFrame.
        """
        return self.metrics_collector.get_dataframe()

    def visualize_results(self, output_dir="results"):
        """
        Generates and saves plots of the simulation results.

        Args:
            output_dir (str): Directory to save the plots.
        """
        df = self.get_results()

        if df.empty:
            print("No data to visualize.")
            return

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.plot(df['step'], df['susceptible_count'], label='Susceptible', color='blue')
        plt.plot(df['step'], df['infected_count'], label='Infected', color='red')
        plt.plot(df['step'], df['recovered_count'], label='Recovered', color='green')
        plt.xlabel('Simulation Step (Days)')
        plt.ylabel('Number of People')
        plt.title('Epidemic Spread Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "epidemic_curve.png")
        plt.savefig(plot_path)
        print(f"Epidemic curve saved to {plot_path}")
        # plt.show() # Uncomment if you want to display the plot immediately


        plt.figure(figsize=(12, 6))
        plt.plot(df['step'], df['total_infections'], label='Cumulative Infections', color='purple')
        plt.xlabel('Simulation Step (Days)')
        plt.ylabel('Cumulative Number of People')
        plt.title('Cumulative Infections Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path_cumulative = os.path.join(output_dir, "cumulative_infections.png")
        plt.savefig(plot_path_cumulative)
        print(f"Cumulative infections plot saved to {plot_path_cumulative}")
        # plt.show() # Uncomment if you want to display the plot immediately


        # Optional: Save metrics to CSV
        csv_path = os.path.join(output_dir, "simulation_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Simulation metrics saved to {csv_path}")

# --- Main Execution ---

if __name__ == "__main__":
    # Define simulation parameters based on the Model Plan
    simulation_parameters = {
        "population_size": 1000,
        "initial_infected_count": 5,
        "infection_probability_per_contact": 0.05,
        "mean_infectious_duration": 10, # Corresponds to recovery after 10 days infected
        "contacts_per_person_per_step": 5,
        "simulation_steps": 200,
        "random_seed": 42 # Using a fixed seed for reproducibility as suggested
    }

    # Create and run the simulation
    sim = Simulation(simulation_parameters)
    sim.run_simulation()

    # Get and visualize results
    results_df = sim.get_results()
    print("\nSimulation Results (first 5 steps):")
    print(results_df.head())
    print("\nSimulation Results (last 5 steps):")
    print(results_df.tail())

    sim.visualize_results()

    # Validation Check: Population Conservation
    initial_pop = results_df['susceptible_count'][0] + results_df['infected_count'][0] + results_df['recovered_count'][0]
    final_pop = results_df['susceptible_count'].iloc[-1] + results_df['infected_count'].iloc[-1] + results_df['recovered_count'].iloc[-1]
    print(f"\nValidation Check: Population Conservation")
    print(f"Initial Population Sum: {initial_pop}")
    print(f"Final Population Sum: {final_pop}")
    if initial_pop == final_pop and initial_pop == simulation_parameters["population_size"]:
        print("Population conservation check passed.")
    else:
        print("Population conservation check FAILED.")

    # Validation Check: Plausible Epidemic Curve (qualitative check based on visualization)
    # This is hard to automate rigorously without defining "plausible".
    # The generated plot should show S decreasing, I rising then falling, R increasing.
    print("\nValidation Check: Plausible Epidemic Curve")
    print(f"Check the generated 'results/epidemic_curve.png' plot.")
    print("Expected: Susceptible decreases, Infected rises and falls (an epidemic curve), Recovered increases.")

    # Validation Check: Status Transition Logic
    # This is primarily validated by inspecting the code logic for Person.update_status
    # and attempt_infection. Automated checks would be complex (e.g., tracking individual
    # agent histories). We rely on the code implementation matching the rules.
    print("\nValidation Check: Status Transition Logic")
    print("Logic is implemented in Person.update_status and interaction.attempt_infection.")
    print("Requires code review to verify correctness.")

    # Validation Check: Parameter Sensitivity
    # Requires running the simulation with different parameters and comparing results.
    # This is not part of a single execution but a process of analysis.
    print("\nValidation Check: Parameter Sensitivity")
    print("Requires running the simulation with different parameters (e.g., infection_probability_per_contact, contacts_per_person_per_step) and observing the effect on the epidemic curve.")
    print("This check is performed by the user running multiple simulations.")