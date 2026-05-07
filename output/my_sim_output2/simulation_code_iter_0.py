# Import necessary libraries
import numpy as np
import random
import matplotlib.pyplot as plt

# agent.py
class Person:
    """
    Represents a person in the simulation with attributes and behaviors.
    """
    def __init__(self, health_status, infection_probability, contact_rate):
        self.health_status = health_status  # "healthy" or "infected"
        self.infection_probability = infection_probability
        self.contact_rate = contact_rate
        self.position = (random.randint(0, 49), random.randint(0, 49))

    def move(self):
        """
        Simulates the movement of a person within the environment.
        Person moves to a random adjacent cell in the grid.
        """
        x, y = self.position
        new_x = (x + random.choice([-1, 0, 1])) % 50
        new_y = (y + random.choice([-1, 0, 1])) % 50
        self.position = (new_x, new_y)

    def interact(self, other):
        """
        Simulates interaction with another person, potentially leading to virus transmission.
        """
        if self.health_status == "infected" and other.health_status == "healthy":
            if random.random() < self.infection_probability:
                other.health_status = "infected"

# environment.py
class Environment:
    """
    Manages the simulation environment, specifically the grid where persons move.
    """
    def __init__(self, dimensions, population_size, initial_infected, infection_probability, contact_rate):
        self.dimensions = dimensions
        self.grid = [[[] for _ in range(dimensions[1])] for _ in range(dimensions[0])]
        self.persons = self.initialize_population(population_size, initial_infected, infection_probability, contact_rate)

    def initialize_population(self, population_size, initial_infected, infection_probability, contact_rate):
        """
        Initializes the population with a mix of healthy and infected persons.
        """
        persons = []
        for _ in range(initial_infected):
            persons.append(Person("infected", infection_probability, contact_rate))
        for _ in range(population_size - initial_infected):
            persons.append(Person("healthy", infection_probability, contact_rate))
        return persons

    def update_grid(self):
        """
        Updates the grid with current positions of all persons.
        """
        self.grid = [[[] for _ in range(self.dimensions[1])] for _ in range(self.dimensions[0])]
        for person in self.persons:
            x, y = person.position
            self.grid[x][y].append(person)

    def simulate_day(self):
        """
        Simulates a single day in the environment.
        """
        for person in self.persons:
            person.move()
        self.update_grid()
        self.handle_interactions()

    def handle_interactions(self):
        """
        Handles interactions between persons who share the same cell.
        """
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                if len(self.grid[x][y]) > 1:
                    for person in self.grid[x][y]:
                        for other in self.grid[x][y]:
                            if person != other:
                                person.interact(other)

# metrics.py
class Metrics:
    """
    Calculates and stores metrics such as total_infected and peak_infection_day.
    """
    def __init__(self):
        self.total_infected = []
        self.peak_infection_day = 0

    def record_daily_metrics(self, environment):
        """
        Records daily metrics.
        """
        infected_count = sum(1 for p in environment.persons if p.health_status == "infected")
        self.total_infected.append(infected_count)

    def calculate_peak_day(self):
        """
        Calculates the peak day of infection.
        """
        self.peak_infection_day = np.argmax(self.total_infected)

    def final_infected_percentage(self, total_population):
        """
        Returns the percentage of the population that was infected by the end of the simulation.
        """
        return (self.total_infected[-1] / total_population) * 100

# simulation.py
class Simulation:
    """
    Controls the simulation loop and tracks metrics.
    """
    def __init__(self, environment, metrics, days):
        self.environment = environment
        self.metrics = metrics
        self.days = days

    def run(self):
        """
        Runs the simulation for a specified number of days.
        """
        for day in range(self.days):
            self.environment.simulate_day()
            self.metrics.record_daily_metrics(self.environment)
        self.metrics.calculate_peak_day()

    def display_results(self):
        """
        Displays the results of the simulation using matplotlib.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics.total_infected, label='Total Infected')
        plt.axvline(x=self.metrics.peak_infection_day, color='r', linestyle='--', label='Peak Infection Day')
        plt.title('Epidemic Spread Simulation')
        plt.xlabel('Days')
        plt.ylabel('Number of Infected Individuals')
        plt.legend()
        plt.show()

# Main function to run the simulation
def main():
    random.seed(42)  # Set random seed for reproducibility
    np.random.seed(42)

    # Initialize environment, metrics, and simulation
    environment = Environment(dimensions=(50, 50), population_size=1000, initial_infected=10,
                              infection_probability=0.1, contact_rate=5)
    metrics = Metrics()
    simulation = Simulation(environment, metrics, days=100)

    # Run the simulation
    simulation.run()

    # Display results
    simulation.display_results()
    print(f"Final Infected Percentage: {metrics.final_infected_percentage(1000):.2f}%")

if __name__ == "__main__":
    main()