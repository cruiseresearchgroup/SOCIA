import random
import matplotlib.pyplot as plt
from collections import defaultdict

class Person:
    """Represents an individual in the simulation."""
    
    def __init__(self, infection_probability, recovery_time, contact_rate):
        self.health_status = 'susceptible'
        self.infection_probability = infection_probability
        self.recovery_time = recovery_time
        self.contact_rate = contact_rate
        self.days_infected = 0

    def move(self):
        """Simulate the movement of a person."""
        # Movement logic can be expanded with spatial models
        pass

    def interact(self, others):
        """Simulate interactions with other people."""
        for _ in range(self.contact_rate):
            other = random.choice(others)
            if self.health_status == 'infected' and other.health_status == 'susceptible':
                self.infect_others(other)

    def infect_others(self, other):
        """Attempt to infect another person."""
        if random.random() < self.infection_probability:
            other.health_status = 'infected'

    def recover(self):
        """Simulate the recovery process."""
        if self.health_status == 'infected':
            self.days_infected += 1
            if self.days_infected >= self.recovery_time:
                self.health_status = 'recovered'


class EpidemicSimulation:
    """Coordinates the simulation of virus spread in a population."""
    
    def __init__(self, population_size, initial_infected, infection_probability, recovery_time, contact_rate):
        self.population = [Person(infection_probability, recovery_time, contact_rate) for _ in range(population_size)]
        for person in random.sample(self.population, initial_infected):
            person.health_status = 'infected'
        self.day = 0
        self.infection_history = []
        self.peak_infection_day = 0
        self.total_infected = initial_infected

    def simulate_day(self):
        """Simulate a single day of the epidemic."""
        self.day += 1
        current_infected = 0
        for person in self.population:
            if person.health_status == 'infected':
                current_infected += 1
                person.recover()
            person.interact(self.population)
        self.infection_history.append(current_infected)
        if current_infected > max(self.infection_history[:-1], default=0):
            self.peak_infection_day = self.day

    def simulate(self, days):
        """Run the simulation for a given number of days."""
        for _ in range(days):
            self.simulate_day()
            print(f"Day {self.day}: {self.infection_history[-1]} currently infected.")

    def collect_metrics(self):
        """Collect metrics from the simulation."""
        recovered_count = sum(1 for person in self.population if person.health_status == 'recovered')
        total_infected = sum(1 for person in self.population if person.health_status != 'susceptible')
        return {
            'total_infected': total_infected,
            'peak_infection_day': self.peak_infection_day,
            'recovered_count': recovered_count
        }

    def visualize_results(self):
        """Visualize the simulation results."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.infection_history, label='Infected')
        plt.axvline(x=self.peak_infection_day, color='r', linestyle='--', label='Peak Infection Day')
        plt.xlabel('Days')
        plt.ylabel('Number of Infected Individuals')
        plt.title('Epidemic Simulation Over Time')
        plt.legend()
        plt.show()


def main():
    """Runs the epidemic simulation."""
    # Parameters for the simulation
    population_size = 1000
    initial_infected = 10
    infection_probability = 0.1
    recovery_time = 14
    contact_rate = 5

    # Initialize and run the simulation
    simulation = EpidemicSimulation(population_size, initial_infected, infection_probability, recovery_time, contact_rate)
    simulation.simulate(50)  # Simulate for 50 days

    # Collect and print metrics
    metrics = simulation.collect_metrics()
    print(f"Total Infected: {metrics['total_infected']}")
    print(f"Peak Infection Day: {metrics['peak_infection_day']}")
    print(f"Recovered Count: {metrics['recovered_count']}")

    # Visualize the results
    simulation.visualize_results()


if __name__ == "__main__":
    main()