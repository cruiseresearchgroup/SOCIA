import numpy as np
import matplotlib.pyplot as plt

# Defining Person entity
class Person:
    def __init__(self, health_status='healthy', location=None, immunity=False, infection_status=False):
        self.health_status = health_status
        self.location = location
        self.immunity = immunity
        self.infection_status = infection_status

    def move(self, grid):
        # Implementing random walk
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up' and self.location[0] < grid.shape[0] - 1:
            self.location = (self.location[0] + 1, self.location[1])
        elif direction == 'down' and self.location[0] > 0:
            self.location = (self.location[0] - 1, self.location[1])
        elif direction == 'right' and self.location[1] < grid.shape[1] - 1:
            self.location = (self.location[0], self.location[1] + 1)
        elif direction == 'left' and self.location[1] > 0:
            self.location = (self.location[0], self.location[1] - 1)

    def interact(self, other, interaction_rate):
        if self.location == other.location and self.infection_status and not other.immunity:
            if np.random.rand() < interaction_rate:
                other.infection_status = True

    def recover(self, recovery_rate):
        if self.infection_status:
            if np.random.rand() < recovery_rate:
                self.infection_status = False
                self.immunity = True

    def die(self, mortality_rate):
        if self.infection_status:
            if np.random.rand() < mortality_rate:
                self.health_status = 'dead'
                self.infection_status = False

class City:
    def __init__(self, dimensions, population_size, initial_infection_rate, recovery_rate, mortality_rate, interaction_rate):
        self.grid = np.zeros(dimensions)
        self.population = []
        self.initial_infection_rate = initial_infection_rate
        self.recovery_rate = recovery_rate
        self.mortality_rate = mortality_rate
        self.interaction_rate = interaction_rate
        for _ in range(population_size):
            loc = (np.random.randint(0, dimensions[0]), np.random.randint(0, dimensions[1]))
            p = Person(location=loc)
            if np.random.rand() < initial_infection_rate:
                p.infection_status = True
            self.population.append(p)

    def simulate_day(self):
        for person in self.population:
            if person.health_status == 'dead':
                continue
            person.move(self.grid)
            for other in self.population:
                if person != other:
                    person.interact(other, self.interaction_rate)
            person.recover(self.recovery_rate)
            person.die(self.mortality_rate)

    def run_simulation(self, days):
        for _ in range(days):
            self.simulate_day()

    def plot_population_status(self):
        statuses = [p.health_status for p in self.population]
        plt.hist(statuses)
        plt.title('Population Health Status')
        plt.xlabel('Status')
        plt.ylabel('Number of People')
        plt.show()

if __name__ == "__main__":
    city = City((50, 20), 1000, 0.05, 0.1, 0.02, 0.5)
    city.run_simulation(100)
    city.plot_population_status()