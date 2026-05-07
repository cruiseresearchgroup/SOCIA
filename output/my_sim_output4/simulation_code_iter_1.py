import random

class Person:
    """Represents an individual in the simulation with health status and behaviors."""
    
    def __init__(self, health_status, infection_probability, recovery_probability, position):
        self.health_status = health_status  # 'susceptible', 'infected', or 'recovered'
        self.infection_probability = infection_probability
        self.recovery_probability = recovery_probability
        self.position = position

    def move(self, grid_size, movement_range=1):
        """Randomly move the person within the grid."""
        dx = random.randint(-movement_range, movement_range)
        dy = random.randint(-movement_range, movement_range)
        new_x = (self.position[0] + dx) % grid_size[0]
        new_y = (self.position[1] + dy) % grid_size[1]
        self.position = (new_x, new_y)

    def interact(self, other, transmission_probability):
        """Interact with another person and potentially transmit infection."""
        if self.health_status == 'infected' and other.health_status == 'susceptible':
            if random.random() < transmission_probability:
                other.health_status = 'infected'

    def recover(self):
        """Determine if an infected person recovers."""
        if self.health_status == 'infected':
            if random.random() < self.recovery_probability:
                self.health_status = 'recovered'