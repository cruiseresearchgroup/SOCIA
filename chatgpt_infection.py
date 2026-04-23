import matplotlib.pyplot as plt
import numpy as np

# Parameters
population = 1000
initial_exposed = 0
initial_infected = 1
initial_recovered = 0
susceptible = population - initial_exposed - initial_infected - initial_recovered

beta = 0.3       # Transmission rate
sigma = 0.2      # Rate of progression from exposed to infected (1/incubation period)
gamma = 0.1      # Recovery rate
days = 100       # Simulation days

# Initialize arrays
S = np.zeros(days)
E = np.zeros(days)
I = np.zeros(days)
R = np.zeros(days)

# Initial conditions
S[0] = susceptible
E[0] = initial_exposed
I[0] = initial_infected
R[0] = initial_recovered

# Simulation loop
for day in range(1, days):
    new_exposed = beta * S[day-1] * I[day-1] / population
    new_infected = sigma * E[day-1]
    new_recovered = gamma * I[day-1]

    S[day] = S[day-1] - new_exposed
    E[day] = E[day-1] + new_exposed - new_infected
    I[day] = I[day-1] + new_infected - new_recovered
    R[day] = R[day-1] + new_recovered

# Plot the results
plt.figure(figsize=(12,8))
plt.plot(S, label="Susceptible")
plt.plot(E, label="Exposed")
plt.plot(I, label="Infected")
plt.plot(R, label="Recovered")

plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SEIR Epidemic Simulation Model')
plt.legend()
plt.grid()
plt.show()
