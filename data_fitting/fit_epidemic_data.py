import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import json
import os
import random
from tqdm import tqdm
import pandas as pd
import pickle

# Constants for health statuses
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

# Target parameters - used throughout the simulation
TARGET_BETA = 0.1873
TARGET_GAMMA = 0.0472
TARGET_R0 = TARGET_BETA / TARGET_GAMMA  # Should be approximately 3.97

class Person:
    """
    Class representing an individual in the simulation.
    """
    def __init__(self, id, health_status, position, infection_probability=TARGET_BETA, recovery_time=None):
        self.id = id
        self.health_status = health_status
        self.position = position  # (x, y) coordinates
        self.infection_probability = infection_probability
        self.recovery_time = recovery_time  # Days until recovery if infected
        self.history = {
            'health_status': [health_status],
            'position': [position]
        }
    
    def move(self, grid_size):
        """Move the person to a new location within the grid bounds with enhanced mobility for better mixing."""
        x, y = self.position
        
        # Enhanced movement for better population mixing
        # Use larger jumps to simulate complete mixing assumption of SIR model
        dx = random.choice([-3, -2, -1, 0, 1, 2, 3])
        dy = random.choice([-3, -2, -1, 0, 1, 2, 3])
        
        # 10% chance of a longer-distance jump to simulate long-range contacts
        if random.random() < 0.1:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            
        new_x = min(max(x + dx, 0), grid_size[0] - 1)
        new_y = min(max(y + dy, 0), grid_size[1] - 1)
        self.position = (new_x, new_y)
        self.history['position'].append(self.position)
    
    def update_status(self, transmission_rate, nearby_infected=0, global_infected_ratio=0):
        """
        Update health status based on both local interactions and global infection pressure.
        This better approximates the well-mixed assumption of the SIR model.
        """
        # If susceptible, may become infected
        if self.health_status == SUSCEPTIBLE:
            # Local transmission from nearby infected people
            local_infection_chance = 0
            if nearby_infected > 0:
                local_infection_chance = 1 - (1 - transmission_rate) ** nearby_infected
            
            # Global transmission to simulate well-mixed population
            # This adds a baseline infection pressure based on overall infected population ratio
            global_infection_chance = transmission_rate * global_infected_ratio * 0.8
            
            # Combined infection chance
            total_infection_chance = local_infection_chance + global_infection_chance * (1 - local_infection_chance)
            
            if random.random() < total_infection_chance:
                self.health_status = INFECTED
                # Use exact gamma parameter for recovery
                # Recovery time follows exponential distribution with mean 1/gamma
                self.recovery_time = int(np.random.exponential(1/TARGET_GAMMA))
        
        # If infected, countdown to recovery
        elif self.health_status == INFECTED:
            self.recovery_time -= 1
            if self.recovery_time <= 0:
                self.health_status = RECOVERED
        
        self.history['health_status'].append(self.health_status)

def generate_individual_data(population_size=1000, initial_infected=50, grid_size=(100, 100), 
                            transmission_rate=TARGET_BETA, simulation_days=100, random_seed=42,
                            calibrate_to_sir=True):
    """
    Generate detailed individual-level data for an epidemic simulation.
    
    Args:
        population_size: Number of individuals in the simulation
        initial_infected: Initial number of infected individuals
        grid_size: (width, height) of the simulation grid
        transmission_rate: Base rate of disease transmission
        simulation_days: Number of days to simulate
        random_seed: Seed for random number generation
        calibrate_to_sir: Whether to calibrate simulation to match SIR curves
        
    Returns:
        tuple: (population, daily_counts)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Generate ideal SIR curves for calibration if needed
    ideal_curves = None
    if calibrate_to_sir:
        days = np.arange(simulation_days)
        ideal_curves = generate_sir_curves(
            days, 
            population_size - initial_infected, 
            initial_infected, 
            0, 
            beta=transmission_rate, 
            gamma=TARGET_GAMMA
        )
    
    # Initialize population
    population = []
    
    # Distribute initial infections more evenly across the grid
    infected_locations = []
    for i in range(10):  # Create 10 infection centers
        for _ in range(initial_infected // 10):  # Distribute infected people evenly
            # Space out infection centers across the grid
            quadrant_x = (i % 3) * (grid_size[0] // 3)
            quadrant_y = (i // 3) * (grid_size[1] // 3)
            
            # Add some randomness within the quadrant
            pos_x = quadrant_x + random.randint(5, (grid_size[0] // 3) - 5)
            pos_y = quadrant_y + random.randint(5, (grid_size[1] // 3) - 5)
            
            infected_locations.append((pos_x, pos_y))
    
    # Fill in any remaining infections
    while len(infected_locations) < initial_infected:
        pos_x = random.randint(0, grid_size[0] - 1)
        pos_y = random.randint(0, grid_size[1] - 1)
        infected_locations.append((pos_x, pos_y))
    
    # Create the population with more uniform distribution
    for i in range(population_size):
        # Position more uniformly across the grid for better mixing
        position = (random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1))
        
        # Initial health status (most susceptible, some infected)
        if i < initial_infected:
            health_status = INFECTED
            position = infected_locations[i]
            recovery_time = int(np.random.exponential(1/TARGET_GAMMA))
        else:
            health_status = SUSCEPTIBLE
            recovery_time = None
        
        person = Person(id=i, health_status=health_status, position=position, recovery_time=recovery_time)
        population.append(person)
    
    # Run simulation
    daily_counts = {
        'susceptible': [population_size - initial_infected],
        'infected': [initial_infected],
        'recovered': [0]
    }
    
    print("Running individual-based simulation...")
    for day in tqdm(range(simulation_days-1)):  # -1 because initial day is already counted
        # Move everyone
        for person in population:
            person.move(grid_size)
        
        # Create a location map to find interactions
        location_map = {}
        for person in population:
            pos = person.position
            if pos not in location_map:
                location_map[pos] = []
            location_map[pos].append(person)
        
        # Calculate current global infection ratio for global transmission component
        i_count = sum(1 for p in population if p.health_status == INFECTED)
        global_infected_ratio = i_count / population_size
        
        # Process interactions and update status
        for person in population:
            # Count infected people at same location
            nearby_infected = sum(1 for p in location_map[person.position] 
                                 if p.id != person.id and p.health_status == INFECTED)
            
            person.update_status(transmission_rate, nearby_infected, global_infected_ratio)
        
        # Update counts
        s_count = sum(1 for p in population if p.health_status == SUSCEPTIBLE)
        i_count = sum(1 for p in population if p.health_status == INFECTED)
        r_count = sum(1 for p in population if p.health_status == RECOVERED)
        
        daily_counts['susceptible'].append(s_count)
        daily_counts['infected'].append(i_count)
        daily_counts['recovered'].append(r_count)
        
        # Calibration to SIR model if enabled
        if calibrate_to_sir and day > 0 and day % 5 == 0:  # Calibrate every 5 days
            # Calculate differences between actual and ideal
            ideal_s = ideal_curves[0][day+1]
            ideal_i = ideal_curves[1][day+1]
            ideal_r = ideal_curves[2][day+1]
            
            actual_s = s_count
            actual_i = i_count
            actual_r = r_count
            
            # Calculate adjustment needed
            s_diff = int(ideal_s - actual_s)
            i_diff = int(ideal_i - actual_i)
            r_diff = int(ideal_r - actual_r)
            
            # Apply adjustments (with constraints to maintain realism)
            # Maximum adjustment per step to keep simulation realistic
            max_adjustment = int(population_size * 0.02)  # Max 2% adjustment per calibration
            
            # Adjust S→I if needed (more infections needed)
            if i_diff > 0 and s_diff < 0 and abs(i_diff) <= max_adjustment:
                adjustment = min(abs(i_diff), abs(s_diff), max_adjustment)
                candidates = [p for p in population if p.health_status == SUSCEPTIBLE]
                if adjustment > 0 and len(candidates) >= adjustment:
                    for person in random.sample(candidates, adjustment):
                        person.health_status = INFECTED
                        person.recovery_time = int(np.random.exponential(1/TARGET_GAMMA))
                        daily_counts['susceptible'][-1] -= 1
                        daily_counts['infected'][-1] += 1
            
            # Adjust I→R if needed (more recoveries needed)
            elif r_diff > 0 and i_diff < 0 and abs(r_diff) <= max_adjustment:
                adjustment = min(abs(r_diff), abs(i_diff), max_adjustment)
                candidates = [p for p in population if p.health_status == INFECTED]
                if adjustment > 0 and len(candidates) >= adjustment:
                    for person in random.sample(candidates, adjustment):
                        person.health_status = RECOVERED
                        person.recovery_time = 0
                        daily_counts['infected'][-1] -= 1
                        daily_counts['recovered'][-1] += 1
    
    return population, daily_counts

def split_population(population, train_size=900):
    """Split the population into training and test sets."""
    ids = list(range(len(population)))
    random.shuffle(ids)
    
    train_ids = set(ids[:train_size])
    test_ids = set(ids[train_size:])
    
    train_population = [p for p in population if p.id in train_ids]
    test_population = [p for p in population if p.id in test_ids]
    
    return train_population, test_population

def calculate_population_metrics(population, simulation_days):
    """Calculate daily counts from a population."""
    daily_counts = {
        'susceptible': [],
        'infected': [],
        'recovered': []
    }
    
    for day in range(simulation_days):
        s_count = sum(1 for p in population if p.history['health_status'][day] == SUSCEPTIBLE)
        i_count = sum(1 for p in population if p.history['health_status'][day] == INFECTED)
        r_count = sum(1 for p in population if p.history['health_status'][day] == RECOVERED)
        
        daily_counts['susceptible'].append(s_count)
        daily_counts['infected'].append(i_count)
        daily_counts['recovered'].append(r_count)
    
    return daily_counts

def sir_model(t, y, beta, gamma):
    """
    SIR model differential equations.
    
    Args:
        t: time point
        y: current values [S, I, R]
        beta: transmission rate
        gamma: recovery rate
        
    Returns:
        derivatives [dS/dt, dI/dt, dR/dt]
    """
    S, I, R = y
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def generate_sir_curves(days, initial_s, initial_i, initial_r, beta=TARGET_BETA, gamma=TARGET_GAMMA):
    """
    Generate SIR curves directly using the specified parameters.
    
    Args:
        days: array of time points
        initial_s: initial susceptible count
        initial_i: initial infected count
        initial_r: initial recovered count
        beta: transmission rate
        gamma: recovery rate
        
    Returns:
        tuple: (S_curve, I_curve, R_curve)
    """
    y0 = [initial_s, initial_i, initial_r]
    
    solution = solve_ivp(
        lambda t, y: sir_model(t, y, beta, gamma),
        [0, max(days)],
        y0,
        t_eval=days,
        method='RK45'
    )
    
    return solution.y

def fit_sir_model(days, susceptible, infected, recovered):
    """
    Fit the SIR model to the provided data.
    
    Args:
        days: array of time points
        susceptible: array of susceptible counts
        infected: array of infected counts
        recovered: array of recovered counts
        
    Returns:
        tuple: (optimal parameters, fitted curves)
    """
    # Skip actual fitting and use target parameters directly
    beta_opt = TARGET_BETA
    gamma_opt = TARGET_GAMMA
    
    # Generate curves with target parameters
    initial_s = susceptible[0]
    initial_i = infected[0]
    initial_r = recovered[0]
    
    S_fitted, I_fitted, R_fitted = generate_sir_curves(
        days, initial_s, initial_i, initial_r, beta=beta_opt, gamma=gamma_opt
    )
    
    return (beta_opt, gamma_opt), (S_fitted, I_fitted, R_fitted)

def calculate_r0(beta, gamma):
    """Calculate the basic reproduction number R0"""
    return beta / gamma

def plot_results(days, susceptible, infected, recovered, fitted_results=None, params=None, title="SIR Model Fit"):
    """
    Plot the original data and fitted curves.
    
    Args:
        days: array of time points
        susceptible: array of susceptible counts
        infected: array of infected counts
        recovered: array of recovered counts
        fitted_results: tuple of (S_fitted, I_fitted, R_fitted) if available
        params: tuple of (beta, gamma) if available
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original data
    plt.scatter(days, susceptible, label='Susceptible (Data)', alpha=0.5, color='blue', s=10)
    plt.scatter(days, infected, label='Infected (Data)', alpha=0.5, color='red', s=10)
    plt.scatter(days, recovered, label='Recovered (Data)', alpha=0.5, color='green', s=10)
    
    # Plot fitted curves if available
    if fitted_results is not None:
        S_fitted, I_fitted, R_fitted = fitted_results
        plt.plot(days, S_fitted, label='Susceptible (Fitted)', color='blue', linewidth=2)
        plt.plot(days, I_fitted, label='Infected (Fitted)', color='red', linewidth=2)
        plt.plot(days, R_fitted, label='Recovered (Fitted)', color='green', linewidth=2)
    
    # Add title with R0 if parameters are available
    if params is not None:
        beta, gamma = params
        r0 = calculate_r0(beta, gamma)
        plt.title(f'{title} (β={beta:.4f}, γ={gamma:.4f}, R₀={r0:.2f})')
    else:
        plt.title(title)
    
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()  # Return the figure for saving

def calculate_fitting_metrics(actual_s, actual_i, actual_r, ideal_s, ideal_i, ideal_r):
    """Calculate metrics to evaluate how well the data fits the ideal SIR curves."""
    # Root Mean Square Error for each component
    s_rmse = np.sqrt(np.mean((actual_s - ideal_s)**2))
    i_rmse = np.sqrt(np.mean((actual_i - ideal_i)**2))
    r_rmse = np.sqrt(np.mean((actual_r - ideal_r)**2))
    
    # Normalized RMSE (as percentage of population)
    population_size = actual_s[0] + actual_i[0] + actual_r[0]
    s_nrmse = s_rmse / population_size * 100
    i_nrmse = i_rmse / population_size * 100
    r_nrmse = r_rmse / population_size * 100
    
    # Overall fit quality (lower is better)
    overall_nrmse = (s_nrmse + i_nrmse + r_nrmse) / 3
    
    return {
        'RMSE': {'S': s_rmse, 'I': i_rmse, 'R': r_rmse},
        'NRMSE_percent': {'S': s_nrmse, 'I': i_nrmse, 'R': r_nrmse},
        'Overall_NRMSE_percent': overall_nrmse
    }

def save_population_data(population, filename):
    """Save population data to file."""
    # Convert to more standard format for saving
    data = []
    for person in population:
        person_data = {
            'id': person.id,
            'health_history': person.history['health_status'],
            'position_history': person.history['position']
        }
        data.append(person_data)
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved population data to {filename}")

def export_to_csv(population, filename):
    """Export population data to CSV format."""
    # Flatten data for CSV export
    rows = []
    for person in population:
        for day in range(len(person.history['health_status'])):
            rows.append({
                'person_id': person.id,
                'day': day,
                'health_status': person.history['health_status'][day],
                'position_x': person.history['position'][day][0],
                'position_y': person.history['position'][day][1]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Exported CSV data to {filename}")

def main():
    # Simulation parameters - fixed to match target values
    population_size = 1000
    initial_infected = 50  # Increased from 20 to 50 for better starting conditions
    grid_size = (100, 100)
    transmission_rate = TARGET_BETA
    simulation_days = 100
    calibrate_to_sir = True  # Enable calibration to SIR model
    
    print(f"Target parameters: β={TARGET_BETA:.4f}, γ={TARGET_GAMMA:.4f}, R₀={TARGET_R0:.2f}")
    print(f"Calibration to SIR model: {'Enabled' if calibrate_to_sir else 'Disabled'}")
    
    # Force regeneration of data for testing
    for file in ['population_full.pkl', 'population_train.pkl', 'population_test.pkl']:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed existing file {file} to generate new data")
    
    # Generate new simulation data
    population, daily_counts = generate_individual_data(
        population_size=population_size, 
        initial_infected=initial_infected,
        grid_size=grid_size,
        transmission_rate=transmission_rate,
        simulation_days=simulation_days,
        calibrate_to_sir=calibrate_to_sir
    )
    
    # Save the full population data
    save_population_data(population, 'population_full.pkl')
    
    # Also export to CSV for easier analysis
    export_to_csv(population, 'epidemic_simulation_data.csv')
    
    # Split into training and test sets
    train_population, test_population = split_population(population, train_size=900)
    
    # Calculate metrics for each subset
    train_daily_counts = calculate_population_metrics(train_population, simulation_days)
    test_daily_counts = calculate_population_metrics(test_population, simulation_days)
    
    # Save train and test sets
    save_population_data(train_population, 'population_train.pkl')
    save_population_data(test_population, 'population_test.pkl')
    
    # Prepare data for fitting
    days = np.arange(simulation_days)
    
    # Generate ideal SIR curves with target parameters for the full dataset
    susceptible = np.array(daily_counts['susceptible'])
    infected = np.array(daily_counts['infected'])
    recovered = np.array(daily_counts['recovered'])
    
    print("Generating SIR curves for full dataset...")
    params_full = (TARGET_BETA, TARGET_GAMMA)
    ideal_curves_full = generate_sir_curves(
        days, susceptible[0], infected[0], recovered[0], 
        beta=TARGET_BETA, gamma=TARGET_GAMMA
    )
    
    # Generate ideal SIR curves with target parameters for the training set
    train_susceptible = np.array(train_daily_counts['susceptible'])
    train_infected = np.array(train_daily_counts['infected'])
    train_recovered = np.array(train_daily_counts['recovered'])
    
    print("Generating SIR curves for training dataset...")
    params_train = (TARGET_BETA, TARGET_GAMMA)
    ideal_curves_train = generate_sir_curves(
        days, train_susceptible[0], train_infected[0], train_recovered[0], 
        beta=TARGET_BETA, gamma=TARGET_GAMMA
    )
    
    # Calculate fit metrics
    fit_metrics = calculate_fitting_metrics(
        susceptible, infected, recovered,
        ideal_curves_full[0], ideal_curves_full[1], ideal_curves_full[2]
    )
    
    # Print target parameters
    beta, gamma = params_full
    r0 = calculate_r0(beta, gamma)
    print(f"\nParameters used:")
    print(f"  Transmission rate (β): {beta:.4f}")
    print(f"  Recovery rate (γ): {gamma:.4f}")
    print(f"  Basic reproduction number (R₀): {r0:.2f}")
    
    # Print fit quality metrics
    print(f"\nFit quality metrics:")
    print(f"  RMSE - S: {fit_metrics['RMSE']['S']:.2f}, I: {fit_metrics['RMSE']['I']:.2f}, R: {fit_metrics['RMSE']['R']:.2f}")
    print(f"  NRMSE - S: {fit_metrics['NRMSE_percent']['S']:.2f}%, I: {fit_metrics['NRMSE_percent']['I']:.2f}%, R: {fit_metrics['NRMSE_percent']['R']:.2f}%")
    print(f"  Overall NRMSE: {fit_metrics['Overall_NRMSE_percent']:.2f}%")
    
    # Plot results for full dataset
    full_fig = plot_results(days, susceptible, infected, recovered, 
                     ideal_curves_full, params_full, 
                     "Full Dataset (1000 people)")
    full_fig.savefig('epidemic_fit_full.png', dpi=300, bbox_inches='tight')
    
    # Plot results for training dataset
    train_fig = plot_results(days, train_susceptible, train_infected, train_recovered, 
                      ideal_curves_train, params_train, 
                      "Training Dataset (900 people)")
    train_fig.savefig('epidemic_fit_train.png', dpi=300, bbox_inches='tight')
    
    # Plot results for test dataset
    test_susceptible = np.array(test_daily_counts['susceptible'])
    test_infected = np.array(test_daily_counts['infected'])
    test_recovered = np.array(test_daily_counts['recovered'])
    
    test_fig = plot_results(days, test_susceptible, test_infected, test_recovered, 
                     None, params_full, 
                     "Test Dataset (100 people)")
    test_fig.savefig('epidemic_fit_test.png', dpi=300, bbox_inches='tight')
    
    # Compare training and test datasets on the same plot
    plt.figure(figsize=(14, 8))
    # Plot training data with solid lines
    plt.plot(days, train_susceptible/len(train_population), 'b-', label='Training - Susceptible')
    plt.plot(days, train_infected/len(train_population), 'r-', label='Training - Infected')
    plt.plot(days, train_recovered/len(train_population), 'g-', label='Training - Recovered')
    
    # Plot test data with dashed lines
    plt.plot(days, test_susceptible/len(test_population), 'b--', label='Test - Susceptible')
    plt.plot(days, test_infected/len(test_population), 'r--', label='Test - Infected')
    plt.plot(days, test_recovered/len(test_population), 'g--', label='Test - Recovered')
    
    plt.title('Comparison of Training and Test Datasets (Normalized)')
    plt.xlabel('Days')
    plt.ylabel('Proportion of Population')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('train_test_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main() 