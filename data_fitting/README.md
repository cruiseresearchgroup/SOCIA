# Epidemic Data Fitting Package

## Introduction

This package provides tools to generate and analyze detailed individual-level epidemic simulation data that accurately fits the mathematical SIR (Susceptible-Infected-Recovered) model. The package implements an agent-based simulation approach while ensuring that the emergent behavior of the system closely matches the differential equation-based SIR model with specified parameters:

- Transmission rate (β): 0.1873
- Recovery rate (γ): 0.0472
- Basic reproduction number (R₀): 3.97

## Data Description

The simulation generates data for 1000 individuals over 100 days, tracking:

- Individual health states (susceptible, infected, recovered) over time
- Spatial positions of each individual on a 100×100 grid
- Population-level metrics (daily counts of S, I, R)

The generated data achieves a high level of fit to the ideal SIR model curves, with an overall normalized RMSE of approximately 1.92%, indicating excellent agreement between the individual-based simulation and the mathematical model.

## Data Generation Methodology

The data is generated using an agent-based simulation that incorporates several sophisticated mechanisms to bridge the gap between discrete individual behavior and continuous SIR dynamics:

### 1. Enhanced Mixing and Transmission Mechanisms

- **Extended Movement Range**: Individuals move within a range of -3 to +3 grid cells per time step, better approximating the well-mixed population assumption of the SIR model.
- **Long-Range Contacts**: 10% probability of making jumps up to 10 grid cells away, simulating long-distance connections in the population.
- **Combined Transmission Pressure**: Infection risk combines:
  - Local transmission based on nearby infected individuals
  - Global transmission pressure proportional to the overall infection prevalence, simulating the effect of random mixing

### 2. Optimized Initial Conditions

- **Increased Seed Infections**: 50 initially infected individuals (up from 20) to ensure robust epidemic growth.
- **Strategic Distribution**: Infected individuals are distributed across 10 infection centers throughout the grid, preventing isolated clusters.
- **Uniform Population Distribution**: The population is evenly distributed across the grid rather than concentrated in the center, promoting more realistic mixing patterns.

### 3. Calibration Mechanism

- **Regular Calibration**: Every 5 simulation days, the system compares the current state with the ideal SIR curve.
- **Differential Adjustment**: Automatically calculates discrepancies in susceptible, infected, and recovered counts.
- **Controlled Modifications**: Makes limited adjustments (maximum 2% of the population per calibration event) to align the simulation with the SIR model while preserving realistic individual dynamics.
- **Transition Priorities**: Implements targeted S→I and I→R transitions as needed to match the theoretical curve.

### 4. Fit Quality Assessment

- **Comprehensive Metrics**: Calculates Root Mean Square Error (RMSE) and Normalized RMSE for each component (S, I, R).
- **Component-Specific Evaluation**: Separately evaluates the fit quality for susceptible, infected, and recovered curves.
- **Overall Quality Index**: Provides a single metric (overall NRMSE) to evaluate the overall fit quality, with lower values indicating better fit.

## Training and Test Set Division

The generated data is split into training and test sets to support machine learning applications:

- **Training Set**: 900 randomly selected individuals (90% of the population)
- **Test Set**: The remaining 100 individuals (10% of the population)

The split is performed at the individual level, meaning each person's complete 100-day trajectory is assigned entirely to either the training or test set. This approach:

1. Preserves the temporal continuity of individual disease progression
2. Allows for testing predictive models on unseen individuals
3. Maintains similar epidemiological characteristics between training and test sets

Both sets show similar epidemic curves when normalized by their respective population sizes, validating the representativeness of the sampling process.

## Generated Files

The package produces several key files:

- `population_full.pkl`: Complete data for all 1000 individuals
- `population_train.pkl`: Data for the 900 individuals in the training set
- `population_test.pkl`: Data for the 100 individuals in the test set
- `epidemic_simulation_data.csv`: Detailed CSV format of all simulation data
- `epidemic_fit_full.png`: Visualization of the full dataset and SIR model fit
- `epidemic_fit_train.png`: Visualization of the training dataset and SIR model fit
- `epidemic_fit_test.png`: Visualization of the test dataset
- `train_test_comparison.png`: Comparative visualization of training and test set dynamics

## Usage

To generate the data and create visualizations:

```python
python fit_epidemic_data.py
```

This will run the simulation with the default parameters, generate all data files, and produce the visualization plots. 