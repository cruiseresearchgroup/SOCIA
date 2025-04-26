# 📊 Data Fitting and Synthesis

This directory contains tools for generating synthetic data for social network mask adoption behavior simulation.

## 🎯 Task Description

The main task implemented in this directory is to simulate and analyze the spread of mask-wearing behavior through social networks. The simulation models how individuals adopt mask-wearing behaviors based on:

- Social influence from different connection types (family, work/school, community)
- Government interventions and information propagation
- Personal risk perception

The model implements a multi-layered social network with agents of varying demographics, and tracks how mask-wearing behavior and information spread through the network over time. The implementation includes calibration to align with epidemiological models.

## 📁 Generated Data Description

Running `data_synthesis.py` generates several output files in the `mask_adoption_data` directory:

### Data Files

| File | Description |
|------|-------------|
| `agent_attributes.csv` | Contains demographic and behavioral attributes of each agent, including age, occupation, risk perception, and social connection counts |
| `social_network.pkl` | Pickled network data representing all connections between agents, categorized by connection type |
| `social_network.gexf` | Graph file (compatible with Gephi) for network visualization and analysis |
| `time_series_data.csv` | Daily data for each agent's mask-wearing status and information reception status |
| `daily_aggregate_data.csv` | Daily aggregated statistics across the population, including overall mask-wearing rate and information spread rate |
| `train_data.csv` | First 30 days of time series data, for model training |
| `test_data.csv` | Last 10 days of time series data, for model testing |

### Visualization Files

| File | Description |
|------|-------------|
| `simulation_results.png` | Line chart showing daily mask-wearing rates and information reception rates across the simulation period, with intervention and train/test split markers |
| `network_day_0.png` | Network visualization at the initial state (Day 0), with nodes colored by mask-wearing and information status |
| `network_day_10.png` | Network visualization at intervention start (Day 10), showing changes in mask-wearing and information propagation |
| `network_day_39.png` | Network visualization at the final simulation state (Day 39), showing the ultimate diffusion of behaviors |

## 🔬 Data Synthesis Methodology

### Population Generation

The simulation starts by generating a population of 1,000 agents with diverse attributes:

- Age distribution across four groups: Youth (0-18), Young Adult (19-40), Middle Age (41-65), and Elderly (66+)
- Occupation assignment: Student, White Collar, Blue Collar, and Retired
- Initial mask-wearing status (10% adoption rate)
- Risk perception (drawn from a beta distribution)

### Social Network Construction

A multi-layered social network is created with three connection types:

1. **Family connections** (strong ties):
   - Agents are grouped into family units of 1-6 members
   - Family members have high influence on each other (80% probability of information transmission)

2. **Work/School connections** (medium ties):
   - Agents are grouped by occupation and age
   - Small-world networks are created within each group
   - Medium influence (50% probability of information transmission)

3. **Community connections** (weak ties):
   - Random connections representing casual community interactions
   - Weak influence (30% probability of information transmission)

### Behavior Simulation

The simulation runs for 40 days with the following mechanisms:

1. **Mask-wearing behavior update**:
   - Base probability (5%)
   - Risk perception influence (weighted by 40%)
   - Social influence from connected agents (weighted by 30%)
   - Government information influence (weighted by 30%, if received)

2. **Information propagation**:
   - Government intervention begins on Day 10
   - Initial information sources are community leaders (highest-centrality nodes)
   - Information spreads through the network based on connection types
   - Probability of receiving information is highest from family connections, lower from community connections

3. **Model calibration**:
   - Information reception continues to 100% after intervention
   - Mask-wearing reaches around 57-59% by the end of the simulation

### Data Analysis

The data is split into:
- Training set (first 30 days)
- Testing set (last 10 days)

This allows for predictive modeling of mask-wearing behavior based on network features, demographic information, and early adoption patterns.

## 📏 Evaluation Metrics

The following metrics are recommended for evaluating predictive models using this dataset:

### 📉 Root Mean Square Error (RMSE)

- Highly suitable for assessing the overall predictive accuracy of mask adoption rates
- Since adoption rates are in the 0-1 range, RMSE values can be directly interpreted as percentage deviations
- Particularly useful for evaluating model performance on the test set (last 10 days)
- Formula: RMSE = √(Σ(predicted_rate - actual_rate)²/n)

### 🔝 Peak Adoption Rate Error

- Data shows mask adoption rates rise after intervention and stabilize around 57-59%
- This metric precisely measures whether a model accurately captures this peak level
- Valuable for assessing model accuracy in predicting the maximum effect of government interventions
- Formula: |predicted_peak_rate - actual_peak_rate|

### ⏱️ Time-to-Peak Error

- Especially meaningful for simulations with a clear intervention point (Day 10)
- Evaluates whether the model accurately predicts the time lag from intervention to behavior change
- Helps understand the dynamics of information propagation and behavior change
- Formula: |predicted_peak_day - actual_peak_day|

### 📈 Intervention Response Rate

- Measures the rate of change in adoption during the first 5 days after intervention (Days 10-15)
- Evaluates whether the model accurately captures the dynamics of initial response
- Helps understand the immediate effectiveness of interventions through social networks
- Formula: (rate_day15 - rate_day10) / 5

## 🚀 Usage

To run the simulation:

```bash
python data_synthesis.py
```

The script will generate all data files and visualizations in the `mask_adoption_data` directory. 