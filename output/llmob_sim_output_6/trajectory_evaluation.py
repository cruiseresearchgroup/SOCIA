#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Set, Optional
from scipy.spatial.distance import jensenshannon
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import logging
import csv
from collections import defaultdict
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))  # Go up two levels to reach project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data_fitting/llmob_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "evaluation_results")

# Constants
TIME_BINS = 144  # 10-minute intervals for a day (24h * 6)
EARTH_RADIUS = 6371  # Earth's radius in kilometers

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_json(file_path: str) -> Any:
    """Load a JSON file with error handling."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise

def load_data() -> Tuple[Dict, Dict, Dict, pd.DataFrame]:
    """
    Load all required data files.
    
    Returns:
        Tuple containing:
        - Ground truth activities data
        - Location categories data
        - POI coordinates data
        - Predicted trajectories data
    """
    logging.info("Loading data files...")
    
    # Load ground truth data
    ground_truth_path = os.path.join(DATA_DIR, "2019Y.json")
    logging.info(f"Loading ground truth data from: {ground_truth_path}")
    ground_truth = load_json(ground_truth_path)
    logging.info(f"Loaded ground truth data with {len(ground_truth)} residents")
    
    # Load location categories
    categories_path = os.path.join(DATA_DIR, "catto.json")
    logging.info(f"Loading categories data from: {categories_path}")
    categories = load_json(categories_path)
    logging.info(f"Loaded categories data with {len(categories)} categories")
    
    # Load POI coordinates
    poi_path = os.path.join(DATA_DIR, "poi_category_192021_longitude_latitude_complement_alignment_clean.json")
    logging.info(f"Loading POI data from: {poi_path}")
    poi_data = load_json(poi_path)
    logging.info(f"Loaded POI data with {len(poi_data)} categories")
    
    # Load predicted trajectories
    predictions_path = os.path.join(SCRIPT_DIR, "results/multi_day_results.csv")
    logging.info(f"Loading predictions data from: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    logging.info(f"Loaded predictions data with {len(predictions)} rows")
    
    return ground_truth, categories, poi_data, predictions

def parse_ground_truth_activities(activities: List[str], target_date: str) -> List[Dict[str, Any]]:
    """
    Parse ground truth activities for a specific date.
    
    Args:
        activities: List of activity strings from 2019Y.json
        target_date: Date string in format 'YYYY-MM-DD'
        
    Returns:
        List of parsed activity dictionaries with time, location, and activity type
    """
    parsed_activities = []
    
    # Regular expression to capture "<location> at <HH:MM:SS>" pattern
    activity_pattern = re.compile(r"(.+?) at (\d{2}:\d{2}:\d{2})")

    for activity_blob in activities:
        # Quickly skip strings that do not contain the desired target date
        header = f"Activities at {target_date}:"
        if header not in activity_blob:
            continue

        # Remove the header and the trailing period (if present)
        day_str = activity_blob.split(header)[1].strip().rstrip('.')

        # Each activity is separated by a comma followed by a space
        raw_acts = [a.strip() for a in day_str.split(',') if a.strip()]

        for raw in raw_acts:
            match = activity_pattern.match(raw)
            if not match:
                logging.warning(f"Could not parse ground-truth token: '{raw}' for resident activities on {target_date}")
                continue

            location, time_str = match.groups()

            try:
                # Combine the extracted time with the target date so that downstream calculations
                # have a full datetime object (consistent with predicted trajectories)
                date_part = datetime.strptime(target_date, "%Y-%m-%d").date()
                time_part = datetime.strptime(time_str, "%H:%M:%S").time()
                time_obj = datetime.combine(date_part, time_part)
            except ValueError:
                logging.warning(f"Invalid time format '{time_str}' in ground-truth for date {target_date}")
                continue

            parsed_activities.append({
                "time": time_obj,
                "location": location.strip(),
                # Ground-truth data does not explicitly contain an activity label; we label it as 'visit'.
                "activity_type": "visit"
            })
        # We found and processed the desired date; no need to inspect further strings for this resident
        break
    
    # Sort activities by time
    parsed_activities.sort(key=lambda x: x["time"])
    return parsed_activities

def extract_predicted_trajectory(predictions_df: pd.DataFrame, resident_id: str, 
                               simulation_date: str) -> List[Dict[str, Any]]:
    """
    Extract predicted trajectory for a specific resident and date.
    
    Args:
        predictions_df: DataFrame containing prediction data
        resident_id: ID of the resident
        simulation_date: Date string in format 'YYYY-MM-DD'
        
    Returns:
        List of trajectory points with time, location, and activity type
    """
    # Filter predictions for this resident and date
    resident_traj = predictions_df[(predictions_df['Resident ID'] == resident_id) & 
                                  (predictions_df['Simulation Date'] == simulation_date)]
    
    if resident_traj.empty:
        logging.warning(f"No predictions found for resident {resident_id} on {simulation_date}")
        return []
    
    trajectory = []
    
    for _, row in resident_traj.iterrows():
        # Skip "finished" activities as they're duplicates
        if isinstance(row['Activity'], str) and "finished" in row['Activity']:
            continue
        
        try:
            time_obj = datetime.strptime(row['Visit Time'], "%H:%M:%S")
            time_obj = datetime.combine(datetime.strptime(simulation_date, "%Y-%m-%d").date(), time_obj.time())
            
            # Extract just the activity type (e.g., "shopping" from "shopping at Mall")
            activity_type = str(row['Activity']).split(" ")[0] if isinstance(row['Activity'], str) else "unknown"
            
            trajectory.append({
                "time": time_obj,
                "location": row['Location'],
                "activity_type": activity_type,
                "duration": float(row['Duration (hours)'])
            })
        except Exception as e:
            logging.warning(f"Error parsing row for {resident_id} on {simulation_date}: {e}")
    
    # Sort trajectory by time
    trajectory.sort(key=lambda x: x["time"])
    return trajectory

def get_location_category(location: str, categories: Dict) -> str:
    """
    Get category for a location by removing any ID suffixes (e.g., '#511').
    
    Args:
        location: Location name (e.g., "Home#511")
        categories: Dictionary of categories from catto.json
        
    Returns:
        Category name, or the base location name if not found
    """
    # Extract base location name (remove #number suffix)
    base_location = location.split("#")[0] if "#" in location else location
    
    # Check if the base name is a category
    if base_location in categories:
        return base_location
    
    # If not found, return the base name as the category
    return base_location

def get_location_coordinates(location: str, poi_data: Dict) -> Optional[Tuple[float, float]]:
    """
    Get coordinates for a location from POI data.
    
    Args:
        location: Location name (e.g., "Home#511")
        poi_data: Dictionary of POI data from JSON file
        
    Returns:
        Tuple of (latitude, longitude) if found, None otherwise
    """
    # Extract base location name and ID (if present)
    base_location = location.split("#")[0] if "#" in location else location
    location_id = location.split("#")[1] if "#" in location else None
    
    # Check if the base location is a category in POI data
    if base_location in poi_data:
        poi_list = poi_data[base_location]
        
        # If we have a specific ID, try to find that exact POI
        if location_id:
            for poi in poi_list:
                if len(poi) > 2 and poi[2] == location:
                    lat, lon = float(poi[1]), float(poi[0])
                    # Ensure latitude is in the correct position (<90 in absolute value)
                    if abs(lat) > 90:
                        lat, lon = lon, lat
                    return (lat, lon)
        
        # If no ID or exact match not found, return the first POI coordinates
        if poi_list and len(poi_list[0]) >= 2:
            lat, lon = float(poi_list[0][1]), float(poi_list[0][0])
            # Ensure latitude is in the correct position (<90 in absolute value)
            if abs(lat) > 90:
                lat, lon = lon, lat
            return (lat, lon)
    
    # Default coordinates for common locations (for locations not in POI data)
    default_coords = {
        "Home": (35.6762, 139.6503),  # Default Tokyo coordinates
        "Workplace": (35.6897, 139.7006),
        "Office": (35.6897, 139.7006),
        "School": (35.7100, 139.8107),
        "University": (35.7100, 139.8107),
        "Library": (35.6762, 139.7334)
    }
    
    if base_location in default_coords:
        return default_coords[base_location]
    
    logging.warning(f"Could not find coordinates for location: {location}")
    return None

def calculate_step_distance(trajectory: List[Dict[str, Any]], 
                           poi_data: Dict) -> List[float]:
    """
    Calculate travel distance (in km) between consecutive locations in a trajectory.
    
    Args:
        trajectory: List of trajectory points
        poi_data: Dictionary of POI data from JSON file
        
    Returns:
        List of distances between consecutive points
    """
    distances = []
    
    for i in range(1, len(trajectory)):
        loc1 = trajectory[i-1]["location"]
        loc2 = trajectory[i]["location"]
        
        # Skip if same location
        if loc1 == loc2:
            distances.append(0.0)
            continue
        
        # Get coordinates
        coord1 = get_location_coordinates(loc1, poi_data)
        coord2 = get_location_coordinates(loc2, poi_data)
        
        if coord1 and coord2:
            # Calculate geodesic distance between points
            distance = geodesic(coord1, coord2).kilometers
            distances.append(distance)
        else:
            # If coordinates not found, assume a small default distance
            distances.append(0.5)  # Default 0.5 km if coordinates not available
    
    return distances

def calculate_step_interval(trajectory: List[Dict[str, Any]]) -> List[float]:
    """
    Calculate time interval (in minutes) between consecutive points in a trajectory.
    
    Args:
        trajectory: List of trajectory points
        
    Returns:
        List of time intervals in minutes
    """
    intervals = []
    
    for i in range(1, len(trajectory)):
        time1 = trajectory[i-1]["time"]
        time2 = trajectory[i]["time"]
        
        # Calculate time difference in minutes
        delta = (time2 - time1).total_seconds() / 60.0
        intervals.append(delta)
    
    return intervals

def time_to_bin(time_obj: datetime) -> int:
    """
    Convert a datetime object to a time bin index (0-143 for a day).
    
    Args:
        time_obj: Datetime object
        
    Returns:
        Time bin index (0-143)
    """
    minutes = time_obj.hour * 60 + time_obj.minute
    bin_idx = int(minutes / 10)  # 10-minute intervals
    return min(bin_idx, TIME_BINS - 1)  # Ensure within bounds

def calculate_dard(trajectory: List[Dict[str, Any]], 
                 categories: Dict) -> Dict[Tuple[int, str], int]:
    """
    Calculate Daily Activity Routine Distribution (time, category) tuples.
    
    Args:
        trajectory: List of trajectory points
        categories: Dictionary of categories from catto.json
        
    Returns:
        Dictionary mapping (time_bin, category) tuples to count
    """
    distribution = defaultdict(int)
    
    for point in trajectory:
        time_bin = time_to_bin(point["time"])
        location = point["location"]
        category = get_location_category(location, categories)
        
        distribution[(time_bin, category)] += 1
    
    return distribution

def calculate_stvd(trajectory: List[Dict[str, Any]], 
                  poi_data: Dict) -> Dict[Tuple[int, float, float], int]:
    """
    Calculate Spatial-Temporal Visits Distribution (time, lat, lon) tuples.
    
    Args:
        trajectory: List of trajectory points
        poi_data: Dictionary of POI data
        
    Returns:
        Dictionary mapping (time_bin, lat_bin, lon_bin) tuples to count
    """
    distribution = defaultdict(int)
    
    # We'll discretize coordinates to reduce the number of possible tuples
    lat_scale = 100  # Multiplier for latitude (roughly 0.01 degree resolution)
    lon_scale = 100  # Multiplier for longitude
    
    for point in trajectory:
        time_bin = time_to_bin(point["time"])
        location = point["location"]
        coords = get_location_coordinates(location, poi_data)
        
        if coords:
            lat, lon = coords
            lat_bin = int(lat * lat_scale) / lat_scale  # Round to 2 decimal places
            lon_bin = int(lon * lon_scale) / lon_scale
            
            distribution[(time_bin, lat_bin, lon_bin)] += 1
    
    return distribution

def calculate_jsd(dist1: Dict, dist2: Dict) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.
    
    Args:
        dist1: First distribution (dictionary of keys to counts)
        dist2: Second distribution (dictionary of keys to counts)
        
    Returns:
        Jensen-Shannon divergence score
    """
    # Get all unique keys
    all_keys = sorted(set(list(dist1.keys()) + list(dist2.keys())))
    
    # Convert to probability distributions
    p = np.zeros(len(all_keys))
    q = np.zeros(len(all_keys))
    
    total1 = sum(dist1.values()) or 1  # Avoid division by zero
    total2 = sum(dist2.values()) or 1
    
    for i, key in enumerate(all_keys):
        p[i] = dist1.get(key, 0) / total1
        q[i] = dist2.get(key, 0) / total2
    
    # Calculate JSD
    m = 0.5 * (p + q)
    jsd = 0.5 * jensenshannon(p, m) + 0.5 * jensenshannon(q, m)
    
    return jsd

def evaluate_trajectories(ground_truth: Dict, categories: Dict, 
                        poi_data: Dict, predictions: pd.DataFrame) -> List[Dict]:
    """
    Evaluate predicted trajectories against ground truth.
    
    Args:
        ground_truth: Dictionary of ground truth activities
        categories: Dictionary of location categories
        poi_data: Dictionary of POI data
        predictions: DataFrame of predicted trajectories
        
    Returns:
        List of evaluation results for each resident and date
    """
    # Get unique resident-date combinations from predictions
    # First, ensure resident ID column is treated as string
    predictions['Resident ID'] = predictions['Resident ID'].astype(str)
    
    # Get all unique resident-date pairs
    resident_dates = predictions[['Resident ID', 'Simulation Date']].drop_duplicates()
    logging.info(f"Found {len(resident_dates)} unique resident-date combinations")
    
    results = []
    
    for _, row in resident_dates.iterrows():
        resident_id = str(row['Resident ID'])
        sim_date = row['Simulation Date']
        
        logging.info(f"Evaluating resident {resident_id} on {sim_date}")
        
        # Skip if resident not in ground truth data
        if resident_id not in ground_truth:
            logging.warning(f"No ground truth data for resident {resident_id}")
            continue
            
        # Get predicted trajectory for this date
        pred_traj = extract_predicted_trajectory(predictions, resident_id, sim_date)
        
        if not pred_traj:
            logging.warning(f"Empty predicted trajectory for resident {resident_id} on {sim_date}")
            continue
            
        # Retrieve ground-truth activities **for the exact same simulation date**
        ground_truth_acts = parse_ground_truth_activities(
            ground_truth[resident_id], sim_date)

        if not ground_truth_acts:
            logging.warning(
                f"No ground-truth activities found for resident {resident_id} on {sim_date}. Skipping evaluation for this pair.")
            continue
        
        # Calculate metrics
        # 1. Step Distance (SD)
        pred_sd = calculate_step_distance(pred_traj, poi_data)
        gt_sd = calculate_step_distance(ground_truth_acts, poi_data)
        
        # 2. Step Interval (SI)
        pred_si = calculate_step_interval(pred_traj)
        gt_si = calculate_step_interval(ground_truth_acts)
        
        # 3. Daily Activity Routine Distribution (DARD)
        pred_dard = calculate_dard(pred_traj, categories)
        gt_dard = calculate_dard(ground_truth_acts, categories)
        
        # 4. Spatial-Temporal Visits Distribution (STVD)
        pred_stvd = calculate_stvd(pred_traj, poi_data)
        gt_stvd = calculate_stvd(ground_truth_acts, poi_data)
        
        # Calculate JSD for each metric
        # For SD and SI, we need to convert to distributions first
        # By binning the values into histograms
        
        # Step Distance JSD
        sd_bins = 20  # Number of bins for distance histogram
        max_sd = max(max(pred_sd, default=0), max(gt_sd, default=0), 1)  # Avoid division by zero
        pred_sd_hist = defaultdict(int)
        gt_sd_hist = defaultdict(int)
        
        for d in pred_sd:
            bin_idx = min(int(d * sd_bins / max_sd), sd_bins - 1)
            pred_sd_hist[bin_idx] += 1
            
        for d in gt_sd:
            bin_idx = min(int(d * sd_bins / max_sd), sd_bins - 1)
            gt_sd_hist[bin_idx] += 1
            
        sd_jsd = calculate_jsd(pred_sd_hist, gt_sd_hist)
        
        # Step Interval JSD
        si_bins = 20  # Number of bins for interval histogram
        max_si = max(max(pred_si, default=0), max(gt_si, default=0), 1)  # Avoid division by zero
        pred_si_hist = defaultdict(int)
        gt_si_hist = defaultdict(int)
        
        for i in pred_si:
            bin_idx = min(int(i * si_bins / max_si), si_bins - 1)
            pred_si_hist[bin_idx] += 1
            
        for i in gt_si:
            bin_idx = min(int(i * si_bins / max_si), si_bins - 1)
            gt_si_hist[bin_idx] += 1
            
        si_jsd = calculate_jsd(pred_si_hist, gt_si_hist)
        
        # DARD and STVD JSD
        dard_jsd = calculate_jsd(pred_dard, gt_dard)
        stvd_jsd = calculate_jsd(pred_stvd, gt_stvd)
        
        # Record results
        result = {
            "resident_id": resident_id,
            "simulation_date": sim_date,
            "day_of_week": datetime.strptime(sim_date, "%Y-%m-%d").strftime("%A"),
            "num_predicted_points": len(pred_traj),
            "num_ground_truth_points": len(ground_truth_acts),
            "sd_jsd": sd_jsd,
            "si_jsd": si_jsd,
            "dard_jsd": dard_jsd,
            "stvd_jsd": stvd_jsd,
            "average_jsd": (sd_jsd + si_jsd + dard_jsd + stvd_jsd) / 4
        }
        
        results.append(result)
        
    return results

def save_results(results: List[Dict]) -> None:
    """
    Save evaluation results to CSV file.
    
    Args:
        results: List of evaluation result dictionaries
    """
    if not results:
        logging.warning("No results to save")
        return
        
    # Save detailed results
    detailed_path = os.path.join(RESULTS_DIR, "trajectory_evaluation_detailed.csv")
    with open(detailed_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    logging.info(f"Saved detailed results to {detailed_path}")
    
    # Calculate and save summary statistics
    summary_path = os.path.join(RESULTS_DIR, "trajectory_evaluation_summary.csv")
    
    # Calculate average metrics
    avg_sd_jsd = sum(r["sd_jsd"] for r in results) / len(results)
    avg_si_jsd = sum(r["si_jsd"] for r in results) / len(results)
    avg_dard_jsd = sum(r["dard_jsd"] for r in results) / len(results)
    avg_stvd_jsd = sum(r["stvd_jsd"] for r in results) / len(results)
    avg_overall_jsd = sum(r["average_jsd"] for r in results) / len(results)
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Number of Evaluated Trajectories", len(results)])
        writer.writerow(["Average Step Distance JSD", f"{avg_sd_jsd:.4f}"])
        writer.writerow(["Average Step Interval JSD", f"{avg_si_jsd:.4f}"])
        writer.writerow(["Average DARD JSD", f"{avg_dard_jsd:.4f}"])
        writer.writerow(["Average STVD JSD", f"{avg_stvd_jsd:.4f}"])
        writer.writerow(["Average Overall JSD", f"{avg_overall_jsd:.4f}"])
    
    logging.info(f"Saved summary results to {summary_path}")

def plot_results(results: List[Dict]) -> None:
    """
    Create visualizations of evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
    """
    if not results:
        logging.warning("No results to plot")
        return
    
    # Extract metrics
    sd_jsd = [r["sd_jsd"] for r in results]
    si_jsd = [r["si_jsd"] for r in results]
    dard_jsd = [r["dard_jsd"] for r in results]
    stvd_jsd = [r["stvd_jsd"] for r in results]
    avg_jsd = [r["average_jsd"] for r in results]
    
    # Plot histograms
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(sd_jsd, bins=20, alpha=0.7)
    plt.title("Step Distance JSD")
    plt.xlabel("JSD")
    plt.ylabel("Count")
    
    plt.subplot(2, 3, 2)
    plt.hist(si_jsd, bins=20, alpha=0.7)
    plt.title("Step Interval JSD")
    plt.xlabel("JSD")
    plt.ylabel("Count")
    
    plt.subplot(2, 3, 3)
    plt.hist(dard_jsd, bins=20, alpha=0.7)
    plt.title("DARD JSD")
    plt.xlabel("JSD")
    plt.ylabel("Count")
    
    plt.subplot(2, 3, 4)
    plt.hist(stvd_jsd, bins=20, alpha=0.7)
    plt.title("STVD JSD")
    plt.xlabel("JSD")
    plt.ylabel("Count")
    
    plt.subplot(2, 3, 5)
    plt.hist(avg_jsd, bins=20, alpha=0.7)
    plt.title("Average JSD")
    plt.xlabel("JSD")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "jsd_distributions.png"))
    logging.info(f"Saved distributions plot to {os.path.join(RESULTS_DIR, 'jsd_distributions.png')}")
    
    # Box plots for each metric
    plt.figure(figsize=(12, 8))
    
    data = [sd_jsd, si_jsd, dard_jsd, stvd_jsd, avg_jsd]
    labels = ["Step Distance", "Step Interval", "DARD", "STVD", "Average"]
    
    plt.boxplot(data, labels=labels)
    plt.title("JSD by Metric Type")
    plt.ylabel("Jensen-Shannon Divergence")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(RESULTS_DIR, "jsd_boxplot.png"))
    logging.info(f"Saved boxplot to {os.path.join(RESULTS_DIR, 'jsd_boxplot.png')}")

def main():
    """Main function to run the evaluation."""
    logging.info("Starting trajectory evaluation")
    
    try:
        # Load data
        ground_truth, categories, poi_data, predictions = load_data()
        
        # Evaluate trajectories
        results = evaluate_trajectories(ground_truth, categories, poi_data, predictions)
        logging.info(f"Evaluated {len(results)} trajectories")
        
        # Save results
        save_results(results)
        
        # Plot results
        plot_results(results)
        
        logging.info("Evaluation completed successfully")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 