"""
This script analyzes raw wind data from Berlin and Munich, focusing on wind speed, direction, and seasonal variations. 
It processes raw data files, calculates wind speed and direction, and generates visualizations, including wind roses and seasonal comparisons. 
This analysis is useful for meteorologists, urban planners, and researchers studying wind patterns in these cities.
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes

def load_datasets():
    """
    Loads wind data from CSV files for Berlin and Munich, converts timestamps to datetime format, 
    sets them as the index, and sorts the data for time-series analysis.
    
    Returns:
        tuple: DataFrames for Berlin and Munich wind data.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path_1 = os.path.join(script_dir, "..", "..", "datasets", "berlin_wind_stats.csv")
        dataset_path_2 = os.path.join(script_dir, "..", "..", "datasets", "munich_wind_stats.csv")
        
        table_1 = pd.read_csv(dataset_path_1)
        table_2 = pd.read_csv(dataset_path_2)
        
        table_1['timestamp'] = pd.to_datetime(table_1['timestamp'], errors='coerce')
        table_1.set_index('timestamp', inplace=True)
        table_1.sort_index(inplace=True)
        
        table_2['timestamp'] = pd.to_datetime(table_2['timestamp'], errors='coerce')
        table_2.set_index('timestamp', inplace=True)
        table_2.sort_index(inplace=True)
        
        return table_1, table_2
    except FileNotFoundError:
        print("Error: One or more dataset files are missing.")
        return None, None

def calculate_wind_speed(speed_u, speed_v):
    """
    Calculates the wind speed using the U and V components of the wind.
    
    Args:
        speed_u (float): Wind speed component in the U direction.
        speed_v (float): Wind speed component in the V direction.
    
    Returns:
        float: Wind speed magnitude.
    """
    return np.sqrt(speed_u ** 2 + speed_v ** 2)

# Load wind datasets
berlin_table, munich_table = load_datasets()

print("Berlin Wind Data Analytics")
print(berlin_table.info())
print(berlin_table)

print("Munich Wind Data Analytics")
print(munich_table.info())

# Handling missing values
berlin_table.dropna(inplace=True)
munich_table.dropna(inplace=True)

# Compute wind speed from U and V components
berlin_table['wind_speed'] = calculate_wind_speed(berlin_table['u10m'], berlin_table['v10m'])
munich_table['wind_speed'] = calculate_wind_speed(munich_table['u10m'], munich_table['v10m'])

# Compute monthly average wind speeds
berlin_monthly_avg = berlin_table['wind_speed'].resample('ME').mean()
munich_monthly_avg = munich_table['wind_speed'].resample('ME').mean()

print("\nBerlin Monthly Average Wind Speed:")
print(berlin_monthly_avg)
print("\nMunich Monthly Average Wind Speed:")
print(munich_monthly_avg)

def determine_season(month):
    """
    Determines the season based on the month.
    
    Args:
        month (int): Numeric month value (1-12).
    
    Returns:
        str: Season name ('Winter', 'Spring', 'Summer', 'Fall').
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Assign seasons to each data point
berlin_table['season'] = berlin_table.index.month.map(determine_season)
munich_table['season'] = munich_table.index.month.map(determine_season)

# Compute seasonal wind speed averages
berlin_seasonal_avg = berlin_table.groupby('season')['wind_speed'].mean()
munich_seasonal_avg = munich_table.groupby('season')['wind_speed'].mean()

print("\nBerlin's seasonal avg wind speed:")
print(berlin_seasonal_avg)

print("\nMunich's seasonal avg wind speed:")
print(munich_seasonal_avg)

# Compute daily average wind speed
berlin_table_daily = berlin_table.resample('D')['wind_speed'].mean()
munich_table_daily = munich_table.resample('D')['wind_speed'].mean()

print("The avg wind speed by day in Berlin:")
print(berlin_table_daily)
print("Extreme wind stats for Berlin (minimum values):")
print(berlin_table_daily.nsmallest(3))

print("\nThe avg wind speed by day in Munich:")
print(munich_table_daily)
print("Extreme wind stats for Munich (maximum values):")
print(munich_table_daily.nlargest(3))

# Plot monthly wind speed comparison
plt.figure(figsize=(10, 5))
plt.plot(berlin_monthly_avg.index, berlin_monthly_avg, label="Berlin", marker='o')
plt.plot(munich_monthly_avg.index, munich_monthly_avg, label="Munich", marker='s')
plt.xlabel("Month")
plt.ylabel("Average Wind Speed (m/s)")
plt.title("Monthly Wind Speed Comparison: Berlin vs Munich")
plt.legend()
plt.grid(True)
plt.show()

# Seasonal wind speed comparison
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
x = np.arange(len(seasons))
plt.figure(figsize=(8, 5))
plt.bar(x - 0.2, berlin_seasonal_avg[seasons], width=0.4, label="Berlin", color='blue')
plt.bar(x + 0.2, munich_seasonal_avg[seasons], width=0.4, label="Munich", color='red')
plt.xticks(ticks=x, labels=seasons)
plt.xlabel("Season")
plt.ylabel("Average Wind Speed (m/s)")
plt.title("Seasonal Wind Speed Comparison: Berlin vs Munich")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

def calculate_wind_direction(speed_u, speed_v):
    """
    Calculates wind direction based on U and V wind components.
    
    Args:
        speed_u (float): Wind component in U direction.
        speed_v (float): Wind component in V direction.
    
    Returns:
        float: Wind direction in degrees.
    """
    return (np.arctan2(speed_v, speed_u) * 180 / np.pi) + 180

# Compute wind direction
berlin_table['wind_direction'] = calculate_wind_direction(berlin_table['u10m'], berlin_table['v10m'])
munich_table['wind_direction'] = calculate_wind_direction(munich_table['u10m'], munich_table['v10m'])

# Wind rose diagram for Munich
ax = WindroseAxes.from_ax()
ax.bar(munich_table['wind_direction'], munich_table['wind_speed'], normed=True, opening=0.8, edgecolor="white")
ax.set_title("Wind Rose Diagram - Munich")
plt.show()
