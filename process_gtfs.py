import pandas as pd
import numpy as np

# Load GTFS data from data/gtfs/
stops = pd.read_csv('data/gtfs/stops.txt')
stop_times = pd.read_csv('data/gtfs/stop_times.txt')
trips = pd.read_csv('data/gtfs/trips.txt')

# Merge stop_times with stops and trips to get route and station info
stop_times = stop_times.merge(trips[['trip_id', 'route_id']], on='trip_id')
stop_times = stop_times.merge(stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], on='stop_id')

# Normalize GTFS times
def normalize_gtfs_time(time_str):
    if pd.isna(time_str):
        return time_str
    hours, minutes, seconds = map(int, time_str.split(':'))
    normalized_hours = hours % 24
    return f"{normalized_hours:02d}:{minutes:02d}:{seconds:02d}"

stop_times['arrival_time'] = stop_times['arrival_time'].apply(normalize_gtfs_time)

# Extract hour from arrival_time
stop_times['hour'] = pd.to_datetime(stop_times['arrival_time'], format='%H:%M:%S').dt.hour

# Group by stop_name, route_id, and hour to calculate trip frequency
schedule = stop_times.groupby(['stop_name', 'route_id', 'hour', 'stop_lat', 'stop_lon']).size().reset_index(name='trip_count')

# Map route_id to route_long_name
schedule['route_long_name'] = schedule['route_id'].map({
    '1': 'Blue Line',
    '2': 'Red Line',
    '3': 'Green Line'
})

# Apply a baseline multiplier to adjust trip frequency by time of day
def baseline_multiplier(hour):
    if hour in [8, 9, 10, 17, 18, 19]:  # Peak hours
        return 1.5
    elif hour in [12, 13, 14]:  # Midday off-peak
        return 0.5
    else:  # Other hours
        return 0.8

schedule['baseline_multiplier'] = schedule['hour'].apply(baseline_multiplier)

# Apply a station weight to adjust trip frequency by station importance
station_weights = {
    'Ameerpet': 1.5,  # Major interchange station
    'Kukatpally': 0.8,
    'Dilsukh Nagar': 0.8,
    'Nampally': 0.8,
    'ESI Hospital': 0.6,
    # Default to 0.7 for other stations
}
schedule['station_weight'] = schedule['stop_name'].map(station_weights).fillna(0.7)

# Adjust trip count with baseline multiplier and station weight
schedule['adjusted_trip_count'] = schedule['trip_count'] * schedule['baseline_multiplier'] * schedule['station_weight']

# Calculate base passenger flow using adjusted trip count
schedule['base_flow'] = schedule['adjusted_trip_count'] * 3  # Reduced from 5 to 3 passengers per trip

# Apply a smooth peak hour multiplier
def peak_multiplier(hour):
    morning_peak = np.exp(-((hour - 9)**2) / (2 * 2**2))
    evening_peak = np.exp(-((hour - 18)**2) / (2 * 2**2))
    combined_peak = max(morning_peak, evening_peak)
    return 1 + 1.5 * combined_peak  # Reduced from 2 to 1.5 to lower the peak effect

schedule['peak_multiplier'] = schedule['hour'].apply(peak_multiplier)
schedule['base_flow'] = schedule['base_flow'] * schedule['peak_multiplier']

# Apply POI factor for busy stations
busy_stations = ['Ameerpet', 'Kukatpally', 'Dilsukh Nagar', 'Nampally']
schedule['poi_factor'] = schedule['stop_name'].apply(lambda x: 1.2 if x in busy_stations else 1.0)
schedule['base_flow'] = schedule['base_flow'] * schedule['poi_factor']

# Add new feature: relative_flow (base_flow normalized by max flow for each station)
max_flow_per_station = schedule.groupby('stop_name')['base_flow'].max()
schedule['relative_flow'] = schedule.apply(lambda row: row['base_flow'] / max_flow_per_station[row['stop_name']], axis=1)

# Add feature: rate of change in base_flow between consecutive hours
schedule = schedule.sort_values(['stop_name', 'route_id', 'hour'])
schedule['flow_change'] = schedule.groupby(['stop_name', 'route_id'])['base_flow'].diff().fillna(0)

# Define fixed thresholds
CONGESTION_THRESHOLDS = {
    'Low': 500,
    'Medium': 1500
}
print(f"Fixed Thresholds: Low < {CONGESTION_THRESHOLDS['Low']:.2f}, Medium {CONGESTION_THRESHOLDS['Low']:.2f}–{CONGESTION_THRESHOLDS['Medium']:.2f}, High > {CONGESTION_THRESHOLDS['Medium']:.2f}")

# Assign congestion levels
def assign_congestion(flow):
    if flow < CONGESTION_THRESHOLDS['Low']:
        return 0  # Low
    elif flow <= CONGESTION_THRESHOLDS['Medium']:
        return 1  # Medium
    else:
        return 2  # High

schedule['congestion'] = schedule['base_flow'].apply(assign_congestion)

# Save processed schedule
schedule.to_csv('data/processed_schedule.csv', index=False)

# Print statistics
print("Base flow statistics:")
print(schedule['base_flow'].describe())
print("\nCongestion distribution:")
print(schedule['congestion'].value_counts(normalize=True))