import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
EVENTBRITE_TOKEN = os.getenv("EVENTBRITE_TOKEN")
CALENDARIFIC_API_KEY = os.getenv("CALENDARIFIC_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Validate API keys
if not EVENTBRITE_TOKEN:
    st.error("Eventbrite token not found in environment variables.")
if not CALENDARIFIC_API_KEY:
    st.error("Calendarific API key not found in environment variables.")
if not OPENWEATHER_API_KEY:
    st.error("OpenWeatherMap API key not found in environment variables.")

# Load processed schedule
@st.cache_data
def load_schedule():
    try:
        return pd.read_csv('data/processed_schedule.csv')
    except FileNotFoundError:
        st.error("Error: 'data/processed_schedule.csv' not found. Run process_gtfs.py first.")
        return None

# Load trained model, scaler, and feature names
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/congestion_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/feature_names.txt', 'r') as f:
            feature_names = f.read().split(',')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.warning("Trained model, scaler, or feature names not found. Run train_models.py to train the model. Using threshold-based predictions only.")
        return None, None, None

def fetch_events_near_stations(stations, current_date):
    """Fetch events near stations from Eventbrite."""
    if not EVENTBRITE_TOKEN:
        return {}
    start_date = current_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"https://www.eventbriteapi.com/v3/events/search/?location.latitude=17.3850&location.longitude=78.4867&location.within=10km&token={EVENTBRITE_TOKEN}&start_date.range_start={start_date}&start_date.range_end={end_date}"
    try:
        response = requests.get(url).json()
        events = {}
        for event in response.get('events', []):
            venue = event.get('venue', {})
            if venue and 'latitude' in venue and 'longitude' in venue:
                lat, lon = float(venue['latitude']), float(venue['longitude'])
                station = nearest_station(lat, lon, stations)
                start_hour = pd.to_datetime(event['start']['local']).hour
                end_hour = pd.to_datetime(event['end']['local']).hour
                magnitude = 1.5
                if station not in events:
                    events[station] = []
                events[station].append((start_hour, end_hour, magnitude))
        return events
    except Exception as e:
        st.error(f"Failed to fetch events: {e}")
        return {}

def fetch_holidays(current_date):
    """Fetch holidays from Calendarific."""
    if not CALENDARIFIC_API_KEY:
        return {}
    year = current_date.year
    month = current_date.month
    url = f"https://calendarific.com/api/v2/holidays?&api_key={CALENDARIFIC_API_KEY}&country=IN&year={year}&month={month}"
    try:
        response = requests.get(url).json()
        holidays = response['response']['holidays']
        holiday_dates = {}
        for holiday in holidays:
            date = pd.to_datetime(holiday['date']['iso']).date()
            festival_keywords = ['holi', 'diwali', 'eid', 'ugadi', 'sankranti', 'dussehra']
            is_festival = any(keyword in holiday['name'].lower() for keyword in festival_keywords)
            holiday_dates[date] = 1.3 if is_festival else 0.8
        return holiday_dates
    except Exception as e:
        st.error(f"Failed to fetch holidays: {e}")
        return {}

def fetch_weather():
    """Fetch live weather data from OpenWeatherMap."""
    if not OPENWEATHER_API_KEY:
        return 0, 25, 60
    url = f"https://api.openweathermap.org/data/2.5/weather?lat=17.3850&lon=78.4867&appid={OPENWEATHER_API_KEY}"
    try:
        response = requests.get(url).json()
        if response.get('cod') == 200 and 'main' in response:
            rainfall = response['rain'].get('1h', 0) if 'rain' in response else 0
            temp = response['main']['temp'] - 273.15
            humidity = response['main']['humidity']
            return rainfall, temp, humidity
        else:
            raise ValueError(f"API error: {response.get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to fetch weather data: {e}")
        return 0, 25, 60

def nearest_station(lat, lon, stations):
    """Find the nearest station to a given latitude and longitude."""
    stations['distance'] = ((stations['stop_lat'] - lat)**2 + (stations['stop_lon'] - lon)**2)**0.5
    return stations.loc[stations['distance'].idxmin(), 'stop_name']

def predict_congestion(model, scaler, input_data, feature_names):
    """Predict congestion level using the trained model."""
    numerical_features = ['base_flow', 'hour', 'is_peak_hour', 'poi_factor']  # Match training features
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['stop_name'])
    
    # Add missing columns with zeros
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Ensure the columns are in the correct order
    input_df = input_df[feature_names]
    
    # Scale numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Predict
    prediction = model.predict(input_df)
    return prediction[0]

# Detect the current theme using st.get_option
theme = st.get_option("theme.base")  # Returns "light", "dark", or None

# Fallback: Use JavaScript to detect the browser's preferred color scheme
if theme is None:
    # Inject JavaScript to detect the theme and store it in session state
    components.html(
        """
        <script>
        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        window.parent.postMessage({theme: isDark ? 'dark' : 'light'}, '*');
        </script>
        """,
        height=0,
    )

    # Listen for the theme message in session state
    if "theme" not in st.session_state:
        st.session_state.theme = "light"  # Default to light if not detected
    theme = st.session_state.get("theme", "light")

# Define theme-aware colors
if theme == "dark":
    background_gradient = "linear-gradient(to right, #263238, #37474f)"  # Dark gray gradient
    sidebar_background = "#37474f"  # Dark slate gray
    text_color = "#e0e0e0"  # Light gray for text in dark theme
    label_color = "#b0bec5"  # Lighter gray for labels
    header_color = "#90caf9"  # Light blue for headers
    border_color = "#90caf9"  # Light blue for borders
    prediction_box_background = "#455a64"  # Darker gray for prediction box
    selected_station_color = "#ffffff"  # White for selected station in dark theme
    override_weather_color = "#e0e0e0"  # Light gray for "Override Weather" in dark theme
else:
    background_gradient = "linear-gradient(to right, #eceff1, #cfd8dc)"  # Light gray gradient
    sidebar_background = "#fafafa"  # Off-white
    text_color = "#263238"  # Dark gray for text in light theme
    label_color = "#263238"  # Dark gray for labels
    header_color = "#1a237e"  # Deep indigo for headers
    border_color = "#1a237e"  # Deep indigo for borders
    prediction_box_background = "#ffffff"  # White for prediction box
    selected_station_color = "#263238"  # Dark gray for selected station in light theme
    override_weather_color = "#000000"  # Black for "Override Weather" in light theme

# Custom CSS for styling
st.markdown(f"""
    <style>
    /* Import a professional font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    /* Background for the entire app */
    [data-testid="stAppViewContainer"] {{
        background: {background_gradient} !important;
        font-family: 'Roboto', sans-serif !important;
    }}

    /* Style the title */
    h1 {{
        color: {header_color} !important;
        text-align: center !important;
        font-size: 2.5em !important;
        font-weight: 700 !important;
        margin-bottom: 0.5em !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1) !important;
    }}

    /* Style the description text */
    .description {{
        color: {text_color} !important;
        font-size: 1.1em !important;
        text-align: center !important;
        margin-bottom: 2em !important;
        font-weight: 400 !important;
    }}

    /* Style the sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_background} !important;
        border-right: 2px solid {border_color} !important;
    }}

    /* Style the sidebar header */
    [data-testid="stSidebar"] h2 {{
        color: {header_color} !important;
        font-size: 1.5em !important;
        font-weight: 500 !important;
    }}

    /* Style the selectbox and slider in the sidebar */
    [data-testid="stSelectbox"] div[role="combobox"],
    [data-testid="stSlider"] div[role="slider"] {{
        background-color: {prediction_box_background} !important;
        border-radius: 5px !important;
        padding: 0.5em !important;
        border: 1px solid {border_color} !important;
    }}

    /* Style the selectbox and slider labels */
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label {{
        color: {label_color} !important;
        font-weight: 500 !important;
    }}

    /* Style the selected station text (more specific selector) */
    [data-testid="stSelectbox"] div[data-baseweb="select"] div[role="button"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] div[role="button"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div[role="button"] div {{
        color: {selected_station_color} !important;
    }}

    /* Style the selectbox dropdown items (for consistency) */
    [data-testid="stSelectbox"] ul li {{
        color: {text_color} !important;
    }}

    /* Style the slider text */
    [data-testid="stSlider"] div[data-testid="stTickBar"],
    [data-testid="stSlider"] div[role="slider"] div {{
        color: {text_color} !important;
    }}

    /* Style the info text (weather, day type, etc.) */
    .info-text {{
        color: {text_color} !important;
        font-size: 1.1em !important;
        margin: 0.5em 0 !important;
    }}

    /* Style the checkbox (more specific selector) */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label span,
    [data-testid="stCheckbox"] label div {{
        color: {override_weather_color} !important;
        font-weight: 400 !important;
    }}

    /* Style the prediction container */
    .prediction-box {{
        background-color: {prediction_box_background} !important;
        border-radius: 10px !important;
        padding: 1.5em !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        margin: 1em 0 !important;
    }}

    /* Style the prediction heading */
    .prediction-box h3 {{
        color: {header_color} !important;
        font-size: 1.5em !important;
        font-weight: 500 !important;
        margin-bottom: 1em !important;
    }}

    /* Style the prediction text */
    .prediction-box p {{
        color: {text_color} !important;
        font-size: 1.1em !important;
        margin: 0.5em 0 !important;
    }}

    /* Style the congestion labels */
    .congestion-low {{
        color: #2e7d32 !important; /* Green */
        font-weight: bold !important;
    }}
    .congestion-medium {{
        color: #f57c00 !important; /* Orange */
        font-weight: bold !important;
    }}
    .congestion-high {{
        color: #d32f2f !important; /* Red */
        font-weight: bold !important;
    }}

    /* Style the note at the bottom */
    .note {{
        color: {label_color} !important;
        font-size: 0.9em !important;
        margin-top: 2em !important;
        font-style: italic !important;
    }}

    /* Style the button (if added in the future) */
    [data-testid="stButton"] button {{
        background-color: {header_color} !important;
        color: white !important;
        border-radius: 5px !important;
        padding: 0.5em 1em !important;
        border: none !important;
        font-size: 1em !important;
        transition: background-color 0.3s !important;
    }}
    [data-testid="stButton"] button:hover {{
        background-color: {border_color} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("Hyderabad Metro Congestion Predictor")
st.markdown("""
<div class="description">
This tool predicts congestion levels at Hyderabad Metro stations based on historical data. Select a station and hour to see the predicted congestion level using both a threshold-based method and an XGBoost machine learning model.
</div>
""", unsafe_allow_html=True)

# Load schedule and model
schedule = load_schedule()
model, scaler, feature_names = load_model()

if schedule is not None and model is not None and scaler is not None and feature_names is not None:
    # Prepare stations data for event fetching
    stations_data = schedule[['stop_name', 'stop_lat', 'stop_lon']].drop_duplicates()
    stations = schedule['stop_name'].unique()

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Input Parameters")
        station = st.selectbox("Select Station", stations, help="Choose a metro station")
        hour = st.slider("Select Hour", 0, 23, 12, help="Choose the hour of the day (0-23)")

        # Fetch live weather
        rainfall, temp, humidity = fetch_weather()
        st.markdown(f"<div class='info-text'>🌤️ Live Weather: Rainfall {rainfall} mm, Temp {temp:.2f}°C, Humidity {humidity}%</div>", unsafe_allow_html=True)

        # Manual weather override
        if st.checkbox("Override Weather", help="Manually adjust weather conditions"):
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=float(rainfall))
            temp = st.number_input("Temperature (°C)", min_value=0.0, value=float(temp))
            humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=int(humidity))

    # Main content
    # Determine current day type for display
    current_date = datetime(2025, 3, 31)  # Example date
    current_day = current_date.weekday()
    display_day_type = 1 if current_day >= 5 else 0
    st.markdown(f"<div class='info-text'>📅 Day Type: {'Weekend' if display_day_type == 1 else 'Weekday'}</div>", unsafe_allow_html=True)

    # Fetch holidays dynamically
    holidays = fetch_holidays(current_date)
    holiday_factor = holidays.get(current_date.date(), 1.0)
    if holiday_factor != 1.0:
        st.markdown(f"<div class='info-text'>🎉 Holiday Detected: {holiday_factor}x flow adjustment</div>", unsafe_allow_html=True)

    # Fetch events dynamically
    events = fetch_events_near_stations(stations_data, current_date)
    event_factor = 1.0
    for station_name, event_list in events.items():
        if station_name == station:
            for start_hour, end_hour, magnitude in event_list:
                if start_hour <= hour <= end_hour:
                    event_factor = magnitude
                    st.markdown(f"<div class='info-text'>🎫 Event Detected Near {station}: {event_factor}x flow adjustment</div>", unsafe_allow_html=True)
                    break

    # Filter schedule for the selected station and hour
    mask = (schedule['stop_name'] == station) & (schedule['hour'] == hour)
    station_data = schedule[mask]

    # Display prediction
    if not station_data.empty:
        # Use the first row for static values
        row = station_data.iloc[0]
        base_flow = row['base_flow']
        congestion = row['congestion']
        congestion_label = ["Low", "Medium", "High"][int(congestion)]

        # Adjust base_flow for live weather and events
        adjusted_flow = base_flow
        adjusted_flow *= event_factor
        adjusted_flow *= 0.9 if rainfall > 0 else 1.0
        adjusted_flow *= 0.95 if temp > 35 else 1.0
        adjusted_flow *= 0.95 if humidity > 80 else 1.0

        # Prepare input data for ML prediction
        input_data = {
            'base_flow': adjusted_flow,
            'hour': hour,
            'is_peak_hour': 1 if hour in [8, 9, 10, 17, 18, 19] else 0,
            'poi_factor': 1.2 if station in ['Ameerpet', 'Kukatpally', 'Dilsukh Nagar', 'Nampally'] else 1.0,
            'stop_name': station
        }

        # ML-based prediction
        congestion_ml = predict_congestion(model, scaler, input_data, feature_names)
        congestion_label_ml = ["Low", "Medium", "High"][int(congestion_ml)]

        # Display prediction in a styled container
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🚇 Predicted Congestion at {station} at {hour}:00</h3>
            <p>📊 Base Passenger Flow: {base_flow:.2f} passengers/hour</p>
            <p>📈 Adjusted Passenger Flow (with events/weather): {adjusted_flow:.2f} passengers/hour</p>
            <p>🔲 Congestion Level (Threshold-Based): <span class="congestion-{congestion_label.lower()}">{congestion_label}</span></p>
            <p>🤖 Congestion Level (XGBoost Model): <span class="congestion-{congestion_label_ml.lower()}">{congestion_label_ml}</span></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No data available for this station and hour.")

    # Add a note about limitations
    st.markdown("""
    <div class="note">
    <strong>Note:</strong> This tool uses static GTFS data and historical passenger flows due to the unavailability of real-time data.
    Predictions are based on averages and may not reflect current conditions. The XGBoost model enhances predictions
    by learning patterns from historical data.
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Please ensure the schedule data, model, and scaler are available to proceed.")
