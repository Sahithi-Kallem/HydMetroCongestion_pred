<<<<<<< HEAD
# HydMetroCongestion_pred
Hyderabad Metro Congestion Predictor app using GTFS data and XGBoost
=======
ECHO is on.
# Hyderabad Metro Congestion Predictor

This project predicts congestion levels at Hyderabad Metro stations using GTFS data and an XGBoost model.

## Features
- Processes GTFS data to calculate passenger flows.
- Uses an XGBoost model to predict congestion levels (Low, Medium, High).
- Visualizes predictions using a Streamlit app.

## Project Structure
- `process_gtfs.py`: Processes GTFS data and calculates passenger flows.
- `train_models.py`: Trains the XGBoost model for congestion prediction.
- `app.py`: Streamlit app for visualizing predictions.
- `data/gtfs/`: Contains raw GTFS data (stops.txt, stop_times.txt, trips.txt).
- `models/`: Stores trained models and scaler (not tracked in Git).

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Sahithi-Kallem/HydMetroCongestion_pred.git
   cd HydMetroCongestion_pred

2. Install dependencies:
    pip install -r requirements.txt

3. Process GTFS data:
    python process_gtfs.py

4. Train the model:
    python train_models.py

5. Run the Streamlit app:
    streamlit run app.py

Requirements
Python 3.8+
Libraries: pandas, numpy, xgboost, scikit-learn, streamlit

Future Improvements
Add dynamic thresholds based on station type.
Implement CI/CD with GitHub Actions.
Add more visualizations for passenger flow trends.
>>>>>>> dd44183 (Add README with project overview and setup instructions)
