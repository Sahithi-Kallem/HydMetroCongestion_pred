import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load processed schedule
schedule = pd.read_csv('data/processed_schedule.csv')

# Define features and target (simplified feature set)
numerical_features = [
    'base_flow',  # Primary feature
    'hour', 'is_peak_hour', 'poi_factor'
]
categorical_features = ['stop_name']
target = 'congestion'

# Add dummy values for features
schedule['is_peak_hour'] = schedule['hour'].isin([8, 9, 10, 17, 18, 19]).astype(int)

# Prepare data for training
X = schedule[numerical_features + categorical_features]
y = schedule[target]

# Scale numerical features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_features])
X_numerical = pd.DataFrame(X_numerical, columns=numerical_features, index=X.index)

# Convert categorical features to dummy variables
X_categorical = pd.get_dummies(X[categorical_features], columns=categorical_features)

# Combine scaled numerical and categorical features
X = pd.concat([X_numerical, X_categorical], axis=1)

# Compute class weights
class_counts = y.value_counts()
total_samples = len(y)
class_weights = {i: total_samples / (len(class_counts) * count) for i, count in class_counts.items()}
weights = y.map(class_weights)

# Split data into training and testing sets
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# Train XGBoost model with tuned hyperparameters
model = xgb.XGBClassifier(
    eval_metric='mlogloss',
    random_state=42,
    max_depth=3,
    learning_rate=0.03,
    n_estimators=300
    # Removed scale_pos_weight
)
model.fit(X_train, y_train, sample_weight=weights_train)

# Evaluate model
y_pred = model.predict(X_test)
# Specify labels to handle missing classes in y_test
print("Model Performance:")
print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['Low', 'Medium', 'High'], zero_division=0))

# Save the model and scaler
joblib.dump(model, 'models/congestion_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature names for prediction
with open('models/feature_names.txt', 'w') as f:
    f.write(','.join(X.columns))