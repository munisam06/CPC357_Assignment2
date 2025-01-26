from google.cloud import bigquery
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

PROJECT_ID = "stoked-mapper-446216-f7"
DATASET_ID = "sensor_dataset"
TABLE_ID = "updated_data"


def load_data_from_bigquery():
    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)
    # Query to extract relevant columns
    query = f"""
        SELECT
            co, smoke, lpg, humidity, temp, light, motion, timestamp
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        WHERE co IS NOT NULL AND smoke IS NOT NULL AND lpg IS NOT NULL
    """
    # Run the query and load results into a pandas DataFrame
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df)} rows from BigQuery")
    return df

# Load data into memory
df = load_data_from_bigquery()

# Preview data
print(f"dataframe preview: ")
print(df.head())


# Convert boolean columns to integers
df["light"] = df["light"].astype(int)
df["motion"] = df["motion"].astype(int)

# Define air quality categories based on thresholds for `co`, `smoke`, and `lpg`
def classify_air_quality(row):
    if row["co"] < 0.004 and row["smoke"] < 0.016 and row["lpg"] < 0.006:
        return "Good"
    elif row["co"] < 0.006 and row["smoke"] < 0.026 and row["lpg"] < 0.008:
        return "Moderate"
    else:
        return "Poor"

# Apply the classification logic
df["air_quality"] = df.apply(classify_air_quality, axis=1)

# Print the processed DataFrame for verification
print(df.head())

# Features (X) and target (y)
X = df[["co", "smoke", "lpg", "humidity", "temp"]]  # Input features
y = df["air_quality"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=200)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, "air_quality_model.pkl")
print("Model saved to air_quality_model.pkl")
# Load the trained model
model = joblib.load("air_quality_model.pkl")

def load_new_data():
    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)

    # Query the latest 10 rows of data
    query = f"""
        SELECT
            co, smoke, lpg, humidity, temp, light, motion, timestamp
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        ORDER BY timestamp DESC
        LIMIT 10000
    """
    return client.query(query).to_dataframe()

# Predict air quality
new_data = load_new_data()

# Preprocess new data
new_data["light"] = new_data["light"].astype(int)
new_data["motion"] = new_data["motion"].astype(int)

# Extract features
X_new = new_data[["co", "smoke", "lpg", "humidity", "temp"]]

# Make predictions
predictions = model.predict(X_new)
new_data["predicted_air_quality"] = predictions

# Display the predictions
print(new_data[["timestamp", "predicted_air_quality"]])



# Plot air quality categories
new_data["predicted_air_quality"].value_counts().plot(kind="bar", color=["green", "orange", "red"])
plt.title("Air Quality Distribution")
plt.xlabel("Air Quality")
plt.ylabel("Count")
plt.show()
plt.savefig("air_quality_distribution.png")
print("Plot saved as air_quality_distribution.png")

