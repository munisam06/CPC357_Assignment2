                                                                                        publish_data.py                                                                                                    
from google.cloud  import storage
import pandas as pd
import paho.mqtt.client as mqtt
import io
from datetime import datetime, timezone
import json
import time

mqtt_broker_address = "34.60.254.174"
mqtt_topic = "sensors"

client = mqtt.Client()
client.connect(mqtt_broker_address, 1883, 60)

#gcs config
client_gcs = storage.Client()
bucket_name = "iot_data-bucket357"
file_name_in_gcs = "iot_telemetry_data_updated.csv"

#access bucket and file
bucket = client_gcs.bucket(bucket_name)
blob = bucket.blob(file_name_in_gcs)

csv_data = blob.download_as_text()

df = pd.read_csv(io.StringIO(csv_data))
df_subset = df.head(50)

for index, row in df_subset.iterrows():
        message = {
                "device": row["device"],
                "co": row["co"],
                "humidity" : row["humidity"],
                "light": row["light"],
                "lpg" : row["lpg"],
                "motion": row["motion"],
                "smoke": row["smoke"],
                "temp": row["temp"],
                "timestamp": row["timestamp"],
}

client.publish(mqtt_topic, json.dumps(message))
#print(f"Published: {message}")
print(f"Published : {message}")
print(f"Loaded {len(df)} rows from the CSV file")


time.sleep(0.1)
