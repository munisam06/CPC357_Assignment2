import paho.mqtt.client as mqtt
from google.cloud import bigquery
import json
from datetime  import datetime,timezone


# BigQuery Config
PROJECT_ID = "stoked-mapper-446216-f7"
DATASET_ID = "sensor_dataset"
TABLE_ID = "sensor_telemetry_data"
bq_client = bigquery.Client(project=PROJECT_ID)

# MQTT Configuration
mqtt_broker_address = "34.60.254.174"
mqtt_topic = "sensors"

# Callback for MQTT connection
def on_connect(client, userdata, flags, reason_code):
    if reason_code == 0:
        print("Connected to MQTT broker")
        client.subscribe(mqtt_topic)
    else:
        print(f"Failed to connect, reason code: {reason_code}")

# Callback for receiving MQTT messages
def on_message(client, userdata, message):
    payload = message.payload.decode("utf-8")
    print(f"Received message: {payload}")

    try:
        # Parse the JSON payload
        data = json.loads(payload)
        iso_timestamp = data["ts"]
        rows_to_insert = [
            {
                "device": data["device"],
                "co": data["co"],
                "humidity": data["humidity"],
                "light": data["light"],
                "lpg": data["lpg"],
                "motion": data["motion"],
                "smoke": data["smoke"],
                "temp": data["temp"],
                 "ts": data["timestamp"]

            }
        ]

        # Insert into BigQuery
        table_ref = bq_client.dataset(DATASET_ID).table(TABLE_ID)
        errors = bq_client.insert_rows_json(table_ref, rows_to_insert)
        if errors:
            print(f"Failed to insert rows: {errors}")
        else:
            print("Data inserted into BigQuery")
    except Exception as e:
        print(f"Error processing message: {e}")

# Create MQTT client and attach callbacks
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect to MQTT broker and listen
client.connect(mqtt_broker_address, 1883, 60)
client.loop_forever()


