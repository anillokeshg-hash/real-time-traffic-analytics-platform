"""Real-Time Traffic Analytics Platform - Flask server."""
import os
from collections import deque

from flask import Flask, jsonify, send_from_directory

from models import classify, forecast_count, is_incident, train_models
from simulator import TICK_MINUTES, TrafficSimulator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

print("Training traffic models...")
classifier, detector, forecaster = train_models()

sim = TrafficSimulator(start_hour=6.0)
history = deque(maxlen=60)   # network-wide points for the chart
alerts = deque(maxlen=12)    # recent incident / congestion alerts
last_level = {}


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/live")
def live():
    """Advance the simulation one step and return analysed traffic data."""
    readings = sim.step()
    clock = sim.clock()
    junctions = []
    for r in readings:
        level = classify(classifier, r)
        incident = is_incident(detector, r)
        # forecast the vehicle count 30 minutes from now
        future_hour = sim.hour + 30 / 60
        fc = forecast_count(forecaster, future_hour, r["capacity"])
        item = {
            "id": r["id"], "name": r["name"], "count": r["count"], "speed": r["speed"],
            "occupancy": r["occupancy"], "level": level, "incident": incident,
            "forecast_count": fc, "forecast_load": round(100 * fc / r["capacity"]),
        }
        junctions.append(item)

        if incident:
            alerts.appendleft({"time": clock, "type": "Incident", "text": f"Possible incident at {r['name']}"})
        elif level == "High" and last_level.get(r["id"]) != "High":
            alerts.appendleft({"time": clock, "type": "Congestion", "text": f"Heavy congestion at {r['name']}"})
        last_level[r["id"]] = level

    total = sum(j["count"] for j in junctions)
    avg_speed = round(sum(j["speed"] for j in junctions) / len(junctions), 1)
    history.append({"time": clock, "vehicles": total, "speed": avg_speed})

    return jsonify({
        "sim_time": clock,
        "interval_min": TICK_MINUTES,
        "junctions": junctions,
        "totals": {
            "vehicles": total,
            "avg_speed": avg_speed,
            "congested": sum(j["level"] == "High" for j in junctions),
            "incidents": sum(j["incident"] for j in junctions),
        },
        "history": list(history),
        "alerts": list(alerts),
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
