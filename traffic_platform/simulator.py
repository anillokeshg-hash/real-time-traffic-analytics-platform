"""Simulated road-side traffic sensors streaming data in real time."""
import math
import random

JUNCTIONS = [
    {"id": "J1", "name": "Central Junction", "capacity": 70},
    {"id": "J2", "name": "Market Road", "capacity": 50},
    {"id": "J3", "name": "Airport Link", "capacity": 80},
    {"id": "J4", "name": "Tech Park Gate", "capacity": 60},
    {"id": "J5", "name": "Railway Station", "capacity": 65},
    {"id": "J6", "name": "Ring Road North", "capacity": 90},
]
FREE_SPEED = 50.0   # km/h on an empty road
TICK_MINUTES = 5    # each reading covers 5 minutes of simulated time


def demand_profile(hour):
    """Share of road capacity used at a given hour (0-24): morning and evening peaks."""
    morning = math.exp(-((hour - 9) ** 2) / (2 * 1.5 ** 2))
    evening = math.exp(-((hour - 18) ** 2) / (2 * 2.0 ** 2))
    return 0.15 + 0.9 * max(morning, evening)


def make_reading(capacity, hour, incident=False):
    """One sensor reading: vehicle count, average speed, lane occupancy."""
    count = capacity * demand_profile(hour) * random.uniform(0.85, 1.15)
    load = count / capacity
    speed = FREE_SPEED * (1 - 0.7 * min(load, 1.3)) + random.uniform(-2, 2)
    occupancy = min(100.0, load * 65 + random.uniform(-4, 4))
    if incident:  # crash / breakdown: fewer vehicles get through, they crawl and queue up
        count *= 0.6
        speed *= 0.3
        occupancy = min(100.0, occupancy * 1.5 + 10)
    return {
        "count": round(max(count, 0)),
        "speed": round(max(speed, 3.0), 1),
        "occupancy": round(max(occupancy, 1.0), 1),
    }


class TrafficSimulator:
    def __init__(self, start_hour=6.0):
        self.hour = start_hour
        self.incident_left = {j["id"]: 0 for j in JUNCTIONS}

    def clock(self):
        h = int(self.hour) % 24
        m = int(round((self.hour % 1) * 60)) % 60
        return f"{h:02d}:{m:02d}"

    def step(self):
        """Advance simulated time and return one reading per junction."""
        self.hour = (self.hour + TICK_MINUTES / 60) % 24
        readings = []
        for j in JUNCTIONS:
            jid = j["id"]
            if self.incident_left[jid] > 0:
                self.incident_left[jid] -= 1
            elif random.random() < 0.02:
                self.incident_left[jid] = random.randint(3, 6)  # incident lasts 15-30 minutes
            r = make_reading(j["capacity"], self.hour, self.incident_left[jid] > 0)
            r.update({"id": jid, "name": j["name"], "capacity": j["capacity"],
                      "incident_truth": self.incident_left[jid] > 0})
            readings.append(r)
        return readings
