import pandas as pd
import geopandas as gpd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ['up', 'down', 'left', 'right']
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1
NUM_EPISODES = 1000

class Sensor:
    def __init__(self, sensor_id, sensor_type, location):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.location = location
        self.readings = []

    def add_reading(self, value, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        self.readings.append({
            'timestamp': timestamp,
            'value': value
        })

    def get_latest_reading(self):
        if self.readings:
            return self.readings[-1]
        return None

class SensorNetwork:
    def __init__(self):
        self.sensors = {}

    def add_sensor(self, sensor_id, sensor_type, location):
        self.sensors[sensor_id] = Sensor(sensor_id, sensor_type, location)

    def add_reading(self, sensor_id, value, timestamp=None):
        if sensor_id in self.sensors:
            self.sensors[sensor_id].add_reading(value, timestamp)

    def get_all_readings(self, sensor_type=None):
        readings = []
        for sensor in self.sensors.values():
            if sensor_type is None or sensor.sensor_type == sensor_type:
                readings.extend(sensor.readings)
        return readings

class TrafficModel(nn.Module):
    def __init__(self):
        super(TrafficModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out

class PredictiveEngine:
    def __init__(self):
        self.traffic_model = RandomForestRegressor()
        self.scaler = MinMaxScaler()
        self.Q = np.zeros((len(ACTIONS), len(ACTIONS)))  
        self.lstm_model = TrafficModel()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)

    def prepare_data(self, historical_data):
        features = []
        targets = []

        for i in range(len(historical_data) - 1):
            features.append([
                historical_data[i]['value'],
                historical_data[i]['timestamp'].hour,
                historical_data[i]['timestamp'].weekday()
            ])
            targets.append(historical_data[i + 1]['value'])

        return np.array(features), np.array(targets)

    def train(self, historical_data):
        features, targets = self.prepare_data(historical_data)
        if len(features) > 0:
            self.traffic_model.fit(features, targets)
        
        sequences = [data['value'] for data in historical_data]
        max_seq_length = max(len(seq) for seq in sequences)
        X = []
        y = []

        for seq in sequences:
            for i in range(len(seq) - 1):
                X.append(seq[i])
                y.append(seq[i + 1])

        X = torch.tensor(X, dtype=torch.float32).view(-1, 1)  
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  

        for epoch in range(20):
            self.optimizer.zero_grad()
            outputs = self.lstm_model(X.unsqueeze(1))  
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, current_state):
        prediction = self.traffic_model.predict([current_state])
        return prediction[0]

    def predict_with_lstm(self, sequence):
        sequence = torch.tensor(sequence, dtype=torch.float32).view(1, -1, 1)
        prediction = self.lstm_model(sequence)
        return prediction.item()

    def optimize(self, current_state):
        state = current_state[0] % len(ACTIONS) 
        if np.random.uniform(0, 1) < EPSILON:
            action = np.random.choice(ACTIONS)  
        else:
            action = ACTIONS[np.argmax(self.Q[state])]  
        new_state = (state + ACTIONS.index(action)) % len(ACTIONS)  
        reward = 1 if new_state == 3 else 0 
        self.Q[state, ACTIONS.index(action)] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(self.Q[new_state]) - self.Q[state, ACTIONS.index(action)])
        return action

class Infrastructure:
    def __init__(self):
        self.roads = None
        self.buildings = None
        self.sensors = SensorNetwork()
        self.status = {}

    def load_infrastructure(self, roads_file, buildings_file):
        self.roads = gpd.read_file(roads_file)
        self.buildings = gpd.read_file(buildings_file)

        for idx, road in self.roads.iterrows():
            sensor_id = f"traffic_sensor_{road['road_id']}"
            self.sensors.add_sensor(sensor_id, 'traffic', (road.geometry.centroid.x, road.geometry.centroid.y))

    def update_status(self, component_id, status):
        self.status[component_id] = {
            'status': status,
            'last_updated': datetime.now()
        }

class DigitalTwin:
    def __init__(self):
        self.infrastructure = Infrastructure()
        self.predictive_engine = PredictiveEngine()
        self.current_state = {}
        self.historical_states = []

    def initialize(self, roads_file, buildings_file):
        self.infrastructure.load_infrastructure(roads_file, buildings_file)

    def update_state(self, new_data):
        timestamp = datetime.now()
        self.current_state = {
            'timestamp': timestamp,
            'data': new_data
        }
        self.historical_states.append(self.current_state)

        for road_id, traffic_value in new_data.items():
            sensor_id = f"traffic_sensor_{road_id}"
            self.infrastructure.sensors.add_reading(sensor_id, traffic_value, timestamp)

    def predict_traffic(self, road_id):
        sensor_id = f"traffic_sensor_{road_id}"
        historical_data = self.infrastructure.sensors.get_all_readings('traffic')

        if len(historical_data) > 1:
            self.predictive_engine.train(historical_data)
            current_state = [
                historical_data[-1]['value'],
                datetime.now().hour,
                datetime.now().weekday()
            ]
            return self.predictive_engine.predict(current_state)
        return None

    def predict_traffic_with_lstm(self, road_id):
        sensor_id = f"traffic_sensor_{road_id}"
        historical_data = self.infrastructure.sensors.get_all_readings('traffic')

        if len(historical_data) > 1:
            sequence = [data['value'] for data in historical_data[-10:]]
            return self.predictive_engine.predict_with_lstm(sequence)
        return None

    def optimize_traffic(self, road_id):
        current_state = [self.current_state['data'][road_id]]
        action = self.predictive_engine.optimize(current_state)
        return action

class DashboardApp:
    def __init__(self, digital_twin):
        self.digital_twin = digital_twin
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html