import pandas as pd
import geopandas as gpd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import json

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

class PredictiveEngine:
    def __init__(self):
        self.traffic_model = RandomForestRegressor()
        self.scaler = MinMaxScaler()
        
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
    
    def predict(self, current_state):
        prediction = self.traffic_model.predict([current_state])
        return prediction[0]

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
            self.sensors.add_sensor(
                sensor_id,
                'traffic',
                (road.geometry.centroid.x, road.geometry.centroid.y)
            )
            
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

class DashboardApp:
    def __init__(self, digital_twin):
        self.digital_twin = digital_twin
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("City Digital Twin"),
            
            html.Div([
                html.Div([
                    html.H3("Traffic Map"),
                    dcc.Graph(id='traffic-map')
                ], className='six columns'),
                
                html.Div([
                    html.H3("Traffic Predictions"),
                    dcc.Graph(id='prediction-chart')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.H3("Infrastructure Status"),
                html.Div(id='infrastructure-status')
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=2000,  
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('traffic-map', 'figure'),
             Output('prediction-chart', 'figure'),
             Output('infrastructure-status', 'children')],
            Input('interval-component', 'n_intervals')
        )
        def update_dashboard(n):
            traffic_data = {
                road_id: np.random.randint(0, 100)
                for road_id in self.digital_twin.infrastructure.roads['road_id']
            }
            
            self.digital_twin.update_state(traffic_data)
            
            traffic_map = self.create_traffic_map()
            
            prediction_chart = self.create_prediction_chart()
            
            status_div = self.create_status_display()
            
            return traffic_map, prediction_chart, status_div
    
    def create_traffic_map(self):
        roads = self.digital_twin.infrastructure.roads
        
        fig = go.Figure()
        
        for idx, road in roads.iterrows():
            x, y = road.geometry.xy
            
            traffic_value = self.digital_twin.current_state.get('data', {}).get(road['road_id'], 0)
            
            color = f'rgb({min(255, traffic_value * 2.55)}, {max(0, 255 - traffic_value * 2.55)}, 0)'
            
            fig.add_trace(go.Scatter(
                x=list(x),
                y=list(y),
                mode='lines',
                line=dict(color=color, width=5),
                name=f'Road {road["road_id"]}'
            ))
        
        fig.update_layout(
            title='Real-time Traffic Map',
            showlegend=False,
            height=600
        )
        
        return fig
    
    def create_prediction_chart(self):
        predictions = []
        actual_values = []
        roads = self.digital_twin.infrastructure.roads
        
        for road_id in roads['road_id']:
            predicted_traffic = self.digital_twin.predict_traffic(road_id)
            current_traffic = self.digital_twin.current_state.get('data', {}).get(road_id, 0)
            
            if predicted_traffic is not None:
                predictions.append({
                    'road_id': road_id,
                    'traffic': predicted_traffic
                })
                actual_values.append({
                    'road_id': road_id,
                    'traffic': current_traffic
                })
        
        fig = go.Figure()
        
        if predictions and actual_values:
            fig.add_trace(go.Bar(
                x=[p['road_id'] for p in predictions],
                y=[p['traffic'] for p in predictions],
                name='Predicted Traffic'
            ))
            
            fig.add_trace(go.Bar(
                x=[a['road_id'] for a in actual_values],
                y=[a['traffic'] for a in actual_values],
                name='Current Traffic'
            ))
            
        fig.update_layout(
            title='Traffic Predictions vs Current Values',
            xaxis_title='Road ID',
            yaxis_title='Traffic Level',
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_status_display(self):
        status_items = []
        
        for road_id in self.digital_twin.infrastructure.roads['road_id']:
            traffic_value = self.digital_twin.current_state.get('data', {}).get(road_id, 0)
            status = 'Normal' if traffic_value < 70 else 'Congested'
            color = 'green' if status == 'Normal' else 'red'
            
            status_items.append(
                html.Div([
                    html.H5(f"Road {road_id}"),
                    html.P(f"Status: {status}", style={'color': color}),
                    html.P(f"Traffic Level: {traffic_value}")
                ], style={'margin': '10px', 'padding': '10px', 'border': '1px solid #ddd'})
            )
        
        return html.Div(status_items, style={'display': 'flex', 'flexWrap': 'wrap'})
    
    def run(self):
        self.app.run_server(debug=True)

if __name__ == '__main__':

    city_twin = DigitalTwin()
    city_twin.initialize('roads.geojson', 'buildings.geojson')

    dashboard = DashboardApp(city_twin)
    dashboard.run()