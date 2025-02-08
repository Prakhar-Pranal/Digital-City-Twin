import dash
from dash import html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class SimulationEngine:
    def __init__(self):
        self.roads = []
        self.population = {}
        self.traffic_history = {}
        
    def add_road(self, start_point, end_point, road_id):
        self.roads.append({
            'id': road_id,
            'start': start_point,
            'end': end_point,
            'traffic': 0
        })
        
    def add_population(self, source, destination, count):
        key = f"{source}-{destination}"
        self.population[key] = count
        
    def simulate_traffic(self):
        # Simple traffic simulation based on population distribution
        for road in self.roads:
            traffic = 0
            for route, count in self.population.items():
                source, dest = route.split('-')
                # If this road is part of the route, add some traffic
                if str(road['id']) in [source, dest]:
                    traffic += count * np.random.uniform(0.5, 1.0)
            road['traffic'] = int(traffic)
            
        return {road['id']: road['traffic'] for road in self.roads}

app = dash.Dash(__name__)

# Initialize simulation engine
sim_engine = SimulationEngine()

app.layout = html.Div([
    html.H1("Interactive City Digital Twin", style={'textAlign': 'center'}),
    
    # Mode Selection
    html.Div([
        html.H3("Select Mode:"),
        dcc.Tabs(id='mode-tabs', value='historic', children=[
            dcc.Tab(label='Historic Data Visualization', value='historic'),
            dcc.Tab(label='Interactive Simulation', value='simulation')
        ])
    ]),
    
    # Historic Data View
    html.Div(id='historic-view', children=[
        html.Div([
            html.H4("Time Range Selection"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=(datetime.now() - timedelta(days=7)).date(),
                end_date=datetime.now().date()
            ),
            dcc.Graph(id='historic-traffic-graph')
        ])
    ], style={'display': 'none'}),
    
    # Simulation View
    html.Div(id='simulation-view', children=[
        html.Div([
            html.Div([
                html.H4("Add Road"),
                dcc.Input(id='road-start-x', type='number', placeholder='Start X', style={'margin': '5px'}),
                dcc.Input(id='road-start-y', type='number', placeholder='Start Y', style={'margin': '5px'}),
                dcc.Input(id='road-end-x', type='number', placeholder='End X', style={'margin': '5px'}),
                dcc.Input(id='road-end-y', type='number', placeholder='End Y', style={'margin': '5px'}),
                html.Button('Add Road', id='add-road-button', style={'margin': '5px'})
            ], style={'border': '1px solid #ddd', 'padding': '10px', 'margin': '10px'}),
            
            html.Div([
                html.H4("Add Population Flow"),
                dcc.Dropdown(id='source-road', placeholder='Source Road', style={'margin': '5px'}),
                dcc.Dropdown(id='destination-road', placeholder='Destination Road', style={'margin': '5px'}),
                dcc.Input(id='population-count', type='number', placeholder='Population Count', style={'margin': '5px'}),
                html.Button('Add Population', id='add-population-button', style={'margin': '5px'})
            ], style={'border': '1px solid #ddd', 'padding': '10px', 'margin': '10px'})
        ], style={'width': '30%', 'float': 'left'}),
        
        html.Div([
            dcc.Graph(id='simulation-graph', style={'height': '600px'}),
            html.Button('Run Simulation', id='run-simulation-button', 
                       style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#4CAF50', 'color': 'white'})
        ], style={'width': '70%', 'float': 'right'})
    ], style={'display': 'none'}),
    
    # Store components for managing state
    dcc.Store(id='roads-store', data=[]),
    dcc.Store(id='population-store', data={}),
    
    # Interval component for simulation updates
    dcc.Interval(id='simulation-interval', interval=1000, disabled=True)
])

# Callbacks for managing tab visibility
@app.callback(
    [Output('historic-view', 'style'),
     Output('simulation-view', 'style')],
    Input('mode-tabs', 'value')
)
def update_view(selected_tab):
    if selected_tab == 'historic':
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}

# Callback for adding roads
@app.callback(
    [Output('roads-store', 'data'),
     Output('source-road', 'options'),
     Output('destination-road', 'options')],
    [Input('add-road-button', 'n_clicks')],
    [State('road-start-x', 'value'),
     State('road-start-y', 'value'),
     State('road-end-x', 'value'),
     State('road-end-y', 'value'),
     State('roads-store', 'data')],
    prevent_initial_call=True
)
def add_road(n_clicks, start_x, start_y, end_x, end_y, roads):
    if not all([start_x, start_y, end_x, end_y]):
        return roads, [], []
        
    new_road = {
        'id': len(roads) + 1,
        'start': [start_x, start_y],
        'end': [end_x, end_y],
        'traffic': 0
    }
    
    roads.append(new_road)
    sim_engine.add_road([start_x, start_y], [end_x, end_y], new_road['id'])
    
    options = [{'label': f'Road {road["id"]}', 'value': road["id"]} for road in roads]
    return roads, options, options

# Callback for adding population
@app.callback(
    Output('population-store', 'data'),
    [Input('add-population-button', 'n_clicks')],
    [State('source-road', 'value'),
     State('destination-road', 'value'),
     State('population-count', 'value'),
     State('population-store', 'data')],
    prevent_initial_call=True
)
def add_population(n_clicks, source, destination, count, population):
    if not all([source, destination, count]):
        return population
        
    key = f"{source}-{destination}"
    population[key] = count
    sim_engine.add_population(source, destination, count)
    
    return population

# Callback for simulation visualization
@app.callback(
    Output('simulation-graph', 'figure'),
    [Input('simulation-interval', 'n_intervals'),
     Input('roads-store', 'data')],
    prevent_initial_call=True
)
def update_simulation(n_intervals, roads):
    if not roads:
        return go.Figure()
        
    traffic_data = sim_engine.simulate_traffic()
    
    fig = go.Figure()
    
    # Draw roads and traffic
    for road in roads:
        # Calculate color based on traffic
        traffic = traffic_data.get(road['id'], 0)
        color = f'rgb({min(255, traffic)}, {max(0, 255-traffic)}, 0)'
        
        fig.add_trace(go.Scatter(
            x=[road['start'][0], road['end'][0]],
            y=[road['start'][1], road['end'][1]],
            mode='lines',
            line=dict(color=color, width=5),
            name=f'Road {road["id"]} (Traffic: {traffic})'
        ))
        
        # Add road labels
        fig.add_annotation(
            x=(road['start'][0] + road['end'][0])/2,
            y=(road['start'][1] + road['end'][1])/2,
            text=f'Road {road["id"]}',
            showarrow=False
        )
    
    fig.update_layout(
        title='Traffic Simulation',
        showlegend=True,
        height=600,
        xaxis=dict(title='X Coordinate'),
        yaxis=dict(title='Y Coordinate')
    )
    
    return fig

# Callback for historic data visualization
@app.callback(
    Output('historic-traffic-graph', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_historic_graph(start_date, end_date):
    # Generate sample historic data
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    roads = [1, 2, 3, 4]  # Sample roads
    
    fig = go.Figure()
    
    for road in roads:
        # Generate sample traffic data with daily patterns
        traffic = [100 + 50 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 10) 
                  for i in range(len(dates))]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=traffic,
            name=f'Road {road}',
            mode='lines'
        ))
    
    fig.update_layout(
        title='Historic Traffic Data',
        xaxis_title='Date',
        yaxis_title='Traffic Volume',
        height=500
    )
    
    return fig

# Callback for simulation control
@app.callback(
    Output('simulation-interval', 'disabled'),
    Input('run-simulation-button', 'n_clicks'),
    State('simulation-interval', 'disabled')
)
def toggle_simulation(n_clicks, current_state):
    if n_clicks:
        return not current_state
    return current_state

if __name__ == '__main__':
    app.run_server(debug=True)