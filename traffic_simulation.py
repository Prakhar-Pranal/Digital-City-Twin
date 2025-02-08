import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

roads = gpd.read_file('roads.geojson')

traffic_levels = {
    1: 'low',
    2: 'medium',
    3: 'high',
    4: 'low',  
    5: 'medium',  
    6: 'high'  
}

ACTIONS = ['up', 'down', 'left', 'right']


LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1 
NUM_EPISODES = 1000

Q = np.zeros((len(roads), len(ACTIONS)))  

class Car:
    def __init__(self, start_road_id, destination_road_id):
        self.current_road_id = start_road_id  
        self.destination_road_id = destination_road_id  
        self.path = []  

    def move(self, action):
        if action == 'up':
            self.current_road_id += 1 
        elif action == 'down':
            self.current_road_id -= 1  
        elif action == 'left':
            self.current_road_id -= 2  
        elif action == 'right':
            self.current_road_id += 2  

        self.current_road_id = max(1, min(self.current_road_id, len(roads)))

cars = [Car(1, 6) for _ in range(10)]  

def calculate_congestion(roads, cars):
    congestion = {road_id: 0 for road_id in roads['road_id']}
    for car in cars:
        congestion[car.current_road_id] += 1
    return congestion

def update_paths(cars, roads, Q):
    for car in cars:
        state = car.current_road_id - 1 
        if np.random.uniform(0, 1) < EPSILON:
            action = np.random.choice(ACTIONS)  
        else:
            action = ACTIONS[np.argmax(Q[state])]  

        car.move(action)

        new_state = car.current_road_id - 1  
        reward = -1  
        if car.current_road_id == car.destination_road_id:
            reward = 100  

        Q[state, ACTIONS.index(action)] += LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * np.max(Q[new_state]) - Q[state, ACTIONS.index(action)]
        )

fig, ax = plt.subplots()

def update(frame):
    ax.clear()

    for idx, road in roads.iterrows():
        x, y = road.geometry.xy
        ax.plot(x, y, color='black', linewidth=2)

        midpoint = (np.mean(x), np.mean(y))
        ax.text(midpoint[0], midpoint[1], traffic_levels.get(road['road_id'], 'low'), color='blue')

    for car in cars:

        road = roads[roads['road_id'] == car.current_road_id]
        x, y = road.geometry.iloc[0].xy
        ax.plot(np.mean(x), np.mean(y), 'ro')  

    update_paths(cars, roads, Q)

    return ax

ani = FuncAnimation(fig, update, frames=100, interval=500, repeat=False)
plt.show()