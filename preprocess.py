import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler

roads = gpd.read_file('roads.geojson')
buildings = gpd.read_file('buildings.geojson')

traffic_data = pd.read_csv('traffic_data.csv')

traffic_data['datetime'] = pd.to_datetime(traffic_data['datetime'])
traffic_data = traffic_data.dropna()

scaler = MinMaxScaler()
traffic_data['normalized_traffic'] = scaler.fit_transform(traffic_data[['traffic_volume']])

roads_with_traffic = roads.merge(traffic_data, on='road_id')

roads_with_traffic.to_file('roads_with_traffic.geojson', driver='GeoJSON')