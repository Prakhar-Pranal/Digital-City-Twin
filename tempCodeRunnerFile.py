from dash import Dash, dcc, html
import plotly.express as px
import geopandas as gpd

# Load preprocessed data
roads_with_traffic = gpd.read_file('roads_with_traffic.geojson')

# Extract coordinates from the geometry column
roads_with_traffic['longitude'] = roads_with_traffic.geometry.apply(lambda geom: geom.centroid.x)
roads_with_traffic['latitude'] = roads_with_traffic.geometry.apply(lambda geom: geom.centroid.y)

# Create 3D scatter plot
fig = px.scatter_3d(
    roads_with_traffic,
    x='longitude',
    y='latitude',
    z='normalized_traffic',
    color='normalized_traffic',
    title='Traffic Volume Visualization'
)

# Create Dash app