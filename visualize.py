import geopandas as gpd
import plotly.express as px

roads_with_traffic = gpd.read_file('roads_with_traffic.geojson')

roads_with_traffic['longitude'] = roads_with_traffic.geometry.apply(lambda geom: geom.centroid.x)
roads_with_traffic['latitude'] = roads_with_traffic.geometry.apply(lambda geom: geom.centroid.y)

fig = px.scatter_3d(
    roads_with_traffic,
    x='longitude',
    y='latitude',
    z='normalized_traffic',
    color='normalized_traffic',
    title='Traffic Volume Visualization'
)

fig.show()