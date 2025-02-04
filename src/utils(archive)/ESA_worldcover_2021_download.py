import geopandas as gpd

s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# load natural earth low res shapefile
ne = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# get AOI geometry 
country = 'Thailand'
geom = ne[ne.name == country].iloc[0].geometry

# load worldcover grid
url = f'{s3_url_prefix}/v100/2020/esa_worldcover_2020_grid.geojson'
grid = gpd.read_file(url)


tiles = grid[grid.intersects(geom)]
        

for tile in tiles.ll_tile:
    print(tile)