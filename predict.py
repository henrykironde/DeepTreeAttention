#OSBS mining
from src import predict
from src import data
from src import neon_paths
from glob import glob
import pandas as pd
import geopandas as gpd
from src.start_cluster import start
from distributed import wait
import os
import re

def find_rgb_files(site, year, config):
    tiles = glob(config["rgb_sensor_pool"], recursive=True)
    tiles = [x for x in tiles if site in x]
    tiles = [x for x in tiles if "/{}/".format(year) in x]
    
    return tiles
    
def convert(rgb_path, hyperspectral_pool, year, savedir):
    #convert .h5 hyperspec tile if needed
    basename = os.path.basename(rgb_path)
    geo_index = re.search("(\d+_\d+)_image", basename).group(1)
    hyperspectral_h5_path = [x for x in hyperspectral_pool if geo_index in x]
    hyperspectral_h5_path = [x for x in hyperspectral_h5_path if year in x][0]
    tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral.tif"
    tif_path = "{}/{}".format(savedir, tif_basename)
    if not os.path.exists(tif_path):
        tif_path = neon_paths.convert_h5(hyperspectral_h5_path, rgb_path, savedir)
    
    return tif_path

config = data.read_config("config.yml")
tiles = find_rgb_files(site="OSBS", config=config, year="2019")

#generate HSI_tif data if needed.
hyperspectral_pool = glob(config["HSI_sensor_pool"], recursive=True)
rgb_pool = glob(config["rgb_sensor_pool"], recursive=True)

cpu_client = start(cpus=50)

tif_futures = cpu_client.map(convert, tiles, hyperspectral_pool=hyperspectral_pool, savedir = config["HSI_tif_dir"], year="2019")
wait(tif_futures)

model_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0abd4a52fcb2453da44ae59740b4a9c8.pl"

hsi_tifs = []
for x in tif_futures:
    try:
        hsi_tifs.append(x.result())
    except:
        pass
cpu_client.close()
gpu_client = start(gpus=10, mem_size="50GB")

#No daemonic dask children
config["workers"] = 0
futures =  []
for x in hsi_tifs:
    future = gpu_client.submit(predict.predict_tile, x,model_path=model_path, config=config)
    futures.append(future)

wait(futures)

predictions = []
for future in futures:
    try:
        trees = future.result()
        if not trees.empty:
            print("No predicted trees in tile")
            predictions.append(trees)        
    except Exception as e:
        print(e)

predictions = pd.concat(predictions)
predictions = gpd.GeoDataFrame(predictions, geometry="geometry")
predictions.to_file("results/OSBS_predictions.shp")
