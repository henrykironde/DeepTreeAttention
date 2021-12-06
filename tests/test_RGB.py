#test RGB from

#Test metadata model
from src.models import RGB
from src import data
import torch
import tempfile
import os
import pytest
from pytorch_lightning import Trainer

ROOT = os.path.dirname(os.path.dirname(data.__file__))
def config():
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_samples"] = 1
    config["crop_dir"] = tempfile.gettempdir()
    config["bands"] = 3
    config["classes"] = 2
    config["top_k"] = 1
    config["convert_h5"] = False
    
    return config

#Data module
@pytest.fixture(scope="session")
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)           
    if not "GITHUB_ACTIONS" in os.environ:
        regen = False
    else:
        regen = True
    
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=regen, data_dir="{}/tests/data".format(ROOT), metadata=True) 
    dm.setup()    
    
    return dm

def test_RGB():
    m = RGB.RGB(classes=10)
    images = torch.randn(20, 3, 11, 11)    
    output = m(images)
    assert output.shape == (20,10)