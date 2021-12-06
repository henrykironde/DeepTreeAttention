#RGB data integration
from src.models import Hang2020
from torch.nn import functional as F
from torch import nn, cat

class RGB(nn.Module):
    """A joint fusion model of HSI sensor data and metadata"""
    def __init__(self, classes):
        super(RGB,self).__init__()   
        pass
    
    def forward(self, x):
        features = x
        
        return features
    
class RGB_sensor_fusion(nn.Module):
    """A joint fusion model of HSI sensor data and metadata"""
    def __init__(self, bands, sites, classes):
        super(RGB_sensor_fusion,self).__init__()   
        self.RGB = RGB(classes)
        self.sensor_model = Hang2020(bands, classes)
                
        #Fully connected concat learner
        self.fc1 = nn.Linear(in_features = classes *2 , out_features = classes)
    
    def forward(self, images, metadata):
        RGB_softmax = self.metadata_model(metadata)
        sensor_softmax = self.sensor_model(images)
        concat_features = cat([RGB_softmax, sensor_softmax], dim=1)
        concat_features = self.fc1(concat_features)
        concat_features = F.relu(concat_features)
        
        return concat_features