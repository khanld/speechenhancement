import torch

from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from utils.model.base_model import BaseModel


class Model(BaseModel):
    def __init__(self, num_hidden_layers, num_attention_heads, intermediate_size, hidden_size):
        super().__init__()
        configuration = Wav2Vec2Config(
                                num_hidden_layers = num_hidden_layers, 
                                num_attention_heads = num_attention_heads, 
                                intermediate_size = intermediate_size,
                                hidden_size = hidden_size)
        self.backbone = Wav2Vec2Model(configuration)
        self.backbone.freeze_feature_encoder()
        self.st_classifier = nn.Sequential(
                    nn.Dropout(0.25),
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256), 
                    nn.Linear(256, 1))

    def forward(self, x):
        x = self.backbone(x, output_hidden_states = True, return_dict = True)
        emb = torch.mean(x.last_hidden_state, dim=1)
        out = self.st_classifier(emb)
        return out
    
  

if __name__ == "__main__":
    with torch.no_grad():
        inps = torch.rand(2, 16000*4)
        model = Model(3, 8, 256, 256)
        print(model(inps)[0].shape)