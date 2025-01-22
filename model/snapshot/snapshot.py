# imports
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from pathlib import Path
import yaml

# global variables
with open(Path(__file__).parent.resolve() / "../../config.yaml", "r") as file:
    _CONFIG = yaml.safe_load(file)


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_uniform_(module.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    # for other modules, using the default weight initialization strategies.


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super(TransformerEncoder, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_mask: Optional[Tensor] = None) -> Tensor:
        attention_output, _ = self.multihead_attention(query, key, value, key_padding_mask=key_mask)
        add_and_norm = self.layer_norm_1(query + attention_output)
        feedforward_output = self.feed_forward(add_and_norm)
        add = add_and_norm + feedforward_output
        output = self.layer_norm_2(add)
        return output



class ScenarioEncoder(nn.Module):
    """
    input: social_matrix (N, 8, 10)
    pre_transform: (N, 8, 10) -> (N, 8, embed_dim)
    layer_norm: (N, 8, embed_dim) -> (N, 8, embed_dim)
    permute: (N, 8, embed_dim) -> (8, N, embed_dim)
    __social_matrix: (8, N, embed_dim)__
    attention_layers: (8, N, embed_dim) -> (8, N, embed_dim)
    post_transform: (8, N, embed_dim) -> (8, N, 8)
    permute: (8, N, 8) -> (N, 8, 8)
    unsqueeze: (N, 8, 8) -> (N, 1, 8, 8)
    __scenario_matrix: (N, 16, 18, 18)__
    """ 
    def __init__(self, embed_dim: int, num_heads: int):
        super(ScenarioEncoder, self).__init__()
        self.pre_transform = nn.Sequential(
            nn.Linear(21, embed_dim),
            nn.LeakyReLU(0.01, inplace=False)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention_layers = nn.ModuleList([TransformerEncoder(embed_dim, num_heads) for _ in range(1)])
        self.post_transform = nn.Sequential(
            nn.LeakyReLU(0.01, inplace=False),
            nn.Linear(embed_dim, 8)
        )
        # depreciated, remove in next version
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=4, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=False)
        )
        
    def forward(self, social_matrix):
        social_attention_mask = (social_matrix[:,:,0] == 0) # (N, 8)
        social_matrix = self.pre_transform(social_matrix) # (N, 8, 10) -> (N, 8, embed_dim)
        social_matrix = self.layer_norm(social_matrix) # (N, 8, embed_dim)
        social_matrix = social_matrix.permute(1, 0, 2) # (N, 8, embed_dim) -> (8, N, embed_dim)

        scenario_matrix = social_matrix

        for layer in self.attention_layers:
            scenario_matrix = layer(scenario_matrix, scenario_matrix, scenario_matrix, social_attention_mask) # (8, N, embed_dim)
        
        scenario_matrix = self.post_transform(scenario_matrix) # (8, N, 8)
        scenario_matrix = scenario_matrix.permute(1, 0, 2).unsqueeze(1) # (N, 1, 8, 8)
        
        return scenario_matrix, social_matrix



class MapEncoder(nn.Module):
    """
    input: vector_matrix (N, 100, 6), social_matrix (N, 8, 10)
    pre_transform: (N, 100, 6) -> (N, 100, embed_dim)
    layer_norm: (N, 100, embed_dim) -> (N, 100, embed_dim)
    permute: (N, 100, embed_dim) -> (100, N, embed_dim)
    attention_layers: (100, N, embed_dim) -> (8, N, embed_dim)
    post_transform: (8, N, embed_dim) -> (8, N, 8)
    permute: (8, N, 8) -> (N, 8, 8)
    unsqueeze: (N, 8, 8) -> (N, 1, 8, 8)
    __map_matrix: (N, 1, 8, 8)__
    """
    def __init__(self, embed_dim: int, num_heads) -> None:
        super(MapEncoder, self).__init__()
        self.pre_transform = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.LeakyReLU(0.01, inplace=False)
            )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention_layers = nn.ModuleList([TransformerEncoder(embed_dim, num_heads) for _ in range(1)])
        self.post_transform = nn.Sequential(
            nn.LeakyReLU(0.01, inplace=False),
            nn.Linear(embed_dim, 8)
            )
        # depreciated, remove in next version
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=4, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=False)
            )
        
    def forward(self, vector_matrix, social_matrix):
        vector_attention_mask = (vector_matrix[:,:,0] == 0) # (N, 100)

        # avoid masking all vectors
        for element in vector_attention_mask:
            if element[0] == True:
                element[0] = False

        vector_matrix = self.pre_transform(vector_matrix) # (N, 100, 6) -> (N, 100, embed_dim)
        vector_matrix = self.layer_norm(vector_matrix) # (N, 100, embed_dim)
        vector_matrix = vector_matrix.permute(1, 0, 2) # (N, 100, embed_dim) -> (100, N, embed_dim)
        query = social_matrix # (8, N, embed_dim)
        
        for layer in self.attention_layers:
            map_matrix = layer(query, vector_matrix, vector_matrix, vector_attention_mask) # (8, N, embed_dim)
            query = map_matrix
        
        map_matrix = self.post_transform(map_matrix) # (8, N, 8)
        map_matrix = map_matrix.permute(1, 0, 2).unsqueeze(1) # (N, 1, 8, 8)
        
        return map_matrix


def _build_scenario_decoder():
    def decoder_conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=False) 
        ]
        return nn.Sequential(*layers)
    
    def decoder_upsamping_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=False) 
        ]
        return nn.Sequential(*layers)

    return nn.Sequential(
    decoder_upsamping_block(2, 16, padding=0), # (N, 2, 8, 8) -> (N, 16, 18, 18)
    decoder_conv_block(16, 16), # (N, 16, 18, 18) -> (N, 16, 18, 18)
    decoder_conv_block(16, 16), # (N, 16, 18, 18) -> (N, 16, 18, 18)
    decoder_upsamping_block(16, 32), # (N, 16, 18, 18) -> (N, 32, 36, 36)
    decoder_conv_block(32, 32), # (N, 32, 36, 36) -> (N, 32, 36, 36)
    decoder_conv_block(32, 16, kernel_size=5, stride=1, padding=0), # (N, 32, 36, 36) -> (N, 16, 32, 32)
    decoder_conv_block(16, 8), # (N, 16, 32, 32) -> (N, 8, 32, 32)
    decoder_conv_block(8, 1, kernel_size=3, stride=1, padding=0), # (N, 16, 30, 30) -> (N, 1, 30, 30)
    )
  

class Snapshot(nn.Module):
    def __init__(self):
        super(Snapshot, self).__init__()

        self.embed_dim = _CONFIG["model"]["EMBED_DIM"]
        self.num_heads = _CONFIG["model"]["NUM_HEADS"]

        self.scenario_encoder = ScenarioEncoder(self.embed_dim, self.num_heads)
        self.map_encoder = MapEncoder(self.embed_dim, self.num_heads)
        self.scenario_decoder = _build_scenario_decoder()

        self.decoder_post = nn.Linear(30,2)

        self.apply(initialize_weights)


    def forward(self, map_matrix: Tensor, social_matrix: Tensor) -> Tensor:
        """
        Input: map_matrix: (N, 100, 6), social_matrix: (N, 8, 10)
        self.scenario_encoder: (N, 8, 10) -> (N, 1, 8, 8), (8, N, embed_dim)
        self.map_encoder: (N, 100, 6), (8, N, embed_dim) -> (N, 1, 8, 8)
        fused_features: (N, 1, 8, 8) + (N, 1, 8, 8) -> (N, 2, 8, 8)
        self.scenario_decoder: (N, 2, 8, 8) -> (N, 1, 30, 30)
        squeeze: (N, 1, 30, 30) -> (N, 30, 30)
        self.decoder_post: (N, 30, 30) -> (N, 30, 2)
        """
        scenario_matrix, social_matrix = self.scenario_encoder(social_matrix)
        map_matrix = self.map_encoder(map_matrix, social_matrix)
        fused_features = torch.cat([map_matrix, scenario_matrix], dim=1) # concatenate along channels (N x 2 X 8 x 8)
        outputs = self.scenario_decoder(fused_features) # (N x 2 X 8 x 8) -> (N, 1, 30, 30)
        outputs = outputs.squeeze(1) # (N, 1, 30, 30) -> (N, 30, 30)
        outputs_post = self.decoder_post(outputs) # (N, 30, 30) -> (N, 30, 2)
        
        outputs_post[:, :, 0] = torch.tanh(outputs_post[:, :, 0]) # direction of velocity
        outputs_post[:, :, 1] = torch.relu(outputs_post[:, :, 1]) # magnitude of velocity
        
        return outputs_post



