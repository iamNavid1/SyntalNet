import yaml
import torch
from models.SyntalNet import SyntalNet

def build_model(config):
    name = config['model']['name']
    input_dim = sum(config['model'].get('input_dims', []))
    hidden_dims = config['model']['hidden_dims']
    num_classes = config['model']['num_classes']
    if name == 'SyntalNet':
        return SyntalNet(input_dim, hidden_dims, num_classes)
    else:
        raise ValueError(f"Model {name} not supported")


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)