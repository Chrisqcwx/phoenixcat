import json
import logging

logger = logging.getLogger(__name__)


def load_yaml(path: str):

    import yaml

    with open(path, 'r') as f:
        data = yaml.load(f)

    return data


def load_torchobj(path: str):

    import torch

    return torch.load(path, map_location='cpu')


def load_csv(path: str):
    import pandas as pd

    return pd.read_csv(path)


def load_json(path: str):

    with open(path, 'r') as f:
        data = json.load(f)

    return data
