import configparser
import json
import math as m
import random as rd
from pathlib import Path
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

import graphics
import mutations
from genome import Genome

from utils import str_to_int


YLIMITS_LENGTH = {
    "g": {
        "Point mutation": (0, 1),
        "Small insertion": (0, 1),
        "Small deletion": (0, 10),
        "Deletion": (0, 500),
        "Duplication": (0, 800),
        "Inversion": (0, 800),
    },
}

YLIMITS_NEAUTRALITY = {
    "g": {
        "Point mutation": (0, 0.6),
        "Small insertion": (0, 0.6),
        "Small deletion": (0, 0.6),
        "Deletion": (0, 0.015),
        "Duplication": (0, 0.03),
        "Inversion": (0, 0.03)
    },
}

class Experiment:
    def __init__(self, config: configparser.ConfigParser):
        self.experiment_config = config["Experiment"]
        self.mutations_config = config["Mutations"]
        self.genome_config = config["Genome"]
        self.mutation_rates_config = config["Mutation rates"]
        self.mutagenese_config = config["Mutagenese"]
        self.simulation_config = config["Simulation"]
        

        self.home_dir = Path(config["Paths"]["Home directory"])
        self.save_path = Path(config["Paths"]["Save directory"])
        self.checkpoints_path = Path(config["Paths"]["Checkpoint directory"])

    def save_population(self, filename: str):
        pass



