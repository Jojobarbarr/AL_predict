import argparse
import json
from pathlib import Path

from mutagenese import Mutagenese
from simulation import Simulation

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="AL_predict", description="Evolution model from a math model over genome structure.")

    arg_parser.add_argument("config_file", type=Path, help="Configuration file for the experiments, provide absolute path.")
    arg_parser.add_argument("-p", "--only_plot", action="store_true", help="If used, the execution will only execute the plotting part.")
    
    args = arg_parser.parse_args()

    with open(args.config_file, "r", encoding="utf8") as json_file:
        config = json.load(json_file)

    if config["Experiment"]["Experiment type"] == "Mutagenese":
        experiment = Mutagenese(config)
    elif config["Experiment"]["Experiment type"] == "Simulation":
        experiment = Simulation(config)

    experiment.run(only_plot=args.only_plot)
    


    