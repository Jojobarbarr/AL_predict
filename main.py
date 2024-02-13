import argparse
import json
from pathlib import Path
from experiment import Experiment

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="AL_predict", description="Evolution model from a math model over genome structure.")
    arg_parser.add_argument("config_file", type=Path, help="Configuration file for the experiments, provide absolute path.")
    args = arg_parser.parse_args()

    with open(args.config_file, "r", encoding="utf8") as json_file:
        config = json.load(json_file)

    experiment = Experiment(config)

    experiment.run()
    


    