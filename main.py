import configparser
import argparse
from pathlib import Path
from experiment import Experiment

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="AL_predict", description="Evolution model from a math model over genome structure.")
    arg_parser.add_argument("config_file", type=Path, help="Configuration file for the experiments, provide absolute path.")
    args = arg_parser.parse_args()

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_file)

    experiment = Experiment(config)

    experiment.run()
    


    