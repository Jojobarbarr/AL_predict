import configparser
import argparse
from pathlib import Path
import json

from genome import Genome
import mutations
import mutagenese_stat

MUTATIONS = {
    "PointMutation": mutations.PointMutation,
    "SmallInsertion": mutations.SmallInsertion,
    "SmallDeletion": mutations.SmallDeletion,
    "Deletion": mutations.Deletion,
    "Duplication": mutations.Duplication,
}

def check_sanity(config: configparser.ConfigParser):
    experiment_types = {"mutagenese", "simulation"}
    experiment_type = config.get("Id", "experiment_type")
    if experiment_type not in experiment_types:
        raise ValueError(f"Experiment type must be in {experiment_types}. You provided {experiment_type}")
    

def mutagenese(config: configparser.ConfigParser):
    genome = Genome(int(config.getfloat("Initial genome", "z_c")), 
                    int(config.getfloat("Initial genome", "z_nc")), 
                    int(config.getfloat("Initial genome", "g")))
    
    mutation_types = json.loads(config.get("Mutations", "mutation_type"))
    experiment_repetitions = int(config.getfloat("Mutagenese", "experiment_repetitions"))
    for mutation_type in mutation_types:
        mutation = MUTATIONS[mutation_type](1, genome, int(config.getint("Mutations", "l_m")))
        mutagenese_stat.experiment(mutation, experiment_repetitions)

        print(mutation.stats)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="AL_predict", description="Evolution model from a math model over genome structure.")
    arg_parser.add_argument("config_file", type=Path, help="Configuration file for the experiments, provide absolute path.")
    args = arg_parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)

    check_sanity(config)
    if config.get("Id", "experiment_type") == "mutagenese":
        mutagenese(config)
    


    