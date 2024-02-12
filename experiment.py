import configparser
import json
from pathlib import Path
from genome import Genome
import mutations
import mutagenese_stat
import graphics

MUTATIONS = {
    "PointMutation": mutations.PointMutation,
    "SmallInsertion": mutations.SmallInsertion,
    "SmallDeletion": mutations.SmallDeletion,
    "Deletion": mutations.Deletion,
    "Duplication": mutations.Duplication,
}

class Experiment:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        # Paths section
        self.home_dir = Path(config.get("Paths", "home directory"))
        self.save_dir = Path(config.get("Paths", "save directory"))
        self.checkpoints_dir = Path(config.get("Paths", "checkpoint directory"))

        self.check_sanity()


    
    def check_sanity(self):
        experiment_types = {"mutagenese", "simulation"}
        experiment_type = self.config.get("Id", "experiment type")
        if experiment_type not in experiment_types:
            raise ValueError(f"Experiment type must be in {experiment_types}. You provided {experiment_type}")
    
    def run(self): 
        if self.config.get("Id", "experiment type") == "mutagenese":
            self.mutagenese()

    def mutagenese(self):
        genome = Genome(int(self.config.getfloat("Initial genome", "z_c")), 
                        int(self.config.getfloat("Initial genome", "z_nc")), 
                        int(self.config.getfloat("Initial genome", "g")),
                        self.config.getboolean("Initial genome", "homogeneous"))
        
        mutation_types = json.loads(self.config.get("Mutations", "mutation type"))
        experiment_repetitions = int(self.config.getfloat("Mutagenese", "experiment repetitions"))
        for mutation_type in mutation_types:
            mutation = MUTATIONS[mutation_type](1, genome, int(self.config.getint("Mutations", "l_m")))
            
            mutagenese_stat.experiment(mutation, experiment_repetitions)

            mutation.stats.compute()
            graphics.save_stats(self.save_dir, mutation.type, mutation.stats.d_stats)