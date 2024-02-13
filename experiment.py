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
    
    def run(self): 
        if self.config["Experiment"]["Experiment type"] == "mutagenese":
            self.mutagenese()

    def mutagenese(self):
        genome = Genome(int(float(self.config["Genome"]["z_c"])), 
                        int(float(self.config["Genome"]["z_nc"])), 
                        int(float(self.config["Genome"]["g"])),
                        self.config["Genome"]["Homogeneous"] == "true")
        
        mutation_types = self.config["Mutations"]["Mutation types"]
        experiment_repetitions = int(float(self.config["Mutagenese"]["Iterations"]))
        for mutation_type in mutation_types:
            mutation = MUTATIONS[mutation_type](1, genome, int(self.config["Mutations"]["l_m"]))
            
            mutagenese_stat.experiment(mutation, experiment_repetitions)

            mutation.stats.compute()
            graphics.save_stats(Path(self.config["Paths"]["Save directory"]), mutation.type, mutation.stats.d_stats)