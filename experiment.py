import configparser
import math as m
from genome import Genome
import mutations
import matplotlib.pyplot as plt
import json
from pathlib import Path
import mutagenese_stat
import graphics

MUTATIONS = {
    "Point mutation": mutations.PointMutation,
    "Small insertion": mutations.SmallInsertion,
    "Small deletion": mutations.SmallDeletion,
    "Deletion": mutations.Deletion,
    "Duplication": mutations.Duplication,
}

YLIMITS = {
    "g": {
        "Point Mutation": (0, 1),
        "Small Insertion": (0, 1),
        "Small Deletion": (0, 10),
        "Deletion": (0, 500),
        "Duplication": (0, 800),
    },
}

class Experiment:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
    
    def run(self, only_plot: bool=False):
        if self.config["Experiment"]["Experiment type"] == "Mutagenese":
            if not only_plot:
                self.prepare_mutagenese()
            if self.config["Mutagenese"]["Variable"] != "No variable":
                self.magnify_mutagenese()

    def run_mutagenese(self, genome: Genome) -> list[mutations.Mutation]:
        mutation_types = self.config["Mutations"]["Mutation types"]
        experiment_repetitions = int(float(self.config["Mutagenese"]["Iterations"]))
        mutations_results = []
        for mutation_type in mutation_types:
            mutation = MUTATIONS[mutation_type](1, genome, int(self.config["Mutations"]["l_m"]))
            
            mutagenese_stat.experiment(mutation, experiment_repetitions)

            mutation.stats.compute(mutation.theory())
            mutations_results.append(mutation)
        return mutations_results

    def prepare_mutagenese(self):
        variable = self.config["Mutagenese"]["Variable"]
        g = int(float(self.config["Genome"]["g"]))
        
        z_c_factor = int(float(self.config["Genome"]["z_c_factor"]))
        z_nc_factor = int(float(self.config["Genome"]["z_nc_factor"]))
        if self.config["Genome"]["z_c_auto"]:
            z_c = z_c_factor * g
        else:
            z_c = int(float(self.config["Genome"]["z_c"]))
        
        if self.config["Genome"]["z_nc_auto"]:
            z_nc = z_nc_factor * g
        else:
            z_nc = int(float(self.config["Genome"]["z_nc"]))

        homogeneous = self.config["Genome"]["Homogeneous"]
        orientation = self.config["Genome"]["Orientation"]
        if variable != "No variable":
            for exposant in range(int(m.log10(float(self.config["Mutagenese"]["From"]))), 
                                  int(m.log10(float(self.config["Mutagenese"]["To"]))) + 1, 
                                  int(self.config["Mutagenese"]["Step"])):
                
                if variable == "g":
                    g = int(float(f"10e{exposant}"))
                    if self.config["Genome"]["z_c_auto"]:
                        z_c = z_c_factor * g
                    if self.config["Genome"]["z_nc_auto"]:
                        z_nc = z_nc_factor * g

                elif variable == "z_c":
                    z_c = int(float(f"10e{exposant}"))

                elif variable == "z_nc":
                    z_nc = int(float(f"10e{exposant}"))
                genome = Genome(g, z_c, z_nc, homogeneous, orientation) # type: ignore
                mutations_results = self.run_mutagenese(genome)
                for mutation in mutations_results:
                    graphics.save_stats(Path(self.config["Paths"]["Save directory"]) / mutation.type, f"{variable}_{exposant}", mutation.stats.d_stats)
        else:
            genome = Genome(g, z_c, z_nc, homogeneous, orientation) # type: ignore
            mutations_results = self.run_mutagenese(genome)
            for mutation in mutations_results:
                graphics.save_stats(Path(self.config["Paths"]["Save directory"]) / mutation.type, "control", mutation.stats.d_stats)
    
    def magnify_mutagenese(self):
        save_path = Path(self.config["Paths"]["Save directory"])
        variable = self.config["Mutagenese"]["Variable"]


        power_min = int(m.log10(float(self.config["Mutagenese"]["From"])))
        power_max = int(m.log10(float(self.config["Mutagenese"]["To"])))
        power_step = int(self.config["Mutagenese"]["Step"])

        x_value = [10 ** power for power in range(power_min, power_max + 1, power_step)]

        mutation_types = self.config["Mutations"]["Mutation types"]
        for mutation_type in mutation_types:
            mutation = MUTATIONS[mutation_type](1, Genome(1, 1, 1), int(self.config["Mutations"]["l_m"]))

            neutral_proportions = []
            neutral_stds =[]
            theoretical_proportions = []
            length_means = []
            length_stds = []
            
            for value in range(power_min, power_max + 1, power_step):

                with open(save_path / mutation.type / f"{variable}_{value}.json", "r", encoding="utf8") as json_file:
                    d_stats = json.load(json_file)

                neutral_proportions.append(d_stats["Neutral mutations proportion"])
                neutral_stds.append(d_stats["Neutral mutations standard deviation"])
                theoretical_proportions.append(d_stats["Neutral probability theory"])
                length_means.append(d_stats["Length mean"])
                length_stds.append(d_stats["Length standard deviation"])


            graphics.plot_and_save_mutagenese(x_value, neutral_proportions, save_path / mutation.type, f"Neutral {mutation.type} proportion", 
                                              variable, (0, 1), theoretical_proportions)
            graphics.plot_and_save_mutagenese(x_value, length_means, save_path / mutation.type, f"{mutation.type.capitalize()} length mean", 
                                              variable, YLIMITS[variable][mutation.type])