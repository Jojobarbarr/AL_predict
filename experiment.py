import configparser
import json
import math as m
import random as rd
from pathlib import Path

import numpy as np
from tqdm import tqdm

import graphics
import mutations
from genome import Genome

MUTATIONS = {
    "Point mutation": mutations.PointMutation,
    "Small insertion": mutations.SmallInsertion,
    "Small deletion": mutations.SmallDeletion,
    "Deletion": mutations.Deletion,
    "Duplication": mutations.Duplication,
    "Inversion": mutations.Inversion,
}

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
    
    def run(self, only_plot: bool=False):
        if self.experiment_config["Experiment type"] == "Mutagenese":
            if not only_plot:
                self.mutagenese()
            if self.mutagenese_config["Variable"] != "No variable":
                self.plot_mutagenese()
        elif self.experiment_config["Experiment type"] == "Simulation":
            self.run_simulation()


    ############################## MUTAGENESE ##############################
            
    def mutagenese(self):
        mutation_types = [MUTATIONS[mutation_type] for mutation_type in self.mutations_config["Mutation types"]]
        experiment_repetitions = str_to_int(self.mutagenese_config["Iterations"])
        l_m = int(self.mutations_config["l_m"])

        variable = self.mutagenese_config["Variable"]

        results = {mutation: {} for mutation in self.mutations_config["Mutation types"]}

        if variable == "No variable":
            genome = self.prepare_mutagenese("No variable", 0)
            for mutation, name in zip(mutation_types, self.mutations_config["Mutation types"]):
                results[name][888] = self.run_mutagenese(mutation(1, genome, l_m), experiment_repetitions)
            del genome # genome can be very large

        else:
            power_min = int(m.log10(float(self.mutagenese_config["From"])))
            power_max = int(m.log10(float(self.mutagenese_config["To"])))
            power_step = int(self.mutagenese_config["Step"])
            for power in range(power_min, power_max + 1, power_step):
                genome = self.prepare_mutagenese(variable, str_to_int(f"1e{power}"))
                for mutation, name in zip(mutation_types, self.mutations_config["Mutation types"]):
                    results[name][power] = self.run_mutagenese(mutation(1, genome, l_m), experiment_repetitions)
                del genome # genome can be very large
        
        graphics.save_stats(self.save_path, results)

    def prepare_mutagenese(self, variable: str, value: int) -> Genome:
        variable = self.mutagenese_config["Variable"]

        homogeneous = self.genome_config["Homogeneous"]
        orientation = self.genome_config["Orientation"]

        if variable == "g":
            g = value
        else:
            g = str_to_int(self.genome_config["g"])
        
        if variable == "z_c":
            z_c = value
        else:
            z_c_factor = str_to_int(self.genome_config["z_c_factor"])
            if self.genome_config["z_c_auto"]:
                z_c = z_c_factor * g
            else:
                z_c = str_to_int(self.genome_config["z_c"])
        
        if variable == "z_nc":
            z_nc = value
        else:
            z_nc_factor = str_to_int(self.genome_config["z_nc_factor"])
            if self.genome_config["z_nc_auto"]:
                z_nc = z_nc_factor * g
            else:
                z_nc = str_to_int(self.genome_config["z_nc"])

        return Genome(g, z_c, z_nc, homogeneous, orientation)   

    def run_mutagenese(self, mutation: mutations.Mutation, experiment_repetitions: int) -> dict[str, float]:
        for _ in tqdm(range(experiment_repetitions), "Experiment progress... ", experiment_repetitions):
            if mutation.is_neutral():
                mutation.apply(virtually=True)

        mutation.stats.compute(mutation.theory())

        return mutation.stats.d_stats
    
    def plot_mutagenese(self):
        variable = self.mutagenese_config["Variable"]

        power_min = int(m.log10(float(self.mutagenese_config["From"])))
        power_max = int(m.log10(float(self.mutagenese_config["To"])))
        power_step = int(self.mutagenese_config["Step"])

        x_value = [10 ** power for power in range(power_min, power_max + 1, power_step)]

        mutation_types = self.mutations_config["Mutation types"]

        for mutation_type in mutation_types:
            save_dir = self.save_path / mutation_type

            neutral_proportions = []
            neutral_stds =[]
            theoretical_proportions = []
            length_means = []
            length_stds = []
            
            for power in range(power_min, power_max + 1, power_step):

                with open(save_dir / f"{power}.json", "r", encoding="utf8") as json_file:
                    d_stats = json.load(json_file)

                neutral_proportions.append(d_stats["Neutral mutations proportion"])
                neutral_stds.append(d_stats["Neutral mutations standard deviation of proportion estimator"])
                theoretical_proportions.append(d_stats["Neutral probability theory"])
                length_means.append(d_stats["Length mean"])
                length_stds.append(d_stats["Length standard deviation of mean estimator"])
            
            graphics.plot_mutagenese(x_value, neutral_proportions, neutral_stds, save_dir, f"Neutral {mutation_type.lower()} proportion", 
                                     variable, YLIMITS_NEAUTRALITY[variable][mutation_type], theoretical_proportions)
            
            graphics.plot_mutagenese(x_value, length_means, length_stds, save_dir, f"{mutation_type} length mean", 
                                     variable, YLIMITS_LENGTH[variable][mutation_type])
    
    ############################## SIMULATION ##############################

    def run_simulation(self):
        generations = str_to_int(self.simulation_config["Generations"])

        plot_point = generations // int(self.simulation_config["Plot points"])
        plot_count = 0

        genomes = self.prepare_simulation()

        l_m = int(self.mutations_config["l_m"])
        mutation_types = self.mutations_config["Mutation types"]

        mutation_rates = np.array([float(self.mutation_rates_config[f"{mutation_type} rate"]) 
                                   for mutation_type in mutation_types])
        mutations = np.array([MUTATIONS[mutation_type](mutation_rates[mutation_index], genome=None, l_m=l_m) 
                              for mutation_index, mutation_type in enumerate(mutation_types)])
        
        total_mutation_rate = sum(mutation_rates)

        generations = str_to_int(self.simulation_config["Generations"])

        for generation in range(generations):
            if generation % plot_point == 0:
                print(f"Generation {generation} / {generations}")
            living_indices = []
            genomes_stats = []
            for genome_index, genome in enumerate(genomes):
                is_dead = False
                mutation_number = rd.binomialvariate(genome.length, p=total_mutation_rate)
                mutation_events = np.random.choice(mutations, size=mutation_number, p=mutation_rates / total_mutation_rate)

                for mutation_event in mutation_events:
                    mutation_event.genome = genome
                    if mutation_event.is_neutral():
                        mutation_event.apply()
                        # genome.check(mutation_event)
                    else:
                        is_dead = True
                        break
        
                if not is_dead:
                    living_indices.append(genome_index)
                    if generation % plot_point == 0:
                        genome.compute_stats()
                        genomes_stats.append(genome.stats.d_stats)

            if generation % plot_point == 0:
                living_percentage = len(living_indices) / len(genomes) * 100
                z_nc_list = sorted([genomes[index].z_nc for index in living_indices])
                z_nc_min = z_nc_list[0]
                z_nc_max = z_nc_list[-1]
                z_nc_median = z_nc_list[len(z_nc_list) // 2]
                population_stats = {
                    "Living percentage": living_percentage,
                    "z_nc min": z_nc_min,
                    "z_nc max": z_nc_max,
                    "z_nc median": z_nc_median
                }
                graphics.save_checkpoint(self.save_path, genomes_stats, population_stats, generation)

                plot_count += 1
                           
            
            if len(living_indices) == 0:
                print(f"Generation {generation} - All individuals are dead")
                break

            new_genomes = []
            while len(new_genomes) < len(genomes):
                for index in living_indices:
                    if rd.random() < 0.5:
                        new_genomes.append(genomes[index])
            genomes = new_genomes[:len(genomes)]

        print(f"Generation {generation} - End of simulation")





    
    def prepare_simulation(self) -> list[Genome]:
        homogeneous = self.genome_config["Homogeneous"]
        orientation = self.genome_config["Orientation"]
        g = str_to_int(self.genome_config["g"])

        z_c_factor = str_to_int(self.genome_config["z_c_factor"])
        if self.genome_config["z_c_auto"]:
            z_c = z_c_factor * g
        else:
            z_c = str_to_int(self.genome_config["z_c"])

        z_nc_factor = str_to_int(self.genome_config["z_nc_factor"])
        if self.genome_config["z_nc_auto"]:
            z_nc = z_nc_factor * g
        else:
            z_nc = str_to_int(self.genome_config["z_nc"])

        population = str_to_int(self.simulation_config["Population size"])

        return [Genome(g, z_c, z_nc, homogeneous, orientation) for _ in range(population)]

def str_to_int(string: str) -> int:
    return int(float(string))


