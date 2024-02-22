import random as rd
from configparser import ConfigParser

import numpy as np
import numpy.typing as npt
from concurrent.futures import ProcessPoolExecutor
import json

from experiment import Experiment
from genome import Genome
from mutations import Mutation
import graphics
from utils import MUTATIONS, str_to_bool, str_to_int


class Simulation(Experiment):
    def __init__(self, config: ConfigParser):
        super().__init__(config)
        print("Initializing simulation")
        self.generations = str_to_int(self.simulation_config["Generations"])
        self.plot_point = self.generations // int(self.simulation_config["Plot points"])

        mutation_types = self.mutations_config["Mutation types"]

        # Mutation rates
        self.mutation_rates = np.array([float(self.mutation_rates_config[f"{mutation_type} rate"]) 
                                        for mutation_type in mutation_types])
        self.total_mutation_rate = sum(self.mutation_rates)

        # Mutations
        l_m = int(self.mutations_config["l_m"])
        self.mutations = np.array([MUTATIONS[mutation_type](self.mutation_rates[mutation_index], genome=None, l_m=l_m)
                                   for mutation_index, mutation_type in enumerate(mutation_types)])
        self.population = str_to_int(self.simulation_config["Population size"])
        self.genomes = np.empty(self.population, dtype=Genome)
        self.init_population()
        self.living_genomes = np.zeros(len(self.genomes), dtype=bool)
        print("Simulation initialized")

    def init_population(self):
        homogeneous = self.genome_config["Homogeneous"]
        orientation = self.genome_config["Orientation"]

        g = str_to_int(self.genome_config["g"])

        # z_c
        z_c_factor = str_to_int(self.genome_config["z_c_factor"])
        if self.genome_config["z_c_auto"]:
            z_c = z_c_factor * g
        else:
            z_c = str_to_int(self.genome_config["z_c"])

        # z_nc
        z_nc_factor = str_to_int(self.genome_config["z_nc_factor"])
        if self.genome_config["z_nc_auto"]:
            z_nc = z_nc_factor * g
        else:
            z_nc = str_to_int(self.genome_config["z_nc"])

        for individual in range(self.population):
            self.genomes[individual] = Genome(g, z_c, z_nc, homogeneous, orientation) # type: ignore
    
    
    def mutation_is_deleterious(self, mutation_event: Mutation, genome_index: int, generation: int) -> bool:
        mutation_event.genome = self.genomes[genome_index]

        if mutation_event.is_neutral():
            mutation_event.apply(generation=generation)
            return False
            
        return True

    def run(self, only_plot: bool=False):
        if not only_plot:
            for generation in range(self.generations):
                if generation % self.plot_point == 0:
                    print(f"Generation {generation} / {self.generations}")
                self.living_genomes = np.ones(len(self.genomes), dtype=bool)
                
                genomes_lengths = np.array([genome.length for genome in self.genomes])
                total_bases_number = genomes_lengths.sum()
                mutation_number = rd.binomialvariate(total_bases_number, p=self.total_mutation_rate)

                biases_mutation = self.mutation_rates / self.total_mutation_rate
                mutation_events = np.random.choice(self.mutations, size=mutation_number, p=biases_mutation)

                biases_genomes = genomes_lengths / total_bases_number
                genomes_affected = np.random.choice(range(len(self.genomes)), size=mutation_number, p=biases_genomes)

                for mutation_event, genome_index in zip(mutation_events, genomes_affected):
                    if self.living_genomes[genome_index] and self.mutation_is_deleterious(mutation_event, genome_index, generation):
                        self.living_genomes[genome_index] = False

                if generation % self.plot_point == 0:
                    print(f"number of living: {self.living_genomes.sum()}")
                    for genome in self.genomes[self.living_genomes]:
                        if genome.last_change > generation - self.plot_point or not genome.stats_computed:
                            genome.compute_stats()
                
                if not self.living_genomes.any():
                    print(f"Generation {generation} - All individuals are dead")
                    break

                if generation % self.plot_point == 0:
                    genomes_stats = [genome.stats.d_stats for genome in self.genomes[self.living_genomes]]

                    living_percentage = self.living_genomes.sum() / len(self.genomes) * 100
                    z_nc_list = sorted([genome.z_nc for genome in self.genomes[self.living_genomes]])
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

                            

                parents = rd.choices(self.genomes[self.living_genomes], k=len(self.genomes))
                
                self.genomes = np.array([parent.clone() for parent in parents])

            print(f"Generation {generation} - End of simulation")

        self.plot_simulation()
    
    
    def plot_simulation(self):
        x_value = np.array([generation for generation in range(0, self.generations, self.generations // int(self.simulation_config["Plot points"]))])
        
        genomes_nc_mean = np.empty(len(x_value))
        genomes_nc_var = np.empty(len(x_value))
        genomes_nc_length_min_mean = np.empty(len(x_value))
        genomes_nc_length_min_var = np.empty(len(x_value))
        genomes_nc_length_max_mean = np.empty(len(x_value))
        genomes_nc_length_max_var = np.empty(len(x_value))
        genomes_nc_length_median_mean = np.empty(len(x_value))
        genomes_nc_length_median_var = np.empty(len(x_value))

        population_living_percentage = np.empty(len(x_value))
        population_nc_length_min = np.empty(len(x_value))
        population_nc_length_max = np.empty(len(x_value))
        population_nc_length_median = np.empty(len(x_value))


        
        for index, x in enumerate(x_value):
            with open(self.save_path / f"generation_{x}.json", "r", encoding="utf8") as json_file:
                stats = json.load(json_file)
            
            genome_raw_stats = stats["genome"]
            population_raw_stats = stats["population"]

            nc_prop = np.array([genome["Non coding proportion"] for genome in genome_raw_stats])
            genomes_nc_mean[index] = nc_prop.mean()
            genomes_nc_var[index] = nc_prop.var() * len(genome_raw_stats) / (len(genome_raw_stats) - 1)

            nc_length_min = np.array([genome["Non coding length min"] for genome in genome_raw_stats])
            genomes_nc_length_min_mean[index] = nc_length_min.mean()
            genomes_nc_length_min_var[index] = nc_length_min.var() ** 0.5

            nc_length_max = np.array([genome["Non coding length max"] for genome in genome_raw_stats])
            genomes_nc_length_max_mean[index] = nc_length_max.mean()
            genomes_nc_length_max_var[index] = nc_length_max.var() ** 0.5

            nc_length_median = np.array([genome["Non coding length median"] for genome in genome_raw_stats])
            genomes_nc_length_median_mean[index] = nc_length_median.mean()
            genomes_nc_length_median_var[index] = nc_length_median.var() ** 0.5

            population_living_percentage[index] = population_raw_stats["Living percentage"]
            population_nc_length_min[index] = population_raw_stats["z_nc min"]
            population_nc_length_max[index] = population_raw_stats["z_nc max"]
            population_nc_length_median[index] = population_raw_stats["z_nc median"]

        null_var = np.zeros(len(x_value))

        print(f"genomes_nc_length_median_mean: {genomes_nc_length_median_mean}")
        
        graphics.plot_simulation(x_value, genomes_nc_mean, genomes_nc_var, self.save_path, "Genomes non coding proportion")
        graphics.plot_simulation(x_value, genomes_nc_length_min_mean, genomes_nc_length_min_var, self.save_path, "Genomes non coding length min")
        graphics.plot_simulation(x_value, genomes_nc_length_max_mean, genomes_nc_length_max_var, self.save_path, "Genomes non coding length max")
        graphics.plot_simulation(x_value, genomes_nc_length_median_mean, genomes_nc_length_median_var, self.save_path, "Genomes non coding length median")
        graphics.plot_simulation(x_value, population_living_percentage, null_var, self.save_path, "Population living percentage")
        graphics.plot_simulation(x_value, population_nc_length_min, null_var, self.save_path, "Population z_nc min")
        graphics.plot_simulation(x_value, population_nc_length_max, null_var, self.save_path, "Population z_nc max")
        graphics.plot_simulation(x_value, population_nc_length_median, null_var, self.save_path, "Population z_nc median")




    

