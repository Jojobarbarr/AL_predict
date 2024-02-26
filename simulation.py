import random as rd
from configparser import ConfigParser

import numpy as np
import numpy.typing as npt
import multiprocessing as mp
import json
from time import perf_counter
from experiment import Experiment
from genome import Genome
from mutations import Mutation
import graphics
from utils import MUTATIONS, str_to_int


class Simulation(Experiment):
    def __init__(self, config: ConfigParser):
        super().__init__(config)

        self.num_workers = min(mp.cpu_count(), 32)
        self.num_workers = 4
        mp.set_start_method("fork")

        print("Initializing simulation")

        self.generations = str_to_int(self.simulation_config["Generations"])
        self.plot_point = self.generations // int(self.simulation_config["Plot points"])

        mutation_types = self.mutations_config["Mutation types"]
        # Mutation rates
        self.mutation_rates = np.array([float(self.mutation_rates_config[f"{mutation_type} rate"]) 
                                        for mutation_type in mutation_types])
        self.total_mutation_rate = sum(self.mutation_rates)
        self.biases_mutation = self.mutation_rates / self.total_mutation_rate

        # Mutations
        l_m = int(self.mutations_config["l_m"])
        self.mutations = np.array([MUTATIONS[mutation_type](self.mutation_rates[mutation_index], genome=None, l_m=l_m)
                                   for mutation_index, mutation_type in enumerate(mutation_types)])
        
        self.population = str_to_int(self.simulation_config["Population size"])
        self.genomes = np.empty(self.population, dtype=Genome)
        self.init_population()

        print(f"Population size: {(self.genomes.size * self.genomes.itemsize) // 1e3} ko")
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
            self.genomes[individual] = Genome(g, z_c, z_nc, homogeneous, orientation) # type: ignore TODO: optim?
    
    
    def mutation_is_deleterious(self, mutation_event: Mutation, genome_index: int, structure_change_genome, parallel: bool=False) -> bool:
        mutation_event.genome = self.genomes[genome_index]

        if mutation_event.is_neutral():
            mutation_event.apply()
            if parallel:
                for index in range(len(structure_change_genome)):
                    if structure_change_genome[index] == -1:
                        break
                structure_change_genome[index] = genome_index
            else:
                # the mutation changed the structure of the genome.
                structure_change_genome.add(genome_index)
            return False
            
        return True
    
    def parallel_test(self, genome_indices: list[int], mutation_events: list[Mutation], living_genomes, structure_change_genome):
        for genome_index, mutation_event in zip(genome_indices, mutation_events):
            if living_genomes[genome_index] and self.mutation_is_deleterious(mutation_event, genome_index, structure_change_genome, parallel=True):
                living_genomes[genome_index] = 0
    
    def generation_step_parallel(self,
                                 generation: int,
                                 genomes_changed: set[int],
                                 genomes_lengths: npt.NDArray[np.int_],
                                 biases_genomes: npt.NDArray[np.float_],
                                 total_bases_number: int
                                 ) -> tuple[set[int], npt.NDArray[np.int_], npt.NDArray[np.float_], int]:
        
        if generation % self.plot_point == 0:
            print(f"Generation {generation} / {self.generations}")
        
        # All individuals are alive at the beginning of a generation step.
        living_genomes = mp.Array('I', [1] * len(self.genomes))

        # list that will store all the genomes affected by a neutral deletion that changed their structure. 
        structure_change_genome = mp.Array("i", [-1] * len(self.genomes))
        
        # The lengths are only computed for genomes that were changed by a mutation.
        delta = 0
        for genome_changed in genomes_changed:
            new_length = self.genomes[genome_changed].length
            delta += new_length - genomes_lengths[genome_changed]
            genomes_lengths[genome_changed] = new_length

        total_bases_number += delta

        # The normalized length are only computed for genomes that were changed by a mutation.
        biases_genomes = genomes_lengths / total_bases_number

        # Determine the total number of event with a binomial law over all the bases.
        mutation_number = rd.binomialvariate(total_bases_number, p=self.total_mutation_rate)

        # Determine which mutations happens with a biased (over mutation rate) choice with replacement.
        mutation_events = np.random.choice(self.mutations, size=mutation_number, p=self.biases_mutation)

        # Determine whiwh genomes are affected with a biased (over the genome length) choice with replacement.
        mutant_genomes = np.random.choice(range(len(self.genomes)), size=mutation_number, p=biases_genomes)

        # Parallel computation
        processes = []
        chunk_size = mutation_number // mp.cpu_count()
        for process_index in range(self.num_workers):
            start = process_index * chunk_size

            if process_index == self.num_workers - 1:
                stop = mutation_number
            else:
                stop = (process_index + 1) * chunk_size

            process = mp.Process(target=self.parallel_test, 
                                 args=(mutant_genomes[start:stop],
                                       mutation_events[start:stop],
                                       living_genomes, 
                                       structure_change_genome
                                      )
                                )
            process.start()
            processes.append(process)
        
        for process in processes:
            process.join()

        living_genomes_np = np.frombuffer(living_genomes.get_obj(), dtype=np.int32)
        structure_change_genome_np = set(np.frombuffer(structure_change_genome.get_obj(), dtype=np.int32))
        structure_change_genome_np.remove(-1)

        if not living_genomes_np.any():
            raise RuntimeError(f"Generation {generation} - All individuals are dead.\n"
                               f"Last checkpoint at generation: {generation - ((generation - 1) % self.plot_point)}")

        if generation % self.plot_point == 0:
            if generation == 0:
                for genome in self.genomes:
                    genome.compute_stats()
            else:
                # Statistics only needs to be compute if structure has changed.
                for genome_index in structure_change_genome_np:
                    self.genomes[genome_index].compute_stats()
            
            genomes_stats = [genome.stats.d_stats for genome in self.genomes[living_genomes_np]]

            living_percentage = living_genomes_np.sum() / len(self.genomes) * 100

            # TODO: Maybe possible to optimize but must be careful here: both structure change genome and dead genome affects the statistics.
            z_nc_list = sorted([genome.z_nc for genome in self.genomes[living_genomes_np]])
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
        # Wright-Fisher model: random draw with replacement of individuals. Population size is constant.
        parents_indices = rd.choices(range(len(self.genomes[living_genomes_np])), k=len(self.genomes))

        # Create the set to map the parents with a structure change to their son (if any)
        next_generation_structure_change = set()
        for son_index, parent_index in enumerate(parents_indices):
            if parent_index in structure_change_genome_np:
                next_generation_structure_change.add(son_index)
        # As several individuals maybe clones, must ensure every instance is independant, a copy.
        self.genomes = np.array([self.genomes[living_genomes_np][parent_index].clone() for parent_index in parents_indices], dtype=Genome)
        
        return (next_generation_structure_change, genomes_lengths, biases_genomes, total_bases_number)

    def generation_step(self, 
                        generation: int, 
                        genomes_changed: set[int], 
                        genomes_lengths: npt.NDArray[np.int_], 
                        biases_genomes: npt.NDArray[np.float_], 
                        total_bases_number: int
                        ) -> tuple[set[int], npt.NDArray[np.int_], npt.NDArray[np.float_], int]:
        # All individuals are alive at the beginning of a generation step.
        living_genomes = np.ones(self.population, dtype=bool)

        # list that will store all the genomes affected by a neutral deletion that changed their structure.
        structure_change_genome = set()
        
        # The lengths are only computed for genomes that were changed by a mutation.
        delta = 0
        for genome_changed in genomes_changed:
            new_length = self.genomes[genome_changed].length
            delta += new_length - genomes_lengths[genome_changed]
            genomes_lengths[genome_changed] = new_length

        total_bases_number += delta

        # The normalized length are only computed for genomes that were changed by a mutation.
        biases_genomes = genomes_lengths / total_bases_number

        # Determine the total number of event with a binomial law over all the bases.
        mutation_number = rd.binomialvariate(total_bases_number, p=self.total_mutation_rate)

        # Determine which mutations happens with a biased (over mutation rate) choice with replacement.
        mutation_events = np.random.choice(self.mutations, size=mutation_number, p=self.biases_mutation)

        # Determine whiwh genomes are affected with a biased (over the genome length) choice with replacement.
        mutant_genomes = np.random.choice(range(len(self.genomes)), size=mutation_number, p=biases_genomes)

        # The first mutation that was drawn affects the first genome that was drawn...
        # As both are random, it doesn't introduce bias.
        for mutation_event, genome_index in zip(mutation_events, mutant_genomes):
            if living_genomes[genome_index] and self.mutation_is_deleterious(mutation_event, genome_index, structure_change_genome):
                living_genomes[genome_index] = False
    
        if not living_genomes.any():
            raise RuntimeError(f"Generation {generation} - All individuals are dead.\n"
                            f"Last checkpoint at generation: {generation - ((generation - 1) % self.plot_point)}")

        if generation % self.plot_point == 0:
            if generation == 0:
                for genome in self.genomes:
                    genome.compute_stats()
            else:
                # Statistics only needs to be compute if structure has changed.
                for genome_index in structure_change_genome:
                    self.genomes[genome_index].compute_stats()
        
            genomes_stats = [genome.stats.d_stats for genome in self.genomes[living_genomes]]

            living_percentage = living_genomes.sum() / len(self.genomes) * 100

            # TODO: Maybe possible to optimize but must be careful here: both structure change genome and dead genome affects the statistics.
            z_nc_list = sorted([genome.z_nc for genome in self.genomes[living_genomes]])
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

        
        # Wright-Fisher model: random draw with replacement of individuals. Population size is constant.
        parents_indices = rd.choices(range(len(self.genomes[living_genomes])), k=len(self.genomes))

        # Create the set to map the parents with a structure change to their son (if any)
        next_generation_structure_change = set()
        for son_index, parent_index in enumerate(parents_indices):
            if parent_index in structure_change_genome:
                next_generation_structure_change.add(son_index)
        
        # As several individuals maybe clones, must ensure every instance is independant, a copy.
        self.genomes = np.array([self.genomes[living_genomes][parent_index].clone() for parent_index in parents_indices], dtype=Genome)

        return (next_generation_structure_change, genomes_lengths, biases_genomes, total_bases_number)
        


    def run(self, only_plot: bool=False, multiprocessing: bool=False):
        if not only_plot:

            genomes_changed = {genome_index for genome_index, _ in enumerate(self.genomes)}
            genomes_lengths = np.array([genome.length for genome in self.genomes])
            total_bases_number = genomes_lengths.sum()
            biases_genomes = np.array([genome_length / total_bases_number for genome_length in genomes_lengths])

            time_perfs = []
            for generation in range(self.generations):
                if multiprocessing:
                    start_time = perf_counter()
                    genomes_changed, genomes_lengths, biases_genomes, total_bases_number = self.generation_step_parallel(generation,
                                                                                                                         genomes_changed,
                                                                                                                         genomes_lengths,
                                                                                                                         biases_genomes,
                                                                                                                         total_bases_number)
                    end_time = perf_counter()

                else:
                    start_time = perf_counter()
                    genomes_changed, genomes_lengths, biases_genomes, total_bases_number = self.generation_step(generation, 
                                                                                                                genomes_changed, 
                                                                                                                genomes_lengths, 
                                                                                                                biases_genomes, 
                                                                                                                total_bases_number)
                    end_time = perf_counter()
                    
                time_perfs.append(end_time - start_time)
                if generation % self.plot_point == 0:
                    print(f"Generation {generation} - Mean elapsed time by generation: {sum(time_perfs) / len(time_perfs)} s")
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




    

