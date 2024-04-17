import json
import math as m
from argparse import Namespace
from typing import Any

from tqdm import tqdm

import graphics
import mutations
from experiment import Experiment
from genome import Genome
from utils import L_M, MUTATIONS, str_to_int


class Mutagenese(Experiment):
    """Mutagenese experiment: one specific mutation is applied virtually to the genome (the genome is constant).

    Attributes:
        variable (str): The variable to change during the experiment.
        l_m (int): The mutation length.
        mutation_length_distribution (str): The mutation length distribution.
        homogeneous (bool): The genome homogeneity.
        orientation (bool): The genome orientation.
        power_min (int): The minimum power of the variable.
        power_max (int): The maximum power of the variable.
        power_step (int): The step between each power.
        mutation_names (list[str]): The names of the mutations.
        mutation_types (list[mutations.Mutation]): The mutations.
        experiment_repetitions (int): The number of repetitions for each experiment.
        results (dict[str, dict[int, dict[str, float]]]): The results of the experiments.
    """

    def __init__(
        self,
        config: dict[str, Any],
        args: Namespace,
    ) -> None:
        """Mutagenese constructor.

        Args:
            config (dict[str, Any]): The configuration of the experiment.
            args (Namespace): The arguments of the program.
        """
        super().__init__(config, args)

        # Mutations specifications
        self.l_m = int(self.mutations_config["l_m"])
        self.mutation_length_distribution = self.mutations_config["length_distribution"]
        self.mutation_names = self.mutations_config["Mutation types"]
        self.mutation_types = [
            MUTATIONS[mutation_type] for mutation_type in self.mutation_names
        ]

        # Genome specifications
        self.homogeneous = self.genome_config["Homogeneous"]
        self.orientation = self.genome_config["Orientation"]

        # Variable specifications
        self.variable = self.mutagenese_config["Variable"]

        self.power_min = int(m.log10(float(self.mutagenese_config["From"])))
        self.power_max = int(m.log10(float(self.mutagenese_config["To"])))
        self.power_step = int(self.mutagenese_config["Step"])

        self.experiment_repetitions = str_to_int(self.mutagenese_config["Iterations"])

        self.results = {mutation: {} for mutation in self.mutation_names}

    def run(
        self,
        only_plot: bool = False,
    ) -> None:
        """Runs the mutagenese experiment.

        Args:
            only_plot (bool, optional): If True, skip the main loop and only plot the results. Assume the experiment was run once. Defaults to False.
        """
        if not only_plot:
            if self.variable == "No variable":
                genome = self.prepare_mutagenese()
                for mutation, name in zip(self.mutation_types, self.mutation_names):
                    print(f"Mutation type: {name}")
                    if name in L_M:
                        # the mutation object has to have an explicit l_m parameter
                        self.results[name][888] = self.loop(
                            mutation(
                                self.l_m, self.mutation_length_distribution, genome
                            )
                        )
                    else:
                        self.results[name][888] = self.loop(
                            mutation(self.mutation_length_distribution, genome)
                        )
            else:
                for power in range(self.power_min, self.power_max + 1, self.power_step):
                    print(f"Experience for {self.variable} = 10^{power}")
                    genome = self.prepare_mutagenese(10**power)
                    for mutation, name in zip(self.mutation_types, self.mutation_names):
                        print(f"Mutation type: {name}")
                        if name in L_M:
                            # the mutation object has to have an explicit l_m parameter
                            self.results[name][power] = self.loop(
                                mutation(
                                    self.l_m, self.mutation_length_distribution, genome
                                )
                            )
                        else:
                            self.results[name][power] = self.loop(
                                mutation(self.mutation_length_distribution, genome)
                            )

            graphics.save_stats(self.save_path, self.results)

        if self.variable != "No variable":
            self.plot_mutagenese()

    def prepare_mutagenese(
        self,
        value: int = 0,
    ) -> Genome:
        """Prepares the genome for the mutagenese experiment.
        Handle the different value of variable parameter.

        Args:
            value (int, optional): Value of the variable (if any). Defaults to 0.

        Returns:
            Genome: The genome object used for the experiment
        """
        # g
        if self.variable == "g":
            g = value
        else:
            g = str_to_int(self.genome_config["g"])

        # z_c
        if self.variable == "z_c":
            z_c = value
        else:
            beta = str_to_int(self.genome_config["beta"])
            if self.genome_config["z_c_auto"]:
                z_c = beta * g
            else:
                z_c = str_to_int(self.genome_config["z_c"])

        # z_nc
        if self.variable == "z_nc":
            z_nc = value
        else:
            alpha = str_to_int(self.genome_config["alpha"])
            if self.genome_config["z_nc_auto"]:
                z_nc = alpha * g
            else:
                z_nc = str_to_int(self.genome_config["z_nc"])

        genome = Genome(g, z_c, z_nc, self.homogeneous, self.orientation)
        return genome

    def loop(
        self,
        mutation: mutations.Mutation,
    ) -> dict[str, float]:
        """Main loop of the mutagenese experiment.

        Args:
            mutation (mutations.Mutation): The mutation virtually applied on the genome

        Returns:
            dict[str, float]: Dictionnary of the statistics of the experiment.
        """
        for _ in tqdm(
            range(self.experiment_repetitions),
            "Experiment progress... ",
            self.experiment_repetitions,
        ):
            if mutation.is_neutral():
                mutation.apply(virtually=True)
        mutation.stats.compute(mutation.theory())
        return mutation.stats.d_stats

    def plot_mutagenese(
        self,
    ) -> None:
        """Plots the result of the experiment"""
        x_value = [
            10**power
            for power in range(self.power_min, self.power_max + 1, self.power_step)
        ]
        for mutation_type in self.mutation_names:
            save_dir = self.save_path / "stats" / mutation_type

            neutral_proportions = []
            neutral_stds = []
            theoretical_proportions = []
            length_means = []
            length_stds = []

            for power in range(self.power_min, self.power_max + 1, self.power_step):

                with open(
                    save_dir / f"{power}.json", "r", encoding="utf8"
                ) as json_file:
                    d_stats = json.load(json_file)

                neutral_proportions.append(d_stats["Neutral mutations proportion"])
                neutral_stds.append(
                    d_stats[
                        "Neutral mutations standard deviation of proportion estimator"
                    ]
                )
                theoretical_proportions.append(d_stats["Neutral probability theory"])
                length_means.append(d_stats["Length mean"])
                length_stds.append(
                    d_stats["Length standard deviation of mean estimator"]
                )

            graphics.plot_mutagenese(
                x_value,
                neutral_proportions,
                neutral_stds,
                save_dir,
                f"Neutral {mutation_type.lower()} proportion",
                self.variable,
                theoretical_proportions,
            )

            graphics.plot_mutagenese(
                x_value,
                length_means,
                length_stds,
                save_dir,
                f"{mutation_type} length mean",
                self.variable,
            )
