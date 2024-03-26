import numpy as np
import pickle as pkl
from pathlib import Path

from genome import Genome

if __name__ == "__main__":
    nbr_replicas = 5
    population = int(1e3)
    save_path = Path("wild_type/0_0/")
    genomes = np.empty((nbr_replicas * population), dtype=Genome)

    for replica in range(nbr_replicas):
        print(f"Replica {replica + 1}")
        with open(
            f"results/simulation/0_0/{replica + 1}/populations/final", "rb"
        ) as pkl_file:
            genomes_loaded = pkl.load(pkl_file)
            genomes[replica * population : population * (replica + 1)] = np.array(
                [genome for genome in genomes_loaded], dtype=Genome
            )
    z_ncs = np.array([genome.z_nc for genome in genomes], dtype=int)
    median_index = np.argsort(z_ncs)[len(z_ncs) // 2]
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "median_individual.pkl", "wb") as pkl_file:
        pkl.dump(genomes[median_index], pkl_file)
