import numpy as np
import pickle as pkl
from pathlib import Path

from genome import Genome

if __name__ == "__main__":
    nbr_replicas = 10
    population = 1e3
    save_path = Path("/wild_type/0_0/median_individual.pkl")
    z_ncs = np.array((nbr_replicas * population), dtype=Genome)
    for replica in range(nbr_replicas):
        with open(
            f"/results/simulation/0_0/{replica + 1}/populations/final", "rb"
        ) as pkl_file:
            genomes = pkl.load(pkl_file)
            z_ncs[replica : population * (replica + 1)] = np.array(
                [genome.z_nc for genome in genomes], dtype=Genome
            )
            median_index = np.argsort(z_ncs)[len(z_ncs) // 2]
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as pkl_file:
        pkl.dump(z_ncs[median_index], pkl_file)
