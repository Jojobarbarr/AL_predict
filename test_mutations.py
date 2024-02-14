import numpy as np

import mutations
from genome import Genome

def test_small_insertions():
    explicit_genome = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    implicit_genome = np.array([1, 6])

    genome = Genome(1, 1, 1)
    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))

    small_insertion = mutations.SmallInsertion(1, genome, 3, DEBUG=True)

    print(f"{'*' * 16}\nTEST 1")
    print(genome)
    small_insertion.test(0, 1)
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 2")
    print(genome)
    small_insertion.test(1, 1)
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 3")
    print(genome)
    small_insertion.test(2, 6) 
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 4")
    print(genome)
    small_insertion.test(7, 1)
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 5")
    print(genome)
    small_insertion.test(4, 6)
    print(genome)


def test_small_deletions():
    explicit_genome = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    implicit_genome = np.array([1, 6])

    genome = Genome(1, 1, 1)
    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))

    small_deletion = mutations.SmallDeletion(1, genome, 3, DEBUG=True)

    print(f"{'*' * 16}\nTEST 1")
    print(genome)
    small_deletion.test(0, 0)
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 2")
    print(genome)
    small_deletion.test(1, 3)
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 3")
    print(genome)
    small_deletion.test(4, 8) 
    print(genome)

def test_deletions():
    explicit_genome = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    implicit_genome = np.array([1, 6])

    genome = Genome(1, 1, 1)
    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))

    deletion = mutations.Deletion(1, genome, 3, DEBUG=True)

    print(f"{'*' * 16}\nTEST 1")
    print(genome)
    deletion.test(0, 0)
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 2")
    print(genome)
    deletion.test(1, 3)
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 3")
    print(genome)
    deletion.test(4, 8) 
    print(genome)

def test_duplications():
    explicit_genome = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0])
    implicit_genome = np.array([1, 5])

    genome = Genome(1, 1, 1)
    genome.set_genome(4, 5, 2, np.copy(implicit_genome), np.copy(explicit_genome))

    duplication = mutations.Duplication(1, genome, 3, DEBUG=True)

    print(f"{'*' * 16}\nTEST 1")
    print(genome)
    duplication.test(1, 2, 2, 5)
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 2")
    print(genome)
    duplication.test(4, 6, 4, 5)
    print(genome)

    genome.set_genome(4, 6, 2, np.copy(implicit_genome), np.copy(explicit_genome))
    print(f"{'*' * 16}\nTEST 3")
    print(genome)
    duplication.test(0, 0, 0, 1) 
    print(genome)

    explicit_genome = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0,])
    implicit_genome = np.array([1, 8, 14])

    genome = Genome(1, 1, 1)
    genome.set_genome(12, 7, 3, np.copy(implicit_genome), np.copy(explicit_genome))
    duplication = mutations.Duplication(1, genome, 3, DEBUG=True)

    print(f"{'*' * 16}\nTEST 4")
    print(genome)
    duplication.test(6, 7, 8, 14) 
    print(genome)
if __name__ == "__main__":
    # test_small_insertions()
    # test_small_deletions()
    # test_deletions()
    test_duplications()