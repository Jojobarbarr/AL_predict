from tqdm import tqdm

import mutations

def experiment(mutation_type: mutations.Mutation, experience_number: int):
    print(f"Experiment on", mutation_type)
    for _ in tqdm(range(experience_number), "Experiment progress... ", experience_number):
        if mutation_type.is_neutral():
            mutation_type.apply(virtually=True)


if __name__ == "__main__":
    pass
