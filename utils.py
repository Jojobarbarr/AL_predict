import numpy as np
import mutations


QUANTILES = np.array([0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 1.0], dtype=float)

EPSILON = 1e-9

MUTATIONS = {
    "Point mutation": mutations.PointMutation,
    "Small insertion": mutations.SmallInsertion,
    "Small deletion": mutations.SmallDeletion,
    "Deletion": mutations.Deletion,
    "Duplication": mutations.Duplication,
    "Inversion": mutations.Inversion,
}

L_M = {
    "Small insertion",
    "Small deletion",
}


def str_to_int(string: str) -> int:
    return int(float(string))


def str_to_bool(string: str) -> bool:
    return string.lower() in {"true", "on", "1", "yes"}
