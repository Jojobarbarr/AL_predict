import sys
import unittest
from unittest.mock import patch

import numpy as np
from parameterized import parameterized

sys.path.append(".")

from genome import Genome
from mutations import (
    Mutation,
    PointMutation,
    SmallInsertion,
    SmallDeletion,
    Deletion,
    Duplication,
    Inversion,
)


class TestMutation(unittest.TestCase):
    def setUp(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            self.genome = Genome(10, 100, 50)
        self.genome.orientation_list = np.array(
            [1, 1, -1, -1, 1, -1, -1, 1, 1, -1], dtype=np.int_
        )
        self.mutation = Mutation("Uniform")
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "Mutation")

    def test_is_neutral(self):
        previous_count = self.mutation.stats.count
        self.mutation.is_neutral()
        self.assertEqual(self.mutation.stats.count, previous_count + 1)

    def test_apply(self):
        previous_neutral_count = self.mutation.stats.neutral_count
        previous_length_sum = self.mutation.stats.length_sum
        previous_length_square_sum = self.mutation.stats.length_square_sum
        self.mutation.apply()
        self.assertEqual(self.mutation.stats.neutral_count, previous_neutral_count + 1)
        self.assertEqual(
            self.mutation.stats.length_sum, previous_length_sum + self.mutation.length
        )
        self.assertEqual(
            self.mutation.stats.length_square_sum,
            previous_length_square_sum + self.mutation.length**2,
        )


class TestPointMutation(unittest.TestCase):
    def setUp(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            self.genome = Genome(10, 100, 50)
        self.genome.orientation_list = np.array(
            [1, 1, -1, -1, 1, -1, -1, 1, 1, -1], dtype=np.int_
        )
        self.mutation = PointMutation("Uniform")
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "PointMutation")


class TestSmallInsertion(unittest.TestCase):
    def setUp(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            self.genome = Genome(10, 100, 50)
        self.genome.orientation_list = np.array(
            [1, 1, -1, -1, 1, -1, -1, 1, 1, -1], dtype=np.int_
        )
        self.mutation = SmallInsertion(l_m=10, mutation_length_distribution="Uniform")
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "SmallInsertion")

    @parameterized.expand(
        [
            (0, 2, [3, 14, 24, 39, 53, 69, 86, 97, 107, 123]),
            (14, 3, [1, 12, 22, 37, 54, 70, 87, 98, 108, 124]),
            (15, 3, [1, 12, 22, 37, 54, 70, 87, 98, 108, 124]),
            (51, 5, [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]),
        ]
    )
    def test_apply(self, insertion_locus, length, expected_result):
        previous_length = self.genome.length
        with patch("random.randint", side_effect=[insertion_locus, length]):
            self.mutation.apply()
        self.assertListEqual(self.genome.loci.tolist(), expected_result)
        self.assertEqual(self.genome.length, previous_length + length)


class TestDeletion(unittest.TestCase):
    def setUp(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            self.genome = Genome(10, 100, 50)
        self.genome.orientation_list = np.array(
            [1, 1, -1, -1, 1, -1, -1, 1, 1, -1], dtype=np.int_
        )
        self.mutation = Deletion("Uniform")
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "Deletion")

    # [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]
    @parameterized.expand(
        [
            (0, 1, [0, 11, 21, 36, 50, 66, 83, 94, 104, 120]),
            (10, 1, [1, 11, 21, 36, 50, 66, 83, 94, 104, 120]),
            (61, 5, [1, 12, 22, 37, 51, 62, 79, 90, 100, 116]),
            (148, 3, [0, 11, 21, 36, 50, 66, 83, 94, 104, 120]),
        ]
    )
    def test_apply(
        self,
        starting_locus,
        length,
        expected_result,
    ):
        previous_length = self.genome.length
        self.mutation.starting_locus = starting_locus
        self.mutation.length = length
        self.mutation.apply()
        self.assertListEqual(self.genome.loci.tolist(), expected_result)
        self.assertEqual(self.genome.length, previous_length - length)


class TestSmallDeletion(unittest.TestCase):
    def setUp(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            self.genome = Genome(10, 100, 50)
        self.genome.orientation_list = np.array(
            [1, 1, -1, -1, 1, -1, -1, 1, 1, -1], dtype=np.int_
        )
        self.mutation = SmallDeletion(l_m=10, mutation_length_distribution="Uniform")
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "SmallDeletion")


class TestDuplication(unittest.TestCase):
    def setUp(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            self.genome = Genome(10, 100, 50)
        self.genome.orientation_list = np.array(
            [1, 1, -1, -1, 1, -1, -1, 1, 1, -1], dtype=np.int_
        )
        self.mutation = Duplication("Uniform")
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "Duplication")

    @parameterized.expand(
        [
            (0, 2, [3, 14, 24, 39, 53, 69, 86, 97, 107, 123]),
            (1, 1, [2, 13, 23, 38, 52, 68, 85, 96, 106, 122]),
            (21, 5, [1, 12, 22, 37, 51, 72, 89, 100, 110, 126]),
            (32, 30, [1, 12, 22, 37, 51, 67, 84, 125, 135, 151]),
        ]
    )
    def test_apply(self, insertion_locus, length, expected_result):
        previous_length = self.genome.length
        self.mutation.length = length
        with patch("random.randint", side_effect=[insertion_locus]):
            self.mutation.apply()
        self.assertListEqual(self.genome.loci.tolist(), expected_result)
        self.assertEqual(self.genome.length, previous_length + length)


class TestInversion(unittest.TestCase):
    def setUp(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            self.genome = Genome(10, 100, 50)
        self.genome.orientation_list = np.array(
            [1, 1, -1, -1, 1, -1, -1, 1, 1, -1], dtype=np.int_
        )
        self.mutation = Inversion("Uniform")
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "Inversion")

    @parameterized.expand(
        [
            (
                [0, 6],
                [1, 11, 22, 37, 51, 67, 84, 95, 105, 121],
                [1, -1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                [7, 0],
                [2, 12, 23, 37, 51, 67, 84, 95, 105, 121],
                [1, -1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                [0, 8],
                [3, 13, 24, 37, 51, 67, 84, 95, 105, 121],
                [1, -1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                [23, 25],
                [1, 12, 22, 37, 51, 67, 84, 95, 105, 121],
                [1, 1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                [51, 23],
                [1, 12, 22, 37, 51, 67, 87, 103, 113, 124],
                [1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
            ),
        ]
    )
    def test_apply(self, breaking_locus, expected_loci, expected_orientation):
        previous_length = self.genome.length
        with patch("random.sample", return_value=breaking_locus):
            self.mutation.apply()
        self.assertListEqual(self.genome.loci.tolist(), expected_loci)
        self.assertListEqual(
            self.genome.orientation_list.tolist(), expected_orientation
        )
        self.assertEqual(self.genome.length, previous_length)


if __name__ == "__main__":
    unittest.main()
