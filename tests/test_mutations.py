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
        self.mutation = Mutation()
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
        self.mutation = PointMutation()
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
        self.mutation = SmallInsertion(l_m=10)
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "SmallInsertion")

    @parameterized.expand(
        [
            (0, 2, [4, 16, 27, 43, 58, 75, 93, 105, 116, 133]),
            (14, 3, [2, 14, 25, 44, 59, 76, 94, 106, 117, 134]),
            (15, 3, [2, 14, 25, 41, 59, 76, 94, 106, 117, 134]),
            (51, 5, [2, 14, 25, 41, 56, 73, 91, 103, 114, 131]),
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
        self.mutation = Deletion()
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "Deletion")

    @parameterized.expand(
        [
            (0, 2, [0, 12, 23, 39, 54, 71, 89, 101, 112, 129]),
            (1, 1, [1, 13, 24, 40, 55, 72, 90, 102, 113, 130]),
            (68, 5, [2, 14, 25, 41, 56, 68, 86, 98, 109, 126]),
            (148, 3, [1, 13, 24, 40, 55, 72, 90, 102, 113, 130]),
        ]
    )
    def test_apply(self, starting_locus, length, expected_result):
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
        self.mutation = SmallDeletion(l_m=10)
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
        self.mutation = Duplication()
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "Duplication")

    @parameterized.expand(
        [
            (0, 0, 2, True),
            (0, 0, 3, True),
            (0, 0, 11, True),
            (0, 0, 12, False),
            (5, 57, 25, True),
            (5, 57, 26, False),
            (10, 145, 4, True),
            (10, 145, 7, True),
            (10, 145, 16, True),
            (10, 145, 17, False),
        ]
    )
    def test_ending_point_is_ok(
        self, next_promoter_locus_index, starting_locus, length, expected_result
    ):
        self.genome.orientation_list = np.array(
            [-1, 1, -1, -1, 1, -1, -1, 1, 1, -1], dtype=np.int_
        )
        self.mutation.length = length
        self.mutation.starting_locus = starting_locus
        result = self.mutation._ending_point_is_ok(
            self.mutation.starting_locus,
            next_promoter_locus_index,
        )
        self.assertEqual(result, expected_result)

    @parameterized.expand(
        [
            (0, 2, [4, 16, 27, 43, 58, 75, 93, 105, 116, 133]),
            (1, 1, [3, 15, 26, 42, 57, 74, 92, 104, 115, 132]),
            (23, 5, [2, 14, 25, 41, 56, 78, 96, 108, 119, 136]),
            (55, 3, [2, 14, 25, 41, 56, 73, 91, 103, 114, 131]),
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
        self.mutation = Inversion()
        self.mutation.genome = self.genome

    def test_str(self):
        self.assertEqual(str(self.mutation), "Inversion")

    @parameterized.expand(
        [
            (
                [0, 6],
                [0, 12, 25, 41, 56, 73, 91, 103, 114, 131],
                [-1, -1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                [7, 0],
                [1, 13, 25, 41, 56, 73, 91, 103, 114, 131],
                [-1, -1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                [0, 8],
                [0, 11, 23, 41, 56, 73, 91, 103, 114, 131],
                [1, -1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                [23, 25],
                [2, 14, 25, 41, 56, 73, 91, 103, 114, 131],
                [1, 1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                [51, 23],
                [2, 14, 25, 41, 56, 68, 85, 96, 108, 126],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
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
