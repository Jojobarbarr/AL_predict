import sys
import unittest
from unittest.mock import patch

import numpy as np
from parameterized import parameterized

sys.path.append(".")

from genome import Genome


class TestGenome(unittest.TestCase):
    def test_wild_genome(self):
        genome = Genome(2, 6, 4)
        self.assertEqual(genome.z_c, 6)
        self.assertEqual(genome.z_nc, 4)
        self.assertEqual(genome.length, 10)
        self.assertEqual(genome.g, 2)
        self.assertFalse(genome.homogeneous)
        self.assertFalse(genome.orientation)
        self.assertEqual(genome.gene_length, 3)
        self.assertEqual(len(genome.loci), 2)
        self.assertEqual(len(genome.orientation_list), 2)
        self.assertEqual(len(genome.loci_interval), 2)

    def test_homogeneous_genome(self):
        genome = Genome(2, 6, 4, homogeneous=True)
        self.assertEqual(genome.z_c, 6)
        self.assertEqual(genome.z_nc, 4)
        self.assertEqual(genome.length, 10)
        self.assertEqual(genome.g, 2)
        self.assertTrue(genome.homogeneous)
        self.assertFalse(genome.orientation)
        self.assertEqual(genome.gene_length, 3)
        self.assertEqual(len(genome.loci), 2)
        self.assertListEqual(genome.loci.tolist(), [0, 5])
        self.assertEqual(len(genome.orientation_list), 2)
        self.assertEqual(len(genome.loci_interval), 2)
        self.assertListEqual(genome.loci_interval.tolist(), [2, 2])

    def test_oriented_genome(self):
        genome = Genome(2, 6, 4, homogeneous=True, orientation=True)
        self.assertEqual(genome.z_c, 6)
        self.assertEqual(genome.z_nc, 4)
        self.assertEqual(genome.length, 10)
        self.assertEqual(genome.g, 2)
        self.assertTrue(genome.homogeneous)
        self.assertTrue(genome.orientation)
        self.assertEqual(genome.gene_length, 3)
        self.assertEqual(len(genome.loci), 2)
        self.assertListEqual(genome.loci.tolist(), [0, 5])
        self.assertEqual(len(genome.orientation_list), 2)
        self.assertListEqual(genome.orientation_list.tolist(), [1, 1])
        self.assertEqual(len(genome.loci_interval), 2)
        self.assertListEqual(genome.loci_interval.tolist(), [2, 2])
        self.assertEqual(genome.max_length_neutral, 4)

    def test_clone(self):
        genome = Genome(2, 6, 4, homogeneous=True, orientation=True)
        genome_clone = genome.clone()
        self.assertEqual(genome.z_c, genome_clone.z_c)
        self.assertEqual(genome.z_nc, genome_clone.z_nc)
        self.assertEqual(genome.length, genome_clone.length)
        self.assertEqual(genome.g, genome_clone.g)
        self.assertEqual(genome.homogeneous, genome_clone.homogeneous)
        self.assertEqual(genome.orientation, genome_clone.orientation)
        self.assertEqual(genome.gene_length, genome_clone.gene_length)
        self.assertListEqual(genome.loci.tolist(), genome_clone.loci.tolist())
        self.assertListEqual(
            genome.orientation_list.tolist(), genome_clone.orientation_list.tolist()
        )
        self.assertListEqual(
            genome.loci_interval.tolist(), genome_clone.loci_interval.tolist()
        )
        self.assertEqual(genome.max_length_neutral, genome_clone.max_length_neutral)

    def test_init_genome(self):
        with patch("random.sample", return_value=[2, 3]):
            genome = Genome(2, 6, 4)
        self.assertListEqual(genome.loci.tolist(), [2, 5])

    def test_compute_intervals(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 50)
        self.assertListEqual(
            genome.loci.tolist(), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]
        )
        genome.compute_intervals()
        self.assertListEqual(
            genome.loci_interval.tolist(), [1, 0, 5, 4, 6, 7, 1, 0, 6, 20]
        )

    @parameterized.expand(
        [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 2),
            (48, 10),
        ]
    )
    def test_insertion_binary_search(self, input_value, expected_result):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 50)
        self.assertListEqual(
            genome.loci.tolist(), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]
        )
        result = genome.insertion_binary_search(input_value)
        self.assertEqual(result, expected_result)

    @parameterized.expand(
        [
            (0, 0),
            (1, 1),
            (2, 3),
            (16, 5),
            (41, 10),
        ]
    )
    def test_deletion_binary_search(self, input_value, expected_result):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 50)
        result = genome.deletion_binary_search(input_value)
        self.assertEqual(result, expected_result)

    @parameterized.expand(
        [
            (0, 0),
            (1, 0),
            (2, 1),
            (52, 5),
            (133, 10),
        ]
    )
    def test_duplication_binary_search(self, input_value, expected_result):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 50)
            self.assertListEqual(
                genome.loci.tolist(), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]
            )
            result = genome.duplication_binary_search(input_value)
            self.assertEqual(result, expected_result)

    @parameterized.expand(
        [
            ((0, 1), [2, 13, 23, 38, 52, 68, 85, 96, 106, 122]),
            ((0, 4), [5, 16, 26, 41, 55, 71, 88, 99, 109, 125]),
            ((3, 2), [1, 12, 22, 39, 53, 69, 86, 97, 107, 123]),
            ((10, 3), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]),
        ]
    )
    def test_insert(self, input_values, expected_result):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 50)
        self.assertListEqual(
            genome.loci.tolist(), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]
        )
        previous_length = genome.length
        genome.insert(*input_values)
        self.assertListEqual(genome.loci.tolist(), expected_result)
        self.assertEqual(genome.length, previous_length + input_values[1])

    @parameterized.expand(
        [
            (
                (0, 1),
                [1, 12, 22, 37, 51, 67, 84, 95, 105, 121],
                [1, 1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                (1, 10),
                [1, 12, 22, 37, 51, 67, 84, 95, 105, 121],
                [-1, 1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                (1, 11),
                [2, 12, 22, 37, 51, 67, 84, 95, 105, 121],
                [-1, 1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                (22, 97),
                [1, 12, 26, 36, 47, 64, 80, 94, 109, 121],
                [1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
            ),
            (
                (132, 3),
                [1, 12, 22, 37, 51, 67, 84, 95, 105, 121],
                [1, 1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
            (
                (77, 7),
                [1, 12, 22, 37, 51, 67, 84, 95, 105, 121],
                [1, 1, -1, -1, 1, -1, -1, 1, 1, -1],
            ),
        ]
    )
    def test_inverse(
        self, input_values, expected_result_loci, expected_result_orientation
    ):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 50)
        genome.orientation_list = np.array(
            [1, 1, -1, -1, 1, -1, -1, 1, 1, -1], dtype=np.int_
        )
        self.assertListEqual(
            genome.loci.tolist(), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]
        )
        genome.inverse(*input_values)
        self.assertListEqual(genome.loci.tolist(), expected_result_loci)
        self.assertListEqual(
            genome.orientation_list.tolist(), expected_result_orientation
        )

    @parameterized.expand(
        [
            ((0, 1), [0, 11, 21, 36, 50, 66, 83, 94, 104, 120]),
            ((48, 2), [1, 12, 22, 37, 49, 65, 82, 93, 103, 119]),
            ((132, 3), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]),
        ]
    )
    def test_delete(self, input_values, expected_result):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 50)
        self.assertListEqual(
            genome.loci.tolist(), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]
        )
        previous_length = genome.length
        genome.delete(*input_values)
        self.assertListEqual(genome.loci.tolist(), expected_result)
        self.assertEqual(genome.length, previous_length - input_values[1])

    def test_blend(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 50)
        self.assertListEqual(
            genome.loci.tolist(), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]
        )
        genome.blend()
        self.assertListEqual(
            genome.loci.tolist(), [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
        )

    def test_blend_not_fully_homogeneous(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 53)
        self.assertListEqual(
            genome.loci.tolist(), [1, 12, 22, 37, 51, 67, 84, 95, 105, 121]
        )
        with patch("random.sample", return_value=[0, 1, 9]):
            genome.blend()
        self.assertListEqual(
            genome.loci.tolist(), [0, 16, 32, 47, 62, 77, 92, 107, 122, 137]
        )
        self.assertEqual(genome.length, 153)

    def test_update_features(self):
        with patch("random.sample", return_value=[1, 3, 4, 10, 15, 22, 30, 32, 33, 40]):
            genome = Genome(10, 100, 50, homogeneous=True, orientation=True)
        self.assertListEqual(
            genome.loci.tolist(), [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
        )
        self.assertEqual(genome.max_length_neutral, 14)
        genome.insert(1, 4)
        self.assertEqual(genome.max_length_neutral, 18)
        genome.insert(10, 7)
        self.assertEqual(genome.max_length_neutral, 21)
        genome.blend()
        self.assertEqual(genome.max_length_neutral, 16)


if __name__ == "__main__":
    unittest.main()
