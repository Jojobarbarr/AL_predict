import unittest

from genome import Genome
from mutations import Mutation


class TestMutation(unittest.TestCase):
    def setUp(self):
        self.genome = Genome(1, 1, 1)
        self.mutation = Mutation("Mutation")
    
    def test_str(self):
        self.assertEqual(str(self.mutation), "Mutation")

    def test_is_neutral(self):
        previous_count = self.mutation.stats.count
        self.mutation.is_neutral(self.genome)
        self.assertEqual(self.mutation.stats.count, previous_count + 1)

    def test_apply(self):
        self.mutation.apply(self.genome)
        previous_neutral_count = self.mutation.stats.neutral_count
        previous_length_sum = self.mutation.stats.length_sum
        previous_length_square_sum = self.mutation.stats.length_square_sum
        self.assertEqual(self.mutation.stats.neutral_count, previous_neutral_count + 1)
        self.assertEqual(self.mutation.stats.length_sum, previous_length_sum + self.mutation.length)
        self.assertEqual(self.mutation.stats.length_square_sum, previous_length_square_sum + self.mutation.length ** 2)

    def test_map_local_to_absolute_locus(self):
        self.genome.set_genome(4, 6, 2, [1, 6], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0])
        self.mutation.insertion_locus = 1
        self.mutation.map_local_to_absolute_locus(self.genome)
        self.assertEqual(self.mutation.insertion_locus, 6)

    def test_ending_point_is_ok(self):
        self.genome.set_genome(4, 6, 2, [1, 6], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0])
        self.mutation.starting_point = 2
        next_promoter_locus_index = 1
        result = self.mutation.ending_point_is_ok(self.genome, next_promoter_locus_index)
        self.assertTrue(result)

    def test_theory(self):
        result = self.mutation.theory(self.genome)
        self.assertEqual(result, (0, 0))

if __name__ == "__main__":
    unittest.main()