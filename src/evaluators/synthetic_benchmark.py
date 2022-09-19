import src.utils.gen_utils
from src.evaluators import Evaluator
from src.utils.gen_utils import (
    generate_correct_substitution_distributions,
    generate_sequences,
    mutate_sequence_correct_probabilities,
)


class SyntheticBenchmark(Evaluator):
    def __init__(
        self, num_targets, num_queries, sub_percent, sequence_length, indels=False
    ):

        self.num_targets = num_targets
        self.num_queries = num_queries
        self.sub_percent = sub_percent
        self.sequence_length = sequence_length
        self.indels = indels
        self.aa_dist = src.utils.gen_utils.amino_distribution
        self.sub_dists = generate_correct_substitution_distributions()

    def _create_target_and_query_dbs(self, model_class):

        target_sequences = generate_sequences(
            self.num_targets, self.sequence_length, self.aa_dist
        )
        # now go through and mutate a random selection of target sequences
        random_query_idx = torch.randperm(target_sequences.shape[0])[: self.num_queries]
        query_templates = target_sequences[random_query_idx]
        # mutate the query templates
        queries = torch.zeros_like(query_templates)
        for i, seq in enumerate(query_templates):
            mutated = mutate_sequence_correct_probabilities(
                sequence=seq,
                indels=self.indels,
                substitutions=int(self.sequence_length * self.sub_percent),
                sub_distributions=self.sub_dists,
                aa_dist=utils.amino_distribution,
            )
            queries[i] = mutated
        # compute embeddings
        # forget about batching
        target_embeddings = []

    def evaluate(self, model_class):
        self._create_target_and_query_dbs(model_class)
        pass
