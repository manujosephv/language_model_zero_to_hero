# Natural Language Toolkit: Language Model Unit Tests
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Manu Joseph <manujosephv@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
"""Samplers for sampling from a language model.
"""

from .api import _random_generator
from tqdm.autonotebook import tqdm as progress
import random
from bisect import bisect
from itertools import accumulate
import numpy as np


def _apply_temperature(distribution, temperature):
    running_sum = 0
    weights = []
    for entry in distribution:
        exp_prob = pow(entry[1], 1/temperature)
        weights.append(exp_prob)
        running_sum += exp_prob
    weights = list(map(lambda x: x / running_sum, weights))
    samples = [entry[0] for entry in distribution]
    return weights, samples


def _pick_random(weights, samples, random_generator):
    cum_weights = list(accumulate(weights))
    total = cum_weights[-1]
    threshold = random_generator.random()
    return samples[bisect(cum_weights, total * threshold)]


def weighted_random_choice(distribution, **kwargs):
    """Like random.choice, but with weights.

        Heavily inspired by python 3.6 `random.choices`.
        """
    temperature = kwargs.get("temperature", 1)
    random_generator = kwargs.get(
        "random_generator", _random_generator(kwargs.get("random_seed", None))
    )
    weights, samples = _apply_temperature(distribution, temperature)
    if sum(weights) > 0:
        return _pick_random(weights, samples, random_generator)
    else:
        eos = kwargs.get("EOS", "</s>")
        return eos


def topk(distribution, **kwargs):
    """implements the topk sampling approach
        """
    temperature = kwargs.get("temperature", 1)
    k = kwargs.get("k", 1)
    random_generator = kwargs.get(
        "random_generator", _random_generator(kwargs.get("random_seed", None))
    )
    #Select top k
    distribution = distribution[:k]
    #Renormalize with temperature
    weights, samples = _apply_temperature(distribution, temperature)
    if sum(weights) > 0:
        return _pick_random(weights, samples, random_generator)
    else:
        eos = kwargs.get("EOS", "</s>")
        return eos

def nucleus_sampling(distribution, **kwargs):
    """implements the nucleus sampling approach
        """
    temperature = kwargs.get("temperature", 1)
    p = kwargs.get("p", 1)
    random_generator = kwargs.get(
        "random_generator", _random_generator(kwargs.get("random_seed", None))
    )
    #Renormalize with temperature =1
    #Useful for stupid backoff type models which doesnt sum up to 1
    weights, samples = _apply_temperature(distribution, 1)
    cum_weights = list(accumulate(weights))
    distribution = distribution[:bisect(cum_weights, p)]
    weights, samples = _apply_temperature(distribution, temperature)
    if sum(weights) > 0:
        return _pick_random(weights, samples, random_generator)
    else:
        eos = kwargs.get("EOS", "</s>")
        return eos


class Hypothesis(object):
    """Defines a hypothesis during beam search."""

    def __init__(self, context, prob, out=None):
        """Hypothesis constructor.
        Args:
            context: start tokens for decoding.
            log_prob: log prob of the start tokens, usually 1.
            out: tokens generated for output.
        """
        self.context = context
        self.prob = prob
        self.out = out
        # Used in Diverse N-Best beam Search
        self._sibling_rank = None
        # Used in DiverseBeamSearch
        self._hamming_penalty = 0

    def extend(self, token, prob):
        """Extend the hypothesis with result from latest step.
        Args:
            token: latest token from decoding.
            log_prob: log prob of the latest decoded tokens.
            new_state: decoder output state. Fed to the decoder for next step.
        Returns:
            New Hypothesis with the results from latest step.
        """
        if hasattr(self, "out"):
            if self.out is None:
                new_out = (token,)
            else:
                new_out = self.out + (token,)
        else:
            new_out = (token,)
        return Hypothesis(self.context + (token,), self.prob * prob, new_out)

    @property
    def latest_token(self):
        return self.context[-1]

    def __str__(self):
        return "Hypothesis(log prob = %.4f, Context = %s)" % (self.prob, self.context)
    
    def __repr__(self):
        return "Hypothesis(log prob = %.4f, Context = %s)" % (self.prob, self.context)


class BeamSearch:
    def __init__(
        self,
        model,
        generate_hypotheses_func=None,
        beam_width=3,
        verbose=True,
        debug_level=0,
        EOS=["</s>"],
        prune_width=None,
        normalize_by_length=True,
        alpha_length_norm=0.75,
    ):
        self.model = model
        if generate_hypotheses_func is not None:
            self._generate_hypotheses = generate_hypotheses_func
        self.beam_width = beam_width
        self.verbose = verbose
        self.debug_level = debug_level
        self.EOS = EOS
        self.prune_width = beam_width if prune_width is None else prune_width
        self.normalize_by_length = normalize_by_length
        self.alpha_length_norm = alpha_length_norm

    def _generate_hypotheses(self, text_seed, beam_width, **kwargs):
        distribution = self.model.context_probabilities(text_seed)
        distribution = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        # If there are multiple words with same probability, we choose
        # one at random
        ret_beam = []
        while len(ret_beam) < beam_width:
            top_samples = [
                (sample, weight)
                for sample, weight in distribution
                if weight == distribution[0][1]
            ]
            r = int(random.uniform(0, len(top_samples) - 1))
            token = top_samples[r]
            distribution.remove(token)
            ret_beam.append(token)
        return ret_beam

    def _best_hyps(self, hyps, normalize_by_length=False):
        """Sort the hyps based on log probs and length.
        Args:
        hyps: A list of hypothesis.
        Returns:
        hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        # This length normalization is only effective for the final results.
        if normalize_by_length:
            return sorted(
                hyps,
                key=lambda h: h.prob / len(h.context) ** self.alpha_length_norm,
                reverse=True,
            )
        else:
            return sorted(hyps, key=lambda h: h.prob, reverse=True)

    def _setup_beam(self, text_seed, beam_width):
        starting_beam = self._generate_hypotheses(text_seed, beam_width)
        # Replicate the initial states K times for the first step.
        hyps = [Hypothesis(text_seed, 1)] * beam_width
        if self.debug_level > 10:
            print("Initial Hypothesis")
            for h in hyps:
                print(h)
        # The first step takes the best K results from first hyps. Following
        # steps take the best K results from K*K hyps.
        for i, (hyp, (word, prob)) in enumerate(zip(hyps, starting_beam)):
            hyps[i] = hyp.extend(token=word, prob=prob)
        if self.debug_level > 0:
            print("Hypothesis Step 1")
            for h in hyps:
                print(h)
        return hyps

    def _beam_extend(self, hyps, beam_width):
        # Extend each hypothesis.
        all_hyps = []
        for i in range(len(hyps)):
            h = hyps[i]
            most_common = self._generate_hypotheses(h.context, beam_width)
            # print ("Most common for Hyp {}: {}".format(i, most_common[:10]))
            for new_word, new_prob in most_common:
                all_hyps.append(h.extend(new_word, new_prob))
        return all_hyps

    def _beam_filter(self, all_hyps, results, prune_width):
        hyps = []
        # No need to normalize for length here because all of the same length
        for h in self._best_hyps(all_hyps, normalize_by_length=False):
            if h.latest_token in self.EOS:
                # Pull the hypothesis off the beam if the end token is reached.
                results.append(h)
            else:
                # Otherwise continue to the extend the hypothesis.
                hyps.append(h)
            if len(hyps) == prune_width or len(results) == prune_width:
                break
        return hyps, results

    def generate(self, text_seed, num_words):
        text_seed = tuple(text_seed)
        if self.verbose:
            pbar = progress(total=num_words, desc="Generating words")
        hyps = self._setup_beam(text_seed, self.beam_width)
        results = []
        steps = 1
        if self.verbose:
            pbar.update(1)
        while steps < num_words and len(results) < self.beam_width:
            # Extend each hypothesis.
            all_hyps = self._beam_extend(hyps, self.beam_width)
            # Filter and collect any hypotheses that have the end token.
            hyps, results = self._beam_filter(all_hyps, results, self.prune_width)
            if self.debug_level > 10:
                print(f"Hypothesis Step {steps+1}")
                for h in hyps:
                    print(h)
            steps += 1
            if self.verbose:
                pbar.update(1)
            if steps == num_words:
                results.extend(hyps)
            if self.debug_level > 0:
                print(f"Results Step {steps+1}")
                for h in results:
                    print(h)
        if self.verbose:
            pbar.close()
        results = self._best_hyps(results, self.normalize_by_length)
        if self.debug_level > 0:
            print(f"Final Sorted Results")
            for h in results:
                print(h)
        return results[0].out


from collections import defaultdict, Counter
import scipy.stats as ss


class DiverseNbestBeamSearch(BeamSearch):
    def __init__(
        self,
        model,
        generate_hypotheses_func=None,
        beam_width=3,
        verbose=True,
        debug_level=0,
        EOS=["</s>"],
        prune_width=None,
        normalize_by_length=True,
        alpha_length_norm=0.75,
        diversity_factor=1,
    ):
        super().__init__(
            model,
            generate_hypotheses_func=generate_hypotheses_func,
            beam_width=beam_width,
            verbose=verbose,
            debug_level=debug_level,
            EOS=EOS,
            prune_width=prune_width,
            normalize_by_length=normalize_by_length,
            alpha_length_norm=alpha_length_norm,
        )
        self.diversity_factor = diversity_factor

    @classmethod
    def _calculate_sibling_rank(cls, hyps):
        sibling_grps = defaultdict(list)
        # map(lambda hyp: sibling_grps[hyp.context].append(hyp), hyps)
        for hyp in hyps:
            sibling_grps[hyp.context[:-1]].append(hyp)

        hyps = []
        for parent, siblings in sibling_grps.items():
            grp_rank = ss.rankdata([-hyp.prob for hyp in siblings])
            for hyp, rank in zip(siblings, grp_rank):
                hyp._sibling_rank = rank
            hyps += siblings
        return hyps

    def _best_hyps(self, hyps, normalize_by_length=False):
        """Sort the hyps based on log probs and length.
        Args:
        hyps: A list of hypothesis.
        Returns:
        hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        # This length normalization is only effective for the final results.

        if normalize_by_length:
            return sorted(hyps, key=lambda h: h.prob / len(h.context), reverse=True)
        else:
            hyps = self._calculate_sibling_rank(hyps)
            return sorted(
                hyps,
                key=lambda h: h.prob - self.diversity_factor * h._sibling_rank,
                reverse=True,
            )


class DiverseBeamSearch(BeamSearch):
    def __init__(
        self,
        model,
        generate_hypotheses_func=None,
        beam_width=3,
        num_groups=None,
        verbose=True,
        debug_level=0,
        EOS=["</s>"],
        prune_width=None,
        normalize_by_length=True,
        alpha_length_norm=0.75,
        diversity_strength=0.8
    ):
        super().__init__(
            model,
            generate_hypotheses_func=generate_hypotheses_func,
            beam_width=beam_width,
            verbose=verbose,
            debug_level=debug_level,
            EOS=EOS,
            prune_width=prune_width,
            normalize_by_length=normalize_by_length,
            alpha_length_norm=alpha_length_norm,
        )
        self.num_groups = beam_width if num_groups is None else num_groups
        if beam_width % num_groups != 0:
            raise ValueError(
                "DiverseBeamSearch requires --beam to be divisible by the number of groups"
            )
        self.diversity_strength = diversity_strength

    def _best_hyps(self, hyps, normalize_by_length=False):
        """Sort the hyps based on log probs and length.
        Args:
        hyps: A list of hypothesis.
        Returns:
        hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        # This length normalization is only effective for the final results.

        if normalize_by_length:
            return sorted(hyps, key=lambda h: h.prob / len(h.context), reverse=True)
        else:
            return sorted(
                hyps,
                key=lambda h: h.prob - self.diversity_strength * h._hamming_penalty,
                reverse=True,
            )

    def generate(self, text_seed, num_words):
        text_seed = tuple(text_seed)
        if self.verbose:
            pbar = progress(total=num_words, desc="Generating words")
        b_prime = self.beam_width // self.num_groups
        hyps = self._setup_beam(text_seed, self.beam_width)
        hyps_group_split = np.array_split(hyps, self.num_groups)
        grouped_beam_dict = dict()
        for g in range(self.num_groups):
            grouped_beam_dict[g] = {"hyps": hyps_group_split[g], "results": []}
        steps = 1
        if self.verbose:
            pbar.update(1)
        while steps < num_words and any(
            len(grouped_beam_dict[g]["results"]) < b_prime
            for g in grouped_beam_dict.keys()
        ):
            inter_group_penalty = Counter()
            for g in range(self.num_groups):
                # Extend each hypothesis.
                all_hyps = self._beam_extend(grouped_beam_dict[g]["hyps"], b_prime)
                #Applying the hamming penalty. For the first group, penalty would be zero
                for h in all_hyps:
                    h._hamming_penalty = inter_group_penalty[h.latest_token]
                # Filter and collect any hypotheses that have the end token.
                hyps, results = self._beam_filter(
                    all_hyps, grouped_beam_dict[g]["results"], self.prune_width
                )
                #Update the counter for hamming penalty
                inter_group_penalty.update([h.latest_token for h in all_hyps])
                if self.debug_level > 10:
                    print(f"Hypothesis Group {g+1} Step {steps+1}")
                    for h in hyps:
                        print(h)
                if self.debug_level > 0:
                    print(f"Results Group {g+1} Step {steps+1}")
                    for h in results:
                        print(h)
                grouped_beam_dict[g]["hyps"] = hyps
                grouped_beam_dict[g]["results"] = results
            steps += 1
            if steps==num_words:
                # Adding the open hypothesis to results
                for g in grouped_beam_dict.keys():
                    grouped_beam_dict[g]['results'].extend(grouped_beam_dict[g]['hyps'])
            if self.verbose:
                pbar.update(1)
        if self.verbose:
            pbar.close()
        
        results = []
        for g in grouped_beam_dict.keys():
            results += grouped_beam_dict[g]['results']
        results = self._best_hyps(results, self.normalize_by_length)
        if self.debug_level > 0:
            print(f"Final Sorted Results")
            for h in results:
                print(h)
        return results[0].out
