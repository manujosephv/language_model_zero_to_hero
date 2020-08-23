# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
#         Manu Joseph <manujosephv@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
"""Language Models"""

from lm.api import LanguageModel, Smoothing
from lm.smoothing import KneserNey, WittenBell, AbsoluteDiscounting, SimpleLinear


class MLE(LanguageModel):
    """Class for providing MLE ngram model scores.

    Inherits initialization from BaseNgramModel.
    """

    def unmasked_score(self, word, context=None):
        """Returns the MLE score for a word given a context.

        Args:
        - word is expcected to be a string
        - context is expected to be something reasonably convertible to a tuple
        """
        return self.context_counts(context).freq(word)


class Lidstone(LanguageModel):
    """Provides Lidstone-smoothed scores.

    In addition to initialization arguments from BaseNgramModel also requires
    a number by which to increase the counts, gamma.
    """

    def __init__(self, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def unmasked_score(self, word, context=None):
        """Add-one smoothing: Lidstone or Laplace.

        To see what kind, look at `gamma` attribute on the class.

        """
        counts = self.context_counts(context)
        word_count = counts[word]
        norm_count = counts.N()
        return (word_count + self.gamma) / (norm_count + len(self.vocab) * self.gamma)


class Laplace(Lidstone):
    """Implements Laplace (add one) smoothing.

    Initialization identical to BaseNgramModel because gamma is always 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class StupidBackoff(LanguageModel):
    """Provides StupidBackoff scores.

    In addition to initialization arguments from BaseNgramModel also requires
    a parameter alpha with which we scale the lower order probabilities.
    """

    def __init__(self, alpha=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def unmasked_score(self, word, context=None):
        """Stupid Backoff.
        """
        if not context:
            # Base recursion
            return self.counts.unigrams.freq(word)
        counts = self.context_counts(context)
        word_count = counts[word]
        norm_count = counts.N()
        if word_count > 0:
            return word_count / norm_count
        else:
            return self.alpha * self.unmasked_score(word, context[1:])


class InterpolatedLanguageModel(LanguageModel):
    """Logic common to all interpolated language models.

    The idea to abstract this comes from Chen & Goodman 1995.
    Do not instantiate this class directly!
    """

    def __init__(self, smoothing_cls, order, **kwargs):
        assert issubclass(smoothing_cls, Smoothing)
        params = kwargs.pop("params", {})
        super().__init__(order, **kwargs)
        self.estimator = smoothing_cls(self.vocab, self.counts, **params)

    def unmasked_score(self, word, context=None):
        if not context:
            # The base recursion case: no context, we only have a unigram.
            if self.estimator._recursion_level is not None:
                assert self.estimator._recursion_level == 0
            unigram_score = self.estimator.unigram_score(word)
            # Resetting recursion counters and flags
            self.estimator._is_top_recursion = True
            self.estimator._recursion_level = None
            return unigram_score
        elif self.estimator._recursion_level is None:
            self.estimator._recursion_level = len(context)
        if not self.counts[context]:
            # It can also happen that we have no data for this context.
            # In that case we defer to the lower-order ngram.
            # This is the same as setting alpha to 0 and gamma to 1.
            alpha, gamma = 0, 1
        else:
            alpha, gamma = self.estimator.alpha_gamma(word, context)
        self.estimator._recursion_level += -1
        self.estimator._is_top_recursion = False
        return alpha + gamma * self.unmasked_score(word, context[1:])


class WittenBellInterpolated(InterpolatedLanguageModel):
    """Interpolated version of Witten-Bell smoothing."""

    def __init__(self, order, **kwargs):
        super().__init__(WittenBell, order, **kwargs)


class AbsoluteDiscountingInterpolated(InterpolatedLanguageModel):
    """Interpolated version of Witten-Bell smoothing."""

    def __init__(self, order, discount=0.75, **kwargs):
        super().__init__(
            AbsoluteDiscounting, order, params={"discount": discount}, **kwargs
        )


class KneserNeyInterpolated(InterpolatedLanguageModel):
    """Interpolated version of Kneser-Ney smoothing."""

    def __init__(self, order, discount=0.75, **kwargs):
        assert (discount <= 1) and (
            discount >= 0
        ), "Discount should be between 0 and 1 for Kneser-Ney probabilities to sum up to unity"
        super().__init__(KneserNey, order, params={"discount": discount}, **kwargs)


class SimpleLinearInterpolation(InterpolatedLanguageModel):
    """Interpolated version of Witten-Bell smoothing."""

    def __init__(self, order, lambda_weights, **kwargs):
        assert isinstance(
            lambda_weights, dict
        ), "lambda_weights should be a dictionary with recursion level as key and weights as value"
        assert (
            len(lambda_weights) == order
        ), "lambda should be list of length equal to order"
        assert sum(lambda_weights.values()) == 1, "lambda weights should sum upto 1"
        super().__init__(
            SimpleLinear, order, params={"lambda_weights": lambda_weights}, **kwargs
        )