# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2020 NLTK Project
# Authors: Ilia Kurenkov <ilia.kurenkov@gmail.com>
#          Manu Joseph <manujosephv@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
"""Language Model Interface."""

import random
from abc import ABCMeta, abstractmethod
from functools import partial
import sys
from nltk.lm.counter import NgramCounter
from nltk.lm.util import log_base2
from nltk.lm.vocabulary import Vocabulary

from collections import defaultdict
from tqdm.autonotebook import tqdm as progress
from sys import getsizeof

def _mean(items):
    """Return average (aka mean) for sequence of items."""
    return sum(items) / len(items)


def _random_generator(seed_or_generator):
    if isinstance(seed_or_generator, random.Random):
        return seed_or_generator
    return random.Random(seed_or_generator)


def greedy_decoding(distribution, **kwargs):
    weights = [entry[1] for entry in distribution]
    if sum(weights) > 0:
        # If there are multiple words with same probability, we choose
        # one at random
        top_samples = [
            sample for sample, weight in distribution if weight == weights[0]
        ]
        r = int(random.uniform(0, len(top_samples) - 1))
        return top_samples[r]
    else:
        eos = kwargs.get("EOS", "</s>")
        return eos

# Heavily inspired by https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
# class progress:
#     SPINNER = itertools.cycle(["-", "/", "|", "\\"])

#     def __init__(self, iterator, total=None, desc=""):
#         super().__init__()
#         try:
#             total = len(iterator)
#         except TypeError:
#             pass
#         self.iterator = iter(iterator)
#         self.total = total
#         self.bar_len = 60
#         self.count = 0
#         self.desc = desc
        

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.total:
#             filled_len = int(round(self.bar_len * self.count / float(self.total)))
#             percents = round(100.0 * self.count / float(self.total), 1)
#             bar = "=" * filled_len + "-" * (self.bar_len - filled_len)
#         else:
#             percents = f"{self.count}/-"
#             bar = "=" * self.bar_len

#         sys.stdout.write(f"[{bar}] {percents}%  {self.desc} {next(self.SPINNER)}\r")
#         sys.stdout.flush()
#         self.count += 1
#         # As suggested by Rom Ruben
#         # (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
#         return next(self.iterator)

#     def __del__(self):
#         sys.stdout.write("\n")
#         sys.stdout.flush()


class Smoothing(metaclass=ABCMeta):
    """Ngram Smoothing Interface

    Implements Chen & Goodman 1995's idea that all smoothing algorithms have
    certain features in common. This should ideally allow smoothing algorithms to
    work both with Backoff and Interpolation.
    """

    def __init__(self, vocabulary, counter):
        """
        :param vocabulary: The Ngram vocabulary object.
        :type vocabulary: nltk.lm.vocab.Vocabulary
        :param counter: The counts of the vocabulary items.
        :type counter: nltk.lm.counter.NgramCounter
        """
        self.vocab = vocabulary
        self.counts = counter
        # Used for order-level discounting or weight factors
        self._recursion_level = None
        # Kneser-Ney uses different formula for highest order
        self._is_top_recursion = True

    @abstractmethod
    def unigram_score(self, word):
        raise NotImplementedError()

    @abstractmethod
    def alpha_gamma(self, word, context):
        raise NotImplementedError()


class LanguageModel(metaclass=ABCMeta):
    """ABC for Language Models.

    Cannot be directly instantiated itself.

    """

    def __init__(
        self,
        order,
        vocabulary=None,
        counter=None,
        verbose=True,
    ):
        """Creates new LanguageModel.

        :param vocabulary: If provided, this vocabulary will be used instead
        of creating a new one when training.
        :type vocabulary: `nltk.lm.Vocabulary` or None
        :param counter: If provided, use this object to count ngrams.
        :type vocabulary: `nltk.lm.NgramCounter` or None
        :param ngrams_fn: If given, defines how sentences in training text are turned to ngram
                          sequences.
        :type ngrams_fn: function or None
        :param pad_fn: If given, defines how senteces in training text are padded.
        :type pad_fn: function or None

        """
        self.order = order
        self.vocab = Vocabulary() if vocabulary is None else vocabulary
        self.counts = NgramCounter() if counter is None else counter
        def_dict_callable = partial(defaultdict, float)
        self._cache = defaultdict(def_dict_callable)
        self.verbose = verbose

    def _update_cache(self, word):
        i, word = word
        ret_list = []
        for order in range(2, self.order + 1):
            for context in self.counts[order].keys():
                if self.counts[order][context].N() > self.cache_limit:
                    ret_list.append((context, word, self.score(word, context)))
        return ret_list

    def _check_cache_size(self):
        return getsizeof(self._cache)/1e6

    def fit(self, text, vocabulary_text=None, verbose=True):
        """Trains the model on a text.

        :param text: Training text as a sequence of sentences.

        """
        if not self.vocab:
            if vocabulary_text is None:
                raise ValueError(
                    "Cannot fit without a vocabulary or text to create it from."
                )
            self.vocab.update(vocabulary_text)
        _iter = (self.vocab.lookup(sent) for sent in text)
        self.counts.update(
            progress(_iter, desc="Fitting the model") if self.verbose else _iter
        )

    def score(self, word, context=None):
        """Masks out of vocab (OOV) words and computes their model score.

        For model-specific logic of calculating scores, see the `unmasked_score`
        method.
        """
        return self.unmasked_score(
            self.vocab.lookup(word), self.vocab.lookup(context) if context else None
        )

    @abstractmethod
    def unmasked_score(self, word, context=None):
        """Score a word given some optional context.

        Concrete models are expected to provide an implementation.
        Note that this method does not mask its arguments with the OOV label.
        Use the `score` method for that.

        :param str word: Word for which we want the score
        :param tuple(str) context: Context the word is in.
        If `None`, compute unigram score.
        :param context: tuple(str) or None
        :rtype: float

        """
        raise NotImplementedError()

    def logscore(self, word, context=None):
        """Evaluate the log score of this word in this context.

        The arguments are the same as for `score` and `unmasked_score`.

        """
        return log_base2(self.score(word, context))

    def context_counts(self, context):
        """Helper method for retrieving counts for a given context.

        Assumes context has been checked and oov words in it masked.
        :type context: tuple(str) or None

        """
        return (
            self.counts[len(context) + 1][context] if context else self.counts.unigrams
        )


    def entropy(self, text_ngrams):
        """Calculate cross-entropy of model for given evaluation text.

        :param Iterable(tuple(str)) text_ngrams: A sequence of ngram tuples.
        :rtype: float

        """
        return -1 * _mean(
            [self.logscore(ngram[-1], ngram[:-1]) for ngram in text_ngrams]
        )

    def perplexity(self, text_ngrams):
        """Calculates the perplexity of the given text.

        This is simply 2 ** cross-entropy for the text, so the arguments are the same.

        """
        return pow(
            2.0, self.entropy(progress(text_ngrams, desc="Calculating Perplexity") if self.verbose else text_ngrams)
        )

    def context_probabilities(self, context):
        """Helper method for retrieving probabilities for a given context,
        including all the words in the vocabulary

        Assumes context has been checked and oov words in it masked.
        :type context: tuple(str) or None

        """
        if context not in self._cache.keys():
            self._cache[context] = {
                word: self.score(word, context) for word in self.vocab.counts.keys()
            }
        return self._cache[context]

    def _generate_single_word(
        self, sampler_func, text_seed, random_generator, sampler_kwargs
    ):
        context = tuple(
            text_seed[-self.order + 1 :] if len(text_seed) >= self.order else text_seed
        )
        distribution = self.context_probabilities(context)
        # Sorting distribution achieves two things:
        # - reproducible randomness when sampling
        # - turns Dictionary into Sequence which `sampler` expects
        distribution = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        return sampler_func(
            distribution, random_generator=random_generator, **sampler_kwargs
        )

    def generate(
        self,
        sampler_func=greedy_decoding,
        num_words=1,
        text_seed=None,
        random_seed=None,
        sampler_kwargs={},
        EOS=None,
    ):
        text_seed = (
            random.sample(self.vocab.counts.keys(), 1)
            if text_seed is None
            else list(text_seed)
        )
        random_generator = _random_generator(random_seed)
        if EOS:
            sampler_kwargs["EOS"] = EOS
        # We build up text one word at a time using the preceding context.
        generated = []
        _iter = range(num_words)
        for _ in (progress(_iter, desc="Generating words") if self.verbose else _iter):
            token = self._generate_single_word(
                sampler_func=sampler_func,
                text_seed=text_seed + generated,
                random_generator=random_generator,
                sampler_kwargs=sampler_kwargs,
            )
            generated.append(token)
            if token == EOS:
                break
        return generated
