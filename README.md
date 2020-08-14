# Language Model: From Zero to Hero

> “You shall know the nature of a word by the company it keeps.” – John
> Rupert Firth

Text Generation(NLG) is ubiquitous in many NLP tasks, from summarization, to dialogue and machine translation. And a Language Model plays it part in each one of them. The goal of language modelling is to estimate the probability distribution of various linguistic units, e.g., words, sentences etc.([Source](https://medium.com/syncedreview/language-model-a-survey-of-the-state-of-the-art-technology-64d1a2e5a466)) In essence, we want our language model to learn the "grammar" and "style" of the text on which it was trained on.

As discussed before, a Language Model learns the "grammar" and "style" of the text on which it was trained on. But how do we translate that to a mathematical statement? We know that there is no hard and fast rules in language. An idea can be expressed in a lot of different ways. Therefore stochasticity is built into language. So, it follows that the mathematical model we aim for language is also probabilistic.

An LM (or what we casually refer to as an LM) is typically comprised of a few components as shown below.

![LM Overview](https://i.ibb.co/QjxsWWj/lm-overview.png)

1. The core Language Model which learns the "style and grammar" of the language.
2. A mechanism to generate robust probability distribution over words - eg. Smoothing or Backoff in Statistical Language Models
3. A mechanism to decode the probability distribution and generate text - eg. Greedy Decoding, Beam Search, top-k sampling, etc.
4. A downstream task - eg. Machine Translation, Text Summarization, etc.

**The Beginning (10 min)** : Pre-processing of Data. Representing Text as Numbers. N-Grams. Sparse vs Dense Vectors & Other Intuitions.

**The Middle ages  - I (25 min)** : Statistical Language Models. Markov Chains. Smoothing and Backoff.

**The Middle ages - II (25 min)** : Decoding Strategies - Beam Search and its variants, Top k sampling, Nucleus Sampling

**The Modern Age (25 min)** : Deep learning. AWD-LSTMs, GPTs

**The Future (20 min)** : Building a simple Chat-bot tying things together

**Q/A : Ad-Hoc and Rest of Time**

Prerequisites:
 - [Probability](https://tinyheero.github.io/2016/03/20/basic-prob.html)
	- Joint Probability
	- Conditional Probability
 - [N-Grams](https://towardsdatascience.com/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing-9d9eef0fa058)
 - [Markov Chains](https://setosa.io/ev/markov-chains/)
 - [Entropy, Cross Entropy](https://deep-and-shallow.com/2020/01/09/deep-learning-and-information-theory/)
 - Basic familiarity with NLP
 