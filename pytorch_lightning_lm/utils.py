import torch
from torch.nn import functional as F
import spacy
import nltk
from tqdm.autonotebook import tqdm

en = spacy.load("en")


def spacy_tokenizer(x):
    return [tok.text for tok in en.tokenizer(x)]


def logscore(model, vocab, word, context, device):
    inp = torch.LongTensor([vocab.stoi[x] for x in context]).unsqueeze(1).to(device)
    word_idx = vocab.stoi[word]
    out = F.log_softmax(model(inp), dim=1)[-1, :]
    return out[word_idx].item()


def perplexity(model, vocab, ngrams, device):
    log_score_sum = 0
    log_score_count = 0
    for ngram in tqdm(ngrams):
        log_score_sum += logscore(model, vocab, ngram[-1], ngram[:-1], device)
        log_score_count += 1
    entropy = -1 * (log_score_sum / log_score_count)
    return pow(2.0, entropy)


def generate_sentence(model, vocab, tokenizer, seed, sampler, sampler_kwargs={}, num_words=20, device='cpu'):
    sentence = []
    toks = tokenizer(seed)
    x = torch.LongTensor([vocab.stoi[x] for x in toks]).to(device)
    for i in range(num_words):
        out = model(x.unsqueeze(1))
        gen_token = sampler(out[-1,:], **sampler_kwargs)
        sentence.append(gen_token.item())
        if gen_token == vocab.stoi['<eos>']:
            break
        gen_token = gen_token.unsqueeze(0)
        x = torch.cat([x, gen_token])[1:]
    return " ".join([vocab.itos[word] for word in sentence])



# import random
# random.seed(42)
# import torch
# import torch.nn as nn
# import numpy as np
# import pytorch_lightning as pl
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from torch.autograd import Variable as V
# import torchtext
# from torchtext import data
# from data_module import QuotesDataModule
# from model import RNNModel
# from metrics import Perplexity
# from torchtext.datasets import LanguageModelingDataset
# from argparse import ArgumentParser

# import spacy
# import nltk
# from tqdm.autonotebook import tqdm
# en = spacy.load("en")

# def spacy_tokenizer(x):
#     return [tok.text for tok in en.tokenizer(x)]

# # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
# trainer = pl.Trainer(gpus=1 if device.type =='cuda' else 0, auto_lr_find=False)
# model = RNNModel.load_from_checkpoint("models/LSTM_LM_unk_2.ckpt")
# model.eval()
# model = model.to(device)
# model.hidden = model.init_hidden(1)
# vocab = torch.load("models/LSTM_LM_vocab_unk_2.sav")
# weight_matrix = vocab.vectors
# ntoken, ninp = weight_matrix.shape
# assert(model.encoder.weight.data.shape == torch.Size([ntoken,ninp]))
# bptt = 16
# TEXT = data.Field(lower=True, tokenize=spacy_tokenizer)
# test_data = LanguageModelingDataset("data/quotesdb/funny_quotes.test.txt", TEXT)

# tokens = test_data.examples[0].text
# from samplers import BeamSearch, DiverseNbestBeamSearch, DiverseBeamSearch, greedy_decoding, weighted_random_choice, topk, nucleus
# # generate_sentence(model, vocab, tokenizer=spacy_tokenizer, seed="when life hands you lemons", sampler=greedy_decoding, num_words=20, device=device)

# # from IPython.display import display, Markdown, Latex

# seeds = [
#     "when life hands you a lemon,",
#     "life is a",
#     "i'd rather be pissed",
#     "all women may not be beautiful but",
#     "i really need a day between",
#     "it's never too late to"
# ]

# def generate_sentences(model, vocab, tokenizer, sampler_func, seeds, sampler_kwargs={}, num_words = 20, device='cpu'):
#     for seed in seeds:
#         if isinstance(sampler_func, BeamSearch):
#             gen_text = sampler_func.generate(text_seed=seed, num_words=20)
#         else:
#             gen_text = generate_sentence(model, vocab, tokenizer=tokenizer, seed=seed, sampler=sampler_func,sampler_kwargs={}, num_words=num_words, device=device)
#         print(gen_text)

# generate_sentences(model, vocab, tokenizer=spacy_tokenizer, sampler_func = greedy_decoding, seeds=seeds, sampler_kwargs={}, num_words = 20, device='cpu')