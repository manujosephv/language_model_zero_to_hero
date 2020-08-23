import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import spacy
from torchtext.datasets import LanguageModelingDataset
from torchtext import data
from torchtext.vocab import Vectors
from pathlib import Path

en = spacy.load("en")


def spacy_tokenizer(x):
    return [tok.text for tok in en.tokenizer(x)]


# pip install pytorch-lightning==0.9.0rc2
class QuotesDataModule(pl.LightningDataModule):

    ALLOWABLE_PRETRAINED = [
        "charngram.100d",
        "fasttext.en.300d",
        "fasttext.simple.300d",
        "glove.42B.300d",
        "glove.840B.300d",
        "glove.twitter.27B.25d",
        "glove.twitter.27B.50d",
        "glove.twitter.27B.100d",
        "glove.twitter.27B.200d",
        "glove.6B.50d",
        "glove.6B.100d",
        "glove.6B.200d",
        "glove.6B.300d",
    ]

    def __init__(
        self,
        train_file: str,
        valid_file: str = None,
        test_file: str = None,
        tokenizer=None,
        pretrained_vectors=None,
        batch_size=32,
        bptt=6,
    ):

        super().__init__()
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = valid_file
        self.tokenizer = spacy_tokenizer if tokenizer is None else tokenizer
        self.TEXT = data.Field(lower=True, tokenize=self.tokenizer)
        self.pretrained_vectors = pretrained_vectors
        self.batch_size = batch_size
        self.bptt = bptt
        self._load_data()
        self._build_vocab()
        self.vocab = self.TEXT.vocab

    def _load_data(self):
        # Read file and tokenize
        self.train_data = LanguageModelingDataset(self.train_file, self.TEXT)
        if self.valid_file:
            self.valid_data = LanguageModelingDataset(self.valid_file, self.TEXT)
        else:
            self.valid_data = None
        if self.test_file:
            self.test_data = LanguageModelingDataset(self.test_file, self.TEXT)
        else:
            self.test_data = None

    def _build_vocab(self):
        if self.pretrained_vectors:
            if isinstance(self.pretrained_vectors, str):
                assert (
                    self.pretrained_vectors in self.ALLOWABLE_PRETRAINED
                ), f"pretrained_vectors should be one of {self.ALLOWABLE_PRETRAINED}"
            elif isinstance(self.pretrained_vectors, Vectors):
                pass
            else:
                raise ValueError(
                    "pretrained_vectors should either be str or torch.vocab.Vectors"
                )

        self.TEXT.build_vocab(self.train_data, vectors=self.pretrained_vectors)

    @classmethod
    def _make_iter(cls, dataset, batch_size, bptt_len):
        if dataset:
            _iter = data.BPTTIterator(
                dataset,
                batch_size=batch_size,
                bptt_len=bptt_len,  # this is where we specify the sequence length
                repeat=False,
                shuffle=True,
            )
        else:
            _iter = []
        return _iter

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self._make_iter(self.train_data, self.batch_size, self.bptt)

    def val_dataloader(self):
        return self._make_iter(self.valid_data, self.batch_size, self.bptt)

    def test_dataloader(self):
        return self._make_iter(self.test_data, self.batch_size, self.bptt)


# dm = QuotesDataModule(
#     train_file="data/quotesdb/funny_quotes.train.txt",
#     valid_file="data/quotesdb/funny_quotes.val.txt",
#     test_file="data/quotesdb/funny_quotes.test.txt",
#     tokenizer=None,
#     batch_size=32,
#     bptt=6,
#     pretrained_vectors="fasttext.simple.300d",
# )
