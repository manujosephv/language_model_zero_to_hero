import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable as V
import torchtext
from torchtext import data
import spacy
from torchtext.datasets import LanguageModelingDataset


class RNNModel(pl.LightningModule):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        batch_size,
        device_type,
        dropout=0.5,
        criterion=nn.CrossEntropyLoss(),
        pretrained_vectors=None,
        metric=None,
        tie_weights=False,
    ):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.device_type = device_type
        self.criterion = criterion
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.pretrained_vectors = pretrained_vectors
        self.metric = metric
        self.save_hyperparameters(
            "rnn_type",
            "ntoken",
            "ninp",
            "nhid",
            "nlayers",
            "batch_size",
            "device_type",
            "dropout",
            "criterion",
            "metric",
        )

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if pretrained_vectors is not None:
            assert pretrained_vectors.shape == torch.Size(
                [ntoken, ninp]
            ), "When using pretrained embeddings, the embedding vector should have the dimensions (ntoken, ninp)"
            self.encoder.weight.data.copy_(pretrained_vectors)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            raise ValueError(
                """An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
            )

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.hidden = self.init_hidden(self.batch_size)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):
        emb = self.drop(self.encoder(input))
        output, self.hidden = self.rnn(emb, self.hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid),
            )
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhid)

    def reset_hidden(self, hidden):
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.detach().to(self.device_type)
        else:
            hidden = tuple(self.reset_hidden(v) for v in hidden)
        return hidden

    def training_step(self, batch, batch_nb):
        # REQUIRED
        text, targets = batch.text, batch.target
        self.hidden = self.reset_hidden(self.hidden)
        output = self(text)
        loss = self.criterion(output.view(-1, self.ntoken), targets.view(-1))
        result_dict = {"loss": loss}
        tensorboard_logs = {"train_loss": loss}
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result_dict[self.metric.name] = metric
            tensorboard_logs[f"train_{self.metric.name}"] = metric
        result_dict["log"] = tensorboard_logs
        return result_dict

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        text, targets = batch.text, batch.target
        self.hidden = self.reset_hidden(self.hidden)
        output = self(text)
        result_dict = {
            "val_loss": self.criterion(output.view(-1, self.ntoken), targets.view(-1))
        }
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result_dict[f"val_{self.metric.name}"] = metric
        return result_dict

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        result_dict = {"val_loss": avg_loss}
        tensorboard_logs = {"val_loss": avg_loss}
        if self.metric is not None:
            avg_metric = torch.stack(
                [x[f"val_{self.metric.name}"] for x in outputs]
            ).mean()
            result_dict[f"val_{self.metric.name}"] = avg_metric
            tensorboard_logs[f"val_{self.metric.name}"] = avg_metric
        result_dict["log"] = tensorboard_logs
        return result_dict

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        text, targets = batch.text, batch.target
        self.hidden = self.reset_hidden(self.hidden)
        output = self(text)
        result_dict = {
            "test_loss": self.criterion(output.view(-1, self.ntoken), targets.view(-1))
        }
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result_dict[f"test_{self.metric.name}"] = metric
        return result_dict

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        result_dict = {"test_loss": avg_loss}
        tensorboard_logs = {"test_loss": avg_loss}
        if self.metric is not None:
            avg_metric = torch.stack(
                [x[f"test_{self.metric.name}"] for x in outputs]
            ).mean()
            result_dict[f"test_{self.metric.name}"] = avg_metric
            tensorboard_logs[f"test_{self.metric.name}"] = avg_metric
        result_dict["log"] = tensorboard_logs
        return result_dict

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)


# from data_module import QuotesDataModule

# dm = QuotesDataModule(
#     train_file="data/quotesdb/funny_quotes.train.txt",
#     valid_file="data/quotesdb/funny_quotes.val.txt",
#     test_file="data/quotesdb/funny_quotes.test.txt",
#     tokenizer=None,
#     batch_size=32,
#     bptt=6,
# )
# device = torch.device('cpu') #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# vocab = dm.TEXT.vocab
# model = RNNModel(
#     rnn_type="LSTM", ntoken=len(vocab), ninp=200, nhid=32, nlayers=2, batch_size=32, device_type= device.type
# ).to(device)


# toks = dm.TEXT.preprocess("When life hands you lemons")
# x = dm.TEXT.numericalize([toks]).to('cpu').squeeze(1)
# seq = torch.zeros(6, dtype=torch.long)
# l = min(len(x), 6)
# seq[-l:] = x[-l:]
# seq = seq.unsqueeze(1)


# toks = dm.TEXT.preprocess("When life hands you lemons")
# x = dm.TEXT.numericalize([toks]).to('cpu').squeeze()
# x = x.unsqueeze(0).repeat(2,1)
# model.hidden = model.init_hidden(1)
# model = model.to('cpu')
# model.eval()
# out = model(seq)
# print (out.shape)

# toks = dm.TEXT.preprocess("When life hands you lemons")
# x = dm.TEXT.numericalize([toks]).to(device).squeeze(1)
# x = x.unsqueeze(1)
# model(x)

# trainer = pl.Trainer(gpus=1 if device.type =='cuda' else 0)
# trainer.fit(model, datamodule=dm)

