import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torchtext
from torchtext import data
import spacy
from torchtext.datasets import LanguageModelingDataset

# Modified version of the LM here
# https://github.com/pytorch/examples/blob/master/word_language_model/main.py
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
        lr=1e-3,
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
        self.ninp = ninp
        self.nlayers = nlayers
        self.pretrained_vectors = pretrained_vectors
        self.metric = metric
        self.lr = lr
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
            "lr",
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
        # initrange = 0.1
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.encoder.weight, gain)
        nn.init.xavier_uniform_(self.decoder.weight, gain)
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # nn.init.zeros_(self.decoder.weight)
        # nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):
        emb = self.drop(self.encoder(input))
        output, self.hidden = self.rnn(emb, self.hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded

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
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss)
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result.log(self.metric.name, metric, prog_bar=True)
        return result

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        text, targets = batch.text, batch.target
        self.hidden = self.reset_hidden(self.hidden)
        output = self(text)
        val_loss = self.criterion(output.view(-1, self.ntoken), targets.view(-1))
        result = pl.EvalResult(early_stop_on=val_loss)
        result.log(
            "val_loss", val_loss, prog_bar=True,
        )
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result.log(f"val_{self.metric.name}", metric, prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        text, targets = batch.text, batch.target
        self.hidden = self.reset_hidden(self.hidden)
        output = self(text)
        result = pl.EvalResult()
        result.log(
            "test_loss", self.criterion(output.view(-1, self.ntoken), targets.view(-1))
        )
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result.log(f"test_{self.metric.name}", metric)
        return result

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def batch_matmul(seq, weight, nonlinearity=""):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if nonlinearity == "tanh":
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze()


class RNNAttentionModel(pl.LightningModule):
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
        lr=1e-3,
        dropout=0.5,
        attention_width=3,
        criterion=nn.CrossEntropyLoss(),
        pretrained_vectors=None,
        metric=None,
        tie_weights=False,
    ):
        super(RNNAttentionModel, self).__init__()
        self.ntoken = ntoken
        self.device_type = device_type
        self.criterion = criterion
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers
        self.pretrained_vectors = pretrained_vectors
        self.metric = metric
        self.lr = lr
        self.attention_width = attention_width
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
            "lr",
            "attention_width",
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

        self.softmax = nn.Softmax()
        self.AttentionLayer = AttentionLayer(nhid, device_type)
        self.init_weights()
        self.hidden = self.init_hidden(self.batch_size)

    def init_weights(self):
        # initrange = 0.1
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.encoder.weight, gain)
        nn.init.xavier_uniform_(self.decoder.weight, gain)
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # nn.init.zeros_(self.decoder.weight)
        # nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):
        emb = self.drop(self.encoder(input))
        output, self.hidden = self.rnn(emb, self.hidden)
        output = self.AttentionLayer.forward(output, self.attention_width)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded

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
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss)
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result.log(self.metric.name, metric, prog_bar=True)
        return result

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        text, targets = batch.text, batch.target
        self.hidden = self.reset_hidden(self.hidden)
        output = self(text)
        val_loss = self.criterion(output.view(-1, self.ntoken), targets.view(-1))
        result = pl.EvalResult(early_stop_on=val_loss)
        result.log(
            "val_loss", val_loss, prog_bar=True,
        )
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result.log(f"val_{self.metric.name}", metric, prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        text, targets = batch.text, batch.target
        self.hidden = self.reset_hidden(self.hidden)
        output = self(text)
        result = pl.EvalResult()
        result.log(
            "test_loss", self.criterion(output.view(-1, self.ntoken), targets.view(-1))
        )
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result.log(f"test_{self.metric.name}", metric)
        return result

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class AttentionLayer(nn.Module):
    """Implements an Attention Layer"""

    def __init__(self, nhid, device_type):
        super(AttentionLayer, self).__init__()
        self.nhid = nhid
        self.weight_W = nn.Parameter(torch.Tensor(nhid, nhid))
        self.weight_proj = nn.Parameter(torch.Tensor(nhid, 1))
        self.softmax = nn.Softmax()
        self.weight_W.data.uniform_(-0.1, 0.1)
        self.weight_proj.data.uniform_(-0.1, 0.1)
        self.device_type = device_type

    def forward(self, inputs, attention_width=3):
        results = None
        for i in range(inputs.size(0)):
            if i < attention_width:
                output = inputs[i]
                output = output.unsqueeze(0)

            else:
                lb = i - attention_width
                if lb < 0:
                    lb = 0
                selector = torch.from_numpy(np.array(np.arange(lb, i)))
                # if self.cuda:
                #     selector = Variable(selector).cuda()
                # else:
                #     selector = Variable(selector)
                selector = Variable(selector).long().to(self.device_type)
                vec = torch.index_select(inputs, 0, selector)
                u = batch_matmul(vec, self.weight_W, nonlinearity="tanh")
                a = batch_matmul(u, self.weight_proj)
                a = self.softmax(a)
                output = None
                for i in range(vec.size(0)):
                    h_i = vec[i]
                    a_i = a[i].unsqueeze(1).expand_as(h_i)
                    h_i = a_i * h_i
                    h_i = h_i.unsqueeze(0)
                    if output is None:
                        output = h_i
                    else:
                        output = torch.cat((output, h_i), 0)
                output = torch.sum(output, 0)
                output = output.unsqueeze(0)

            if results is None:
                results = output

            else:
                results = torch.cat((results, output), 0)

        return results


# from data_module import QuotesDataModule

# dm = QuotesDataModule(
#     train_file="data/quotesdb/funny_quotes.train.txt",
#     valid_file="data/quotesdb/funny_quotes.val.txt",
#     test_file="data/quotesdb/funny_quotes.test.txt",
#     tokenizer=None,
#     batch_size=32,
#     bptt=6,
# )
# device = torch.device('cuda') #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# vocab = dm.TEXT.vocab
# model = RNNAttentionModel(
#     rnn_type="LSTM", ntoken=len(vocab), ninp=200, nhid=32, nlayers=2, batch_size=32, device_type= device.type
# ).to(device)

# trainer = pl.Trainer(gpus=1 if device.type =='cuda' else 0, 
#                      max_epochs=5, 
#                      logger=True, 
#                      auto_lr_find=False,
#                     fast_dev_run=True)#, logger= wandb_logger) #fast_dev_run=True,

# trainer.fit(model, datamodule=dm)

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

