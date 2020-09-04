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
import math
from torchtext.datasets import LanguageModelingDataset
import matplotlib.pyplot as plt

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
        weight_decay=0,
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
        self.dropout = dropout
        self.pretrained_vectors = pretrained_vectors
        self.metric = metric
        self.lr = lr
        self.weight_decay = weight_decay
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
            "dropout",
            "weight_decay",
            "tie_weights"
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
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()


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
        weight_decay=0,
        dropout=0.5,
        attention="self",
        query_dim=None,
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
        self.dropout = dropout
        self.pretrained_vectors = pretrained_vectors
        self.metric = metric
        self.lr = lr
        self.attention = attention
        self.query_dim = query_dim
        self.weight_decay = weight_decay
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
            "dropout",
            "attention",
            "query_dim",
            "weight_decay",
            "tie_weights"
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
                                options are ['LSTM', 'GRU']"""
            )

        if attention == "self":
            self.attn_layer = SelfAttentionLayer(nhid, device_type, self.query_dim)
            self.decoder = nn.Linear(self.query_dim, ntoken)
        elif attention == "dot":
            self.attn_layer = DotProductAttentionLayer(nhid, device_type)
            self.decoder = nn.Linear(2 * nhid, ntoken)
        elif attention == "additive":
            self.attn_layer = AdditiveAttentionLayer(nhid, device_type)
            self.decoder = nn.Linear(nhid, ntoken)
        else:
            raise ValueError(
                "An invalid option for attention was supplied. Options are ['self','dot','additive']"
            )

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if self.attention == 'dot':
                raise ValueError(
                    "When using the tied flag, dot attention cannot be used"
                )
            if self.attention == 'self':
                if query_dim != ninp:
                    raise ValueError(
                        "When using the tied flag and self attention, query_dim must be equal to emsize"
                    )
            else:
                if nhid != ninp:
                    raise ValueError(
                        "When using the tied flag, nhid must be equal to emsize"
                    )
            self.decoder.weight = self.encoder.weight

        self.softmax = nn.Softmax()

        self.init_weights()
        self.hidden = self.init_hidden(self.batch_size)

    def init_weights(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.encoder.weight, gain)
        nn.init.zeros_(self.decoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight, gain)

    def forward(self, input):
        emb = self.drop(self.encoder(input))
        output, self.hidden = self.rnn(emb, self.hidden)
        output, attention_weights = self.attn_layer(output)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded, attention_weights

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
        output, _ = self(text)
        loss = self.criterion(output.view(-1, self.ntoken), targets.view(-1))
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss)
        if self.metric is not None:
            metric = self.metric(output.view(-1, self.ntoken), targets.view(-1))
            result.log(self.metric.name, metric, prog_bar=True)
        return result

    # def on_after_backward(self):
    #     plot_grad_flow(self.named_parameters())

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        text, targets = batch.text, batch.target
        self.hidden = self.reset_hidden(self.hidden)
        output, _ = self(text)
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
        output, _ = self(text)
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
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)


class DotProductAttentionLayer(nn.Module):
    """Implements an Attention Layer"""

    def __init__(self, nhid, device_type):
        super(DotProductAttentionLayer, self).__init__()
        self.nhid = nhid
        self.W_output = nn.Parameter(torch.Tensor(nhid, nhid))
        self.device_type = device_type
        self.initialize_weights()

    def initialize_weights(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W_output.data, gain)

    def forward(self, inputs, attention_width=3):
        results = None
        attention_thru_time = []
        for i in range(inputs.size(0)):
            if i < 2:
                output = inputs[i].unsqueeze(0)
                output = torch.cat((output, output), dim=-1)
                attention_thru_time.append(torch.ones(1))
            else:
                lb = 0
                selector = torch.arange(lb, i + 1).long().to(self.device_type)
                # Selecting and making batch first
                # T x B x H
                vec = torch.index_select(inputs, 0, selector)
                # B x H x T
                hidden = vec[:-1].permute(1, 2, 0)
                # B x T x H
                output = vec[-1].unsqueeze(0).permute(1, 0, 2)
                output_prime = torch.bmm(
                    output, self.W_output.unsqueeze(0).repeat(output.size(0), 1, 1)
                )
                attention_weights = torch.bmm(output_prime, hidden)
                # B x 1 x T
                attention_weights = torch.softmax(attention_weights, dim=-1)
                # B x H x T
                context_vector = hidden * attention_weights.expand_as(hidden)
                # B x H
                context_vector = torch.sum(context_vector, dim=-1)
                context_vector = torch.rand(context_vector.size()).to(self.device_type)
                # B x 1 x 2H
                output = torch.cat((output, context_vector.unsqueeze(1)), dim=-1)
                output = output.permute(1, 0, 2)
                attention_thru_time.append(attention_weights.detach().cpu())

            if results is None:
                results = output
            else:
                results = torch.cat((results, output), 0)

        return results, attention_thru_time


class SelfAttentionLayer(nn.Module):
    """Implements an Attention Layer"""

    def __init__(self, nhid, device_type, query_dim=None):
        super(SelfAttentionLayer, self).__init__()
        self.nhid = nhid
        self.query_dim = nhid if query_dim is None else query_dim
        self.W_q = nn.Parameter(torch.Tensor(nhid, self.query_dim))
        self.W_k = nn.Parameter(torch.Tensor(nhid, self.query_dim))
        self.W_v = nn.Parameter(torch.Tensor(nhid, self.query_dim))
        self.device_type = device_type
        self.initialize_weights()

    def initialize_weights(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W_q.data, gain)
        nn.init.xavier_uniform_(self.W_k.data, gain)
        nn.init.xavier_uniform_(self.W_v.data, gain)

    def forward(self, inputs):
        results = None
        attention_thru_time = []
        for i in range(inputs.size(0)):
            if i < 1:
                output = inputs[i].unsqueeze(0)
                output = torch.bmm(
                    output, self.W_v.unsqueeze(0).repeat(output.size(0), 1, 1)
                )
                attention_thru_time.append(torch.ones(1))
            else:
                lb = 0
                selector = torch.arange(lb, i + 1).long().to(self.device_type)
                # Selecting and making batch first
                # B x T x H
                vec = torch.index_select(inputs, 0, selector).permute(1, 0, 2)
                # B x T x H
                output = vec[:, -1, :].unsqueeze(1)
                # B x T x H/2
                q = torch.bmm(
                    output, self.W_q.unsqueeze(0).repeat(output.size(0), 1, 1)
                )
                k = torch.bmm(vec, self.W_k.unsqueeze(0).repeat(vec.size(0), 1, 1))
                v = torch.bmm(vec, self.W_v.unsqueeze(0).repeat(vec.size(0), 1, 1))

                attention_weights = torch.bmm(q, k.permute(0, 2, 1)) / int(
                    math.sqrt(self.query_dim)
                )
                # B x 1 x T
                attention_weights = torch.softmax(attention_weights, dim=-1).squeeze()
                # B x T x H/2
                context_vector = v * attention_weights.unsqueeze(-1).expand_as(v)
                # B x H/2
                output = torch.sum(context_vector, dim=1).unsqueeze(0)
                attention_thru_time.append(attention_weights.detach().cpu())

            if results is None:
                results = output
            else:
                results = torch.cat((results, output), 0)

        return results, attention_thru_time


# https://www.aclweb.org/anthology/I17-1045.pdf
class AdditiveAttentionLayer(nn.Module):
    """Implements an Attention Layer"""

    def __init__(self, nhid, device_type):
        super(AdditiveAttentionLayer, self).__init__()
        self.nhid = nhid
        self.weight_W = nn.Parameter(torch.Tensor(nhid, nhid))
        self.weight_proj = nn.Parameter(torch.Tensor(nhid, 1))
        self.concat = nn.Linear(2 * nhid, nhid)
        self.device_type = device_type
        self.initialize_weights()

    def initialize_weights(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.weight_W.data, gain)
        nn.init.xavier_uniform_(self.weight_proj.data, gain)
        nn.init.zeros_(self.concat.bias)
        nn.init.xavier_uniform_(self.concat.weight, gain)

    def forward(self, inputs):
        results = None
        attention_thru_time = []
        for i in range(inputs.size(0)):
            if i < 2:  # i < attention_width:
                output = inputs[i]
                output = output.unsqueeze(0)
                attention_thru_time.append(torch.ones(1))
            else:
                selector = torch.arange(0, i + 1).long().to(self.device_type)
                # Selecting and making batch first
                # T x B x H
                vec = torch.index_select(inputs, 0, selector)
                # B x T x H
                hidden = vec[:-1].permute(1, 0, 2)
                # B x T x H
                output = vec[-1].unsqueeze(0).permute(1, 0, 2)
                # B x T x H
                hidden_prime = torch.tanh(
                    torch.bmm(
                        hidden, self.weight_W.unsqueeze(0).repeat(hidden.size(0), 1, 1)
                    )
                )
                # B x T x 1
                attention_weights = torch.bmm(
                    hidden_prime,
                    self.weight_proj.unsqueeze(0).repeat(hidden.size(0), 1, 1),
                )
                #  B x T x 1
                attention_weights = torch.softmax(attention_weights, dim=1)
                # B x T x H
                context_vector = hidden * attention_weights.expand_as(hidden)
                # B x H
                context_vector = torch.sum(context_vector, dim=1)
                # 1 x B x H
                context_vector = context_vector.unsqueeze(1)
                output = torch.cat((output, context_vector), dim=-1)
                output = torch.tanh(self.concat(output)).permute(1, 0, 2)
                attention_thru_time.append(attention_weights.detach().cpu())

            if results is None:
                results = output
            else:
                results = torch.cat((results, output), 0)

        return results, attention_thru_time


class AttentionLayer_bkp(nn.Module):
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


class TransformerModel(pl.LightningModule):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        ntoken,
        ninp,
        nhead,
        nhid,
        nlayers,
        batch_size,
        device_type,
        lr=1e-3,
        weight_decay = 0,
        dropout=0.5,
        criterion=nn.CrossEntropyLoss(),
        pretrained_vectors=None,
        metric=None,
    ):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError(
                "TransformerEncoder module does not exist in PyTorch 1.1 or lower."
            )
        self.model_type = "Transformer"
        self.ntoken = ntoken
        self.device_type = device_type
        self.criterion = criterion
        self.batch_size = batch_size
        self.nhid = nhid
        self.ninp = ninp
        self.nhead = nhead
        self.nlayers = nlayers
        self.pretrained_vectors = pretrained_vectors
        self.metric = metric
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(
            "ntoken",
            "ninp",
            "nhid",
            "nhead",
            "nlayers",
            "batch_size",
            "device_type",
            "dropout",
            "criterion",
            "metric",
            "lr",
            "weight_decay"
        )

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)
        self.drop = nn.Dropout(dropout)
        if pretrained_vectors is not None:
            assert pretrained_vectors.shape == torch.Size(
                [ntoken, ninp]
            ), "When using pretrained embeddings, the embedding vector should have the dimensions (ntoken, ninp)"
            self.encoder.weight.data.copy_(pretrained_vectors)

        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.encoder.weight, gain)
        nn.init.zeros_(self.decoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight, gain)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def training_step(self, batch, batch_nb):
        # REQUIRED
        text, targets = batch.text, batch.target
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
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay= self.weight_decay)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def test():
    from data_module import QuotesDataModule

    dm = QuotesDataModule(
        train_file="data/quotesdb/funny_quotes.train.txt",
        valid_file="data/quotesdb/funny_quotes.val.txt",
        test_file="data/quotesdb/funny_quotes.test.txt",
        tokenizer=None,
        batch_size=32,
        bptt=6,
    )
    device = torch.device(
        "cuda"
    )  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vocab = dm.TEXT.vocab
    # model = TransformerModel(
    #     ntoken=len(vocab),
    #     ninp=200,
    #     nhead=2,
    #     nhid=32,
    #     nlayers=2,
    #     batch_size=32,
    #     device_type=device.type,
    #     lr=1e-3,
    #     dropout=0.5,
    #     criterion=nn.CrossEntropyLoss(),
    #     pretrained_vectors=None,
    #     metric=None,
    # ).to(device)

    model = RNNAttentionModel(
        rnn_type="LSTM",
        ntoken=len(vocab),
        ninp=200,
        nhid=200,
        nlayers=2,
        batch_size=32,
        device_type=device.type,
        lr=1e-3,
        dropout=0.5,
        attention="self",
        query_dim=200,
        criterion=nn.CrossEntropyLoss(),
        pretrained_vectors=None,
        metric=None,
        tie_weights=True,
    ).to(device)

    trainer = pl.Trainer(
        gpus=1 if device.type == "cuda" else 0,
        max_epochs=10,
        logger=True,
        auto_lr_find=False,
        fast_dev_run=True,
    )  # , logger= wandb_logger) #fast_dev_run=True,

    trainer.fit(model, datamodule=dm)

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


if __name__ == "__main__":
    test()
