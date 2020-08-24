import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import TensorMetric


class Perplexity(TensorMetric):
    """
    Computes the perplexity of the model.
    """

    def __init__(self, pad_idx: int, *args, **kwargs):
        super().__init__(name='ppl')
        self.pad_idx = pad_idx

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        loss = F.cross_entropy(pred, target, reduction='none')
        non_padding = target.ne(self.pad_idx)
        loss = loss.masked_select(non_padding).sum()

        num_words = non_padding.sum()
        ppl = torch.exp(
            torch.min(loss / num_words, torch.tensor([100]).type_as(loss))
        )
        return ppl
