import torch
from typing import List, Tuple


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    """
    Computes the precision@k for the specified values of k.
    Args:
        output: logits tensor of shape (B, num_classes)
        target: ground truth labels of shape (B)
        topk: tuple of k values
    Returns:
        list of accuracies in percentage (0 to 100)
    """
    with torch.no_grad():
        maxk = min(max(topk), output.shape[1])
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            actual_k = min(k, maxk)
            correct_k = correct[:actual_k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res
