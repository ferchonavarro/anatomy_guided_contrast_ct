import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class CrossEntropyLoss(_Loss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        # self.nll_loss = nn.CrossEntropyLoss(weight)
        # self.cross_entropy_loss = CrossEntropyLoss2d()

    def forward(self, input, target, weight=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """

        if weight is not None:
            y_1 = F.cross_entropy(input=input, target=target, weight=weight, reduction='mean')
        else:
            y_1 = F.cross_entropy(input=input, target=target, weight=None, reduction='mean')

        return y_1
