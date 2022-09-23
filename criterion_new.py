import torch
import torch.nn as nn

class dot_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, simls, target) :

        B, C = logit.shape
        log_prob = logit.log_softmax(dim=-1)
        #print(log_prob)
        entropy_to_add = -log_prob[torch.arange(B), target]
        # print(torch.arange(B))
        # print(entropy_to_add)
        loss = entropy_to_add.mean()
        return loss

class Accuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logit:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        B, C = logit.shape
        pred = logit.argmax(dim=-1)
        acc = torch.sum(pred == target) / float(target.numel())
        return acc

# if __name__ == '__main__':
#     a = torch.tensor([[0.0412, 0.0303, 0.1121],
#             [0.0583, 0.0413, 0.1531],
#             [0.1802, -0.0081, 0.0972],
#             [0.1457, -0.0189, 0.0235],
#             [-0.0522, 0.1053, 0.1644],
#             [0.0987, -0.0642, 0.1831],
#             [0.0684, 0.1632, 0.1211],
#             [0.0885, -0.0460, 0.1349]])
#     b = torch.tensor([1,0,0,0,1,2,2,2])
#     model = CrossEntropyWithLogit()
#     output = model(a,b)

