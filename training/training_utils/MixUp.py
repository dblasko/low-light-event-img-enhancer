import torch


class MixUp:
    def __init__(self):
        self.distribution = torch.distributions.beta.Beta(
            torch.tensor([1.2]), torch.tensor([1.2])
        )

    def augment(self, gt, ip):
        batch_size = gt.size(0)
        idx = torch.randperm(batch_size)
        gt1 = gt[idx]
        ip1 = ip[idx]

        lam = self.distribution.rsample((batch_size, 1)).view(-1, 1, 1, 1).to(gt.device)
        gt = lam * gt + (1 - lam) * gt1
        ip = lam * ip + (1 - lam) * ip1

        return gt, ip
