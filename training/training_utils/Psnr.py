import torch


def psnr(target, predicted):
    diff = torch.clamp(predicted, 0, 1) - torch.clamp(target, 0, 1)
    rmse = (diff**2).mean().sqrt()
    return 20 * torch.log10(1 / rmse)


def batch_psnr(targets, predictions, range=None):
    psnrs = []
    for target, prediction in zip(targets, predictions):
        psnrs.append(psnr(target, prediction))
    return sum(psnrs) / len(psnrs)
