import torch

def gauss_attn(x:torch.Tensor, sigmas:torch.Tensor, dist:torch.Tensor=None) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if sigmas.dim() == 1:
        sigmas = sigmas.repeat((x.shape[0], 1))
    if dist is None:
        dist = torch.cdist(x, x, p=1)
    if dist.dim() == 2:
        dist = dist.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)
    dist = dist.unsqueeze(1).repeat((1, sigmas.shape[-1], 1, 1))
    dist = dist.permute((0, 2, 3, 1))
    if sigmas.dim() == 2:
        sigmas = sigmas.unsqueeze(1)
    if sigmas.dim() == 3:
        sigmas = sigmas.unsqueeze(2)
    y = ((-(dist**2)/(2*(sigmas**2))).exp())
    y = y.permute((0, 3, 1, 2))
    return y

class GaussianAttention(torch.nn.Module):
    def __init__(self, sigmas):
        super(GaussianAttention, self).__init__()
        self.sigmas = torch.nn.Parameter(torch.tensor(sigmas))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return gauss_attn(x, self.sigmas)
