"""Portable sampling for the original half-precision D2PRL descriptors."""
def sample_half(features, grid):
    import torch.nn.functional as F
    # PyTorch 2.8's ARM CPU fp16 grid sampler segfaults on real D2PRL fields.
    # Keep upstream fp16 feature/grid quantisation, interpolate in fp32 on CPU,
    # then round back to the original descriptor dtype before comparisons.
    if features.device.type == 'cpu':
        return F.grid_sample(features.float(), grid.float(), align_corners=True).to(features.dtype)
    return F.grid_sample(features, grid, align_corners=True)


def evaluate_batched(self, features, offset_x, offset_y):
    """Group candidate samples; preserve the upstream fp16 arithmetic/order."""
    import torch
    b, channels, h, w = features.shape
    lx = self.left_x_coordinate[:b].to(offset_x.device)
    ly = self.left_y_coordinate[:b].to(offset_y.device)
    rx = torch.clamp(lx + offset_x, min=0, max=w-1)
    ry = torch.clamp(ly + offset_y, min=0, max=h-1)
    rx -= (w-1)/2; rx /= (w-1)/2
    ry -= (h-1)/2; ry /= (h-1)/2
    grid = torch.stack((rx, ry), dim=-1).half()
    group = channels // 3
    strengths = []
    # Eight candidates bound temporary memory without restricting the search.
    for begin in range(0, grid.shape[1], 8):
        g = grid[:, begin:begin+8]; k = g.shape[1]
        a = sample_half(features, g.reshape(b, k*h, w, 2)).reshape(b, channels, k, h, w)
        ref = features.unsqueeze(2)
        best = None
        for shift in (0, group, 2*group):
            rolled = torch.roll(a, shift, 1) if shift else a
            differences = -1.0 * torch.abs(ref-rolled)
            for channel in range(0, channels, group):
                strength = torch.mean(differences[:,channel:channel+group],dim=1)
                best = strength if best is None else torch.maximum(best,strength)
        strengths.append(best)
    strength = torch.cat(strengths,dim=1)
    strength = torch.softmax(strength*1000,dim=1)
    x = torch.sum(offset_x*strength,dim=1,keepdim=True)
    y = torch.sum(offset_y*strength,dim=1,keepdim=True)
    return torch.clamp(x+lx,min=0,max=w-1)-lx, torch.clamp(y+ly,min=0,max=h-1)-ly
