# imports
import torch

def compute_pos(outputs: torch.Tensor):
    # shape of outputs: (N, 30, 2)
    heading = outputs[:,:,0] * torch.pi
    speed = outputs[:,:,1]
    
    vx = speed * torch.cos(heading) # (N, 30)
    vy = speed * torch.sin(heading) # (N, 30)
    
    dx = vx * 0.1 # (N, 30)
    dy = vy * 0.1 # (N, 30)

    # upsample to 60 timesteps
    dx = dx.repeat_interleave(2, dim=1) # (N, 60)
    dy = dy.repeat_interleave(2, dim=1) # (N, 60)
    
    px = torch.cumsum(dx, dim=1) # (N, 60)
    py = torch.cumsum(dy, dim=1) # (N, 60)
    position = torch.stack([px, py], dim=2) # (N, 60, 2)

    return position


def calculate_ADE_FDE(pred_pos, ground_truth):
    distances = torch.linalg.norm(pred_pos - ground_truth, dim=2) # N x 60
    ADE = distances.mean(dim=1) # N
    FDE = distances[:, -1] # N
    
    return ADE.mean(), FDE.mean()
