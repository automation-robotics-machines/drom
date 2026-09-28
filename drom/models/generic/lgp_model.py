import torch

class LatentGoalPredictor(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=256, goal_dim=7):
        super().__init__()
        self.state_dim = state_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, goal_dim)
        )

    def forward(self, x):
        pred = self.model(x)
        pred = torch.clip(pred, min=-1.0, max=+1.0)
        return pred
    
    @torch.no_grad()
    def warmup(self, device='cuda'):
        shape = (2, self.state_dim)
        x = torch.randn(shape, device=device)
        self.model(x)