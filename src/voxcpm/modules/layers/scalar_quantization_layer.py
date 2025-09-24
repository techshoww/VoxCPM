import torch
import torch.nn as nn
import os
import time
import numpy as np

class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.scale = scale

        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)
    
        self.calib_dir = "calib_fsq_layer"
        if eval(os.getenv("save_calib", "False")):
            if os.path.exists(self.calib_dir):
                os.removedirs(self.calib_dir)
            os.makedirs(self.calib_dir)

    def forward(self, hidden):
        if eval(os.getenv("save_calib", "False")):
            t_str = str(time.time())
            if hidden.shape[0]==1 and hidden.shape[1]==1024:
                np.save(f"{self.calib_dir}/fsq_layer.hidden.{t_str}.npy", hidden.detach().cpu().numpy())

        hidden = self.in_proj(hidden)
        hidden = torch.tanh(hidden)

        if self.training:
            quantized = torch.round(hidden * self.scale) / self.scale
            hidden = hidden + (quantized - hidden).detach()
        else:
            hidden = torch.round(hidden * self.scale) / self.scale

        return self.out_proj(hidden)