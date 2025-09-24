import torch
import torch.nn as nn
from ..minicpm4 import MiniCPMModel, MiniCPM4Config
from einops import rearrange
import os
import numpy as np
import time 
class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        self.config = config
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)

        assert config.vocab_size == 0, "vocab_size must be 0 for local encoder"
        self.encoder = MiniCPMModel(config)

        self.calib_dir = "calib_feat_encoder"
        if eval(os.getenv("save_calib", "False")):
            if os.path.exists(self.calib_dir):
                os.removedirs(self.calib_dir)
            os.makedirs(self.calib_dir)
            

    def forward(self, x):
        """
        x: [B, T, P, D]
        """
        B, T, P, D = x.shape
        if eval(os.getenv("save_calib", "False")):
            if T==1:
                np.save(f"{self.calib_dir}/feat_encoder.x.{time.time()}.npy", x.detach().cpu().numpy())
        
        x = self.in_proj(x)
        special_tokens = self.special_token.expand(B, T, 1, -1)
        x = torch.cat([special_tokens, x], dim=2)
        # x = rearrange(x, "b t p c -> (b t) p c")
        b,t,p,c = x.shape
        x = x.reshape(b*t,p,c)
        outputs, _ = self.encoder(x, is_causal=False)
        cls_output = outputs[:, 0, :]

        # return rearrange(cls_output, "(b t) c -> b t c", b=B)
        bt, c = cls_output.shape
        cls_output = cls_output.reshape(B, -1, c)
        return cls_output
