import torch
import torch.nn as nn
import numpy as np
from ..minicpm4 import MiniCPMModel, MiniCPM4Config
import os
if os.getenv("AX_INFER", "false").lower() == "true":
    from ...npu_infer.utils_lm import MiniCPMModel_AXInfer 
    from ...npu_infer.utils_axinfer import AxModelInfer
from einops import rearrange


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        self.config = config
        
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))
        if os.getenv("AX_INFER", "false").lower() != "true":
            self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))
            self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
            self.encoder = MiniCPMModel(config)     
        else:
            axmodel_dir = os.getenv("AXMODEL_DIR")
            self.special_tokens = np.load(f"{axmodel_dir}/axmodels/feat_encoder.special_token.npy") 
            self.in_proj = AxModelInfer(f"{axmodel_dir}/axmodels/feat_encoder.in_proj.onnx")
            self.encoder = MiniCPMModel_AXInfer(config, f"{axmodel_dir}/feat_encoder_encoder-axmodels/", 
                                            "MiniCPMForCausalLM", 256, 512, chunk_len=64)

    def forward(self, x):
        """
        x: [B, T, P, D]
        """
        B, T, P, D = x.shape

        if os.getenv("AX_INFER", "false").lower() != "true":
            x = self.in_proj(x)
            special_tokens = self.special_token.expand(B, T, 1, -1)
            x = torch.cat([special_tokens, x], dim=2)
        else:
            assert B==1, f"not support B={B}"
            device = x.device
            outputs = []
            for i in range(T):
                input = {"x":x[:,i:i+1].detach().cpu().numpy()}
                output = self.in_proj(input)[0]
                output = np.concatenate([self.special_tokens, output], axis=2)
                outputs.append(output)
            outputs = np.concatenate(outputs,axis=1)
            x = torch.from_numpy(outputs).to(device)

        x = rearrange(x, "b t p c -> (b t) p c")
        if os.getenv("AX_INFER", "false").lower() != "true":
            outputs, _ = self.encoder(x, is_causal=False)
        else:
            outputs = []
            for i in range(x.shape[0]):
                output = self.encoder(x[i:i+1], is_causal=False)
                outputs.append(output)
            outputs = torch.cat(outputs, 0)

        cls_output = outputs[:, 0, :]

        return rearrange(cls_output, "(b t) c -> b t c", b=B)
