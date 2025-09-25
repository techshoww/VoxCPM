import torch
import torch.nn as nn
from ..minicpm4 import MiniCPMModel, MiniCPM4Config
from ...npu_infer.utils_lm import MiniCPMModel_AXInfer 
from einops import rearrange


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        self.config = config
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)

        assert config.vocab_size == 0, "vocab_size must be 0 for local encoder"
        # self.encoder = MiniCPMModel(config)  
        self.encoder = MiniCPMModel_AXInfer(config, "../../VoxCPM.Axera/model_convert/feat_encoder_encoder-axmodels/", 
                                            "MiniCPMForCausalLM", 256, 512, chunk_len=64)

    def forward(self, x):
        """
        x: [B, T, P, D]
        """
        B, T, P, D = x.shape

        x = self.in_proj(x)
        special_tokens = self.special_token.expand(B, T, 1, -1)
        x = torch.cat([special_tokens, x], dim=2)
        x = rearrange(x, "b t p c -> (b t) p c")
        # outputs, _ = self.encoder(x, is_causal=False)

        outputs = []
        for i in range(x.shape[0]):
            output = self.encoder(x[i:i+1])
            outputs.append(output)
        outputs = torch.cat(outputs, 0)
        cls_output = outputs[:, 0, :]

        return rearrange(cls_output, "(b t) c -> b t c", b=B)
