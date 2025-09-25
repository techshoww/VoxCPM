import torch
from ..minicpm4 import MiniCPMModel, MiniCPM4Config
import torch.nn as nn
import math
from ...npu_infer.utils_lm import MiniCPMModel_AXInfer 
from ...npu_infer.utils_axinfer import AxModelInfer
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        # half_dim = self.dim // 2
        # emb = math.log(10000) / (half_dim - 1)
        # emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=device) * -emb)
        emb = scale * x.unsqueeze(1) * self.emb.unsqueeze(0).to(x.dtype).to(device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    def forward_zero(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        emb = scale * x.unsqueeze(1) * self.emb.unsqueeze(0).to(x.dtype).to(device)
        emb = torch.cat((torch.zeros_like(emb), torch.ones_like(emb)), dim=-1)
        
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int = None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class VoxCPMLocDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        config: MiniCPM4Config,
        in_channels: int = 64,
    ):
        super().__init__()
        # self.in_channels = in_channels
        # self.out_channels = in_channels
        self.config = config

        # self.in_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        # self.cond_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        # self.out_proj = nn.Linear(config.hidden_size, self.out_channels, bias=True)

        # self.time_embeddings = SinusoidalPosEmb(config.hidden_size)
        # self.time_mlp = TimestepEmbedding(
        #     in_channels=config.hidden_size,
        #     time_embed_dim=config.hidden_size,
        # )
        # self.delta_time_mlp = TimestepEmbedding(
        #     in_channels=config.hidden_size,
        #     time_embed_dim=config.hidden_size,
        # )

        self.part1 = AxModelInfer("../../VoxCPM.Axera/model_convert/axmodels/locdit.part1.axmodel")
        self.part3 = AxModelInfer("../../VoxCPM.Axera/model_convert/axmodels/locdit.part3.axmodel")
        assert config.vocab_size == 0, "vocab_size must be 0 for local DiT"
        # self.decoder = MiniCPMModel(config)
        self.decoder = MiniCPMModel_AXInfer(config, "../../VoxCPM.Axera/model_convert/feat_decoder_estimator_decoder-axmodels/", 
                                            "MiniCPMForCausalLM", 256, 512, chunk_len=64)


    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of inputs
        mu: (N, C) tensor of hidden embedding
        t: (N,) tensor of diffusion timesteps
        cond: (N, C, T') tensor of prefix conditions
        dt: (N,) used for mean velocity (may be supported in the future...)
        """
        x = self.forward_part1(x, mu, t, cond, dt)
        hidden = self.forward_part2(x)
        output = self.forward_part3(hidden)
        return output
    
    def forward_part1(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of inputs
        mu: (N, C) tensor of hidden embedding
        t: (N,) tensor of diffusion timesteps
        cond: (N, C, T') tensor of prefix conditions
        dt: (N,) used for mean velocity (may be supported in the future...)
        """
        # x = self.in_proj(x.transpose(1, 2).contiguous())

        # cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        # prefix = cond.size(1)

        # t = self.time_embeddings(t).to(x.dtype)
        # t = self.time_mlp(t)
        # dt = self.time_embeddings.forward_zero(dt).to(x.dtype)
        # dt = self.delta_time_mlp(dt)
        # t = t + dt

        # x = torch.cat([(mu + t).unsqueeze(1), cond, x], dim=1)

        device = x.device
        input = {"x":x.detach().cpu().numpy(), "mu":mu.detach().cpu().numpy(),
                    "t":t.detach().cpu().numpy(), "cond":cond.detach().cpu().numpy()}
        y = self.part1(input)[0]
        return torch.from_numpy(y).to(device)

    def forward_part2(
        self,
        x: torch.Tensor
    ):
        
        hidden = []
        for i in range(x.shape[0]):
            output = self.decoder(x[i:i+1])
            hidden.append(output)
        hidden = torch.cat(hidden, 0)

        return hidden
    
    def forward_part3(
        self,
        hidden: torch.Tensor
    ):
        # prefix = 2
        # hidden = hidden[:, prefix + 1 :, :]
        # hidden = self.out_proj(hidden)

        # return hidden.transpose(1, 2).contiguous()

        device = hidden.device
        input = {"hidden": hidden.detach().cpu().numpy()}
        y = self.part3(input)[0]
        return torch.from_numpy(y).to(device)