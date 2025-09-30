import torch
import torch.nn as nn
import os
if os.getenv("AX_INFER", "false").lower() == "true":
    from ...npu_infer.utils_axinfer import AxModelInfer

class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim: int = 64, scale: int = 9):
        super().__init__()

        if os.getenv("AX_INFER", "false").lower() != "true":
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.latent_dim = latent_dim
            self.scale = scale

            self.in_proj = nn.Linear(in_dim, latent_dim)
            self.out_proj = nn.Linear(latent_dim, out_dim)
        else:
            axmodel_dir = os.getenv("AXMODEL_DIR")
            self.model = AxModelInfer(f"{axmodel_dir}/axmodels/fsq_layer.axmodel")
    
    def forward(self, hidden):
        if os.getenv("AX_INFER", "false").lower() != "true":
            hidden = self.in_proj(hidden)
            hidden = torch.tanh(hidden)

            if self.training:
                quantized = torch.round(hidden * self.scale) / self.scale
                hidden = hidden + (quantized - hidden).detach()
            else:
                hidden = torch.round(hidden * self.scale) / self.scale

            return self.out_proj(hidden)
        else:
            device = hidden.device
            if len(hidden.shape)==3:
                outputs = []
                for i in range(hidden.shape[1]):
                    input = {"hidden":hidden[:,i].detach().cpu().numpy()}
                    output = self.model(input)[0]
                    output = torch.from_numpy(output)
                    output = output.unsqueeze(1)
                    outputs.append(output)
                
                outputs = torch.cat(outputs, 1).to(device)
                return outputs
            else:
                
                input = {"hidden":hidden.detach().cpu().numpy()}
                output = self.model(input)[0]
                output = torch.from_numpy(output).to(device)
                return output