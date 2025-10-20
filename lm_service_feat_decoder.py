import os
import torch
import numpy as np
from ml_dtypes import bfloat16
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import argparse
from voxcpm.npu_infer.utils_lm import MiniCPMModel_AXInfer
from voxcpm.model.voxcpm import VoxCPMConfig
# --- Pydantic 模型定义 ---
class InputIdsRequest(BaseModel):
    input_ids: List[int]
    shape: List[int]

class EmbedTokensResponse(BaseModel):
    status: str = "success"
    output: List[int]    # bfloat16 的 uint16 位表示
    shape: List[int]

class InputEmbedsRequest(BaseModel):
    input_embeds: List[int] # bfloat16 的 uint16 位表示
    shape: List[int]
    is_causal: bool = True

class ForwardResponse(BaseModel):
    status: str = "success"
    output: List[int]     # bfloat16 的 uint16 位表示
    shape: List[int]

class ForwardStepRequest(BaseModel):
    input_embeds: List[int] # bfloat16 的 uint16 位表示
    shape: List[int]
    position_id: int

class ForwardStepResponse(BaseModel):
    status: str = "success"
    output: List[int]     # bfloat16 的 uint16 位表示
    shape: List[int]


# --- FastAPI 应用 ---
app = FastAPI(title="MiniCPM Model Inference API (1D Input with FastAPI)", version="1.0.0")

# 初始化模型实例 (需要提供正确的 cfg, model_dir, model_name 等参数)
config = VoxCPMConfig.model_validate_json(open(os.path.join("../../VoxCPM-0.5B/", "config.json")).read())
decoder_config = config.lm_config.model_copy(deep=True)
decoder_config.hidden_size = config.dit_config.hidden_dim
decoder_config.intermediate_size = config.dit_config.ffn_dim
decoder_config.num_attention_heads = config.dit_config.num_heads
decoder_config.num_hidden_layers = config.dit_config.num_layers
decoder_config.kv_channels = config.dit_config.kv_channels
decoder_config.vocab_size = 0

model_instance = MiniCPMModel_AXInfer(decoder_config, f"../../VoxCPM.Axera/model_convert/feat_decoder_estimator_decoder-axmodels/", 
                                                "MiniCPMForCausalLM", 256, 512, chunk_len=64)


@app.post("/embed_tokens", response_model=EmbedTokensResponse)
async def run_embed_tokens(request: InputIdsRequest):
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        input_ids_np = np.array(request.input_ids, dtype=np.int32)
        input_ids_tensor = torch.from_numpy(input_ids_np.reshape(request.shape))

        output_tensor = model_instance.embed_tokens(input_ids_tensor)

        # 1. 转换为 numpy float32
        output_np = output_tensor.detach().cpu().numpy()
        # 2. 转换为 bfloat16
        output_bf16 = output_np.astype(bfloat16)
        # 3. 获取 uint16 位表示
        output_uint16 = output_bf16.view(np.uint16)
        # 4. 展平并返回
        output_shape = list(output_uint16.shape)
        output_flat_list = output_uint16.flatten().tolist()

        return EmbedTokensResponse(output=output_flat_list, shape=output_shape)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embed_tokens: {str(e)}")

@app.post("/forward", response_model=ForwardResponse)
async def run_forward(request: InputEmbedsRequest):
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # 1. 将 JSON int 列表转换为 numpy uint16
        input_embeds_uint16_np = np.array(request.input_embeds, dtype=np.uint16)
        # 2. 重塑
        input_embeds_reshaped = input_embeds_uint16_np.reshape(request.shape)
        # 3. 将 uint16 位表示转换为 bfloat16
        input_embeds_bf16 = input_embeds_reshaped.view(bfloat16)
        # 4. 将 bfloat16 转换为 torch float32 (模型期望的输入格式)
        input_embeds_tensor = torch.from_numpy(input_embeds_bf16.astype(np.float32))

        # 调用模型的 forward 方法 (接收 torch float32)
        output_tensor = model_instance.forward(input_embeds_tensor, is_causal=request.is_causal)

        # 5. 转换输出: torch float32 -> numpy float32 -> bfloat16 -> uint16 -> int list
        output_np = output_tensor.detach().cpu().numpy()
        output_bf16 = output_np.astype(bfloat16) # 模型内部是 bfloat16 计算，这里转回来
        output_uint16 = output_bf16.view(np.uint16)
        output_shape = list(output_uint16.shape)
        output_flat_list = output_uint16.flatten().tolist()

        return ForwardResponse(output=output_flat_list, shape=output_shape)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during forward pass: {str(e)}")

@app.post("/forward_step", response_model=ForwardStepResponse)
async def run_forward_step(request: ForwardStepRequest):
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # 1. 将 JSON int 列表转换为 numpy uint16
        input_embeds_uint16_np = np.array(request.input_embeds, dtype=np.uint16)
        # 2. 重塑
        input_embeds_reshaped = input_embeds_uint16_np.reshape(request.shape)
        # 3. 将 uint16 位表示转换为 bfloat16
        input_embeds_bf16 = input_embeds_reshaped.view(bfloat16)
        # 4. 将 bfloat16 转换为 torch float32 (模型期望的输入格式)
        input_embeds_tensor = torch.from_numpy(input_embeds_bf16.astype(np.float32))

        # 调用模型的 forward_step 方法 (接收 torch float32)
        output_tensor = model_instance.forward_step(input_embeds_tensor, request.position_id)

        # 5. 转换输出: torch float32 -> numpy float32 -> bfloat16 -> uint16 -> int list
        output_np = output_tensor.detach().cpu().numpy()
        output_bf16 = output_np.astype(bfloat16) # 模型内部是 bfloat16 计算，这里转回来
        output_uint16 = output_bf16.view(np.uint16)
        output_shape = list(output_uint16.shape)
        output_flat_list = output_uint16.flatten().tolist()

        return ForwardStepResponse(output=output_flat_list, shape=output_shape)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during forward step: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MiniCPM FastAPI Inference Server')
    parser.add_argument('--host', type=str, default='localhost', help='Host address to bind to')
    parser.add_argument('--port', type=int, default=9004, help='Port number to bind to')
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)