import os
import torch
import numpy as np
from ml_dtypes import bfloat16
import math
from .utils_axinfer import AxModelInfer, AxLMInfer


class MiniCPMModel_AXInfer(AxLMInfer):
    def __init__(
        self, cfg, model_dir, model_name, prefill_len, lastN, chunk_len=-1, run_dynamic=False, lazy_load=False, provider_options=None
    ):

        super().__init__(cfg, model_dir, model_name, prefill_len, lastN, chunk_len, run_dynamic, lazy_load, provider_options=provider_options)

        self.embeds = np.load(f"{model_dir}/model.embed_tokens.weight.npy")
        self.total_prefill = prefill_len
        self.chunk_len = chunk_len

        kv_dim = (
            cfg.hidden_size
            // cfg.num_attention_heads
            * cfg.num_key_value_heads
        )
        self.k_caches = [
            np.zeros((1, lastN, kv_dim), dtype=bfloat16)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.v_caches = [
            np.zeros((1, lastN, kv_dim), dtype=bfloat16)
            for _ in range(cfg.num_hidden_layers)
        ]

        self.speech_token_size = 6561

    def embed_tokens(self, input_ids):
        print("input_ids",input_ids.shape)
        device = input_ids.device
        input_ids = input_ids.detach().cpu().numpy()

        assert len(input_ids.shape)==2, f"not support shape:{input_ids.shape}"
        ret = []
        for ids in input_ids:
            emb =  np.take(self.embeds, ids, axis=0)
            ret.append(emb)
        
        ret = np.stack(ret)
        print("ret",ret.shape)
        return torch.from_numpy(ret.astype(np.float32)).to(device)

    def forward(self, input_embeds, is_causal=True):
        device = input_embeds.device
        input_embeds = input_embeds.detach().cpu().numpy().astype(bfloat16)

        token_len = input_embeds.shape[1]
        indices = np.zeros((1, self.prefill_len), dtype=np.uint32)
        position_ids = np.arange(token_len).reshape(1,-1)
        indices[:, 0:token_len] = position_ids.astype(np.uint32)
        mask = np.zeros((1, self.prefill_len, self.prefill_len)) - 65536
        data = np.zeros((1, self.prefill_len, self.hidden_size)).astype(bfloat16)
        data[:, 0:token_len] = input_embeds
        if is_causal:
            for i in range(token_len):
                mask[:, i, : i + 1] = 0
        else:
            mask[:, 0:token_len, 0:token_len] = 0

        mask = mask.astype(bfloat16)
        chunk_num = math.ceil(token_len / self.chunk_len)
        for i in range(self.num_hidden_layers):
            if self.chunk_len <= 0:
                input_feed = {
                    "K_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                    "V_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=1)

                self.k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
                self.v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
                data[:, 0:token_len] = outputs[2][:, :token_len, :]
            else:
                layer_output = []
                for ck in range(chunk_num):
                    
                    gid = ck + 1
                    if ck==0:
                        input_feed = {
                            "K_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                            "V_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                            "indices": indices[:, 0:self.chunk_len],
                            "input": data[:, 0:self.chunk_len],
                            "mask": mask[:, 0:self.chunk_len, 0:self.chunk_len],
                        }
                        outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=gid)
                        self.k_caches[i][:, :self.chunk_len, :] = outputs[0][:, :self.chunk_len, :]
                        self.v_caches[i][:, :self.chunk_len, :] = outputs[1][:, :self.chunk_len, :]

                    else:
                        input_feed = {
                            "K_cache": self.k_caches[i][:, :ck*self.chunk_len, :],
                            "V_cache": self.v_caches[i][:, :ck*self.chunk_len, :],
                            "indices": indices[:, ck*self.chunk_len:(ck+1)*self.chunk_len],
                            "input": data[:, ck*self.chunk_len:(ck+1)*self.chunk_len],
                            "mask": mask[:, ck*self.chunk_len:(ck+1)*self.chunk_len, 0:(ck+1)*self.chunk_len],
                        }
                        outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=gid)
                        self.k_caches[i][:, ck*self.chunk_len:(ck+1)*self.chunk_len, :] = outputs[0][:, :self.chunk_len, :]
                        self.v_caches[i][:, ck*self.chunk_len:(ck+1)*self.chunk_len, :] = outputs[1][:, :self.chunk_len, :]

                    layer_output.append(outputs[2][:, :self.chunk_len, :])
                
                data = np.concatenate(layer_output, axis=1)

        ret = []
        for i in range(token_len):
            post_out = self.post_process_session(
                {"input": data[:, i : i+1, :]}
            )[1]
            ret.append(post_out)
        ret = np.concatenate(ret, axis=1)
        return torch.from_numpy(ret.astype(np.float32)).to(device)

    def forward_step(self, input_embeds, position_id):
        device = input_embeds.device
        input_embeds = input_embeds.unsqueeze(1)
        input_embeds = input_embeds.detach().cpu().numpy().astype(bfloat16)
        # start_ids = np.max(indices) + 1
        
        mask = np.zeros((1, 1, self.lastN + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, : self.lastN] -= 65536
        mask[:, :, :position_id] = 0
        # for start_indice in range(np.max(indices) + 1, self.lastN + 1):

        # if self.prefill_len > 0 and start_indice < token_len:
        #     continue
        indices = np.array([position_id], np.uint32).reshape((1, 1))
        # start_ids += 1

        data = input_embeds.astype(bfloat16)
        
        for i in range(self.cfg.num_hidden_layers):
            input_feed = {
                "K_cache": self.k_caches[i],
                "V_cache": self.v_caches[i],
                "indices": indices,
                "input": data,
                "mask": mask,
            }

            outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=0)

            self.k_caches[i][:, position_id, :] = outputs[0][:, :, :]
            self.v_caches[i][:, position_id, :] = outputs[1][:, :, :]
            data = outputs[2]
        
        post_out = self.post_process_session({"input": data})[1]
    
        return torch.from_numpy(post_out.astype(np.float32)).squeeze(1).to(device)
        