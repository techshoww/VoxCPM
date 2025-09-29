import os
import numpy as np
import onnxruntime as ort
from axengine import InferenceSession, axclrt_provider_name, axengine_provider_name
from ml_dtypes import bfloat16
# from scipy.special import softmax
from transformers import AutoTokenizer
import gc
import dill 
import torch
from torch import nn
from functools import wraps

class InferEngine:
    def __init__(self, path, provider_options=None):
        self.path = path

        model_type = os.path.splitext(path)[1]

        engine = None
        if model_type == ".onnx":
            engine = ort.InferenceSession
            self.session = engine(path)
        elif model_type == ".axmodel":
            engine = InferenceSession
            self.session = engine(path, provider_options=provider_options)
        else:
            raise NotImplementedError(f"Not supported model type: {model_type}")

        assert engine is not None

        

    def run(self, outputnames, inputs, shape_group=None):
        if shape_group is None:
            outputs = self.session.run(outputnames, inputs)
        else:
            outputs = self.session.run(outputnames, inputs, shape_group=shape_group)
        return outputs

    def _unload(self):
        if isinstance(self.session, InferenceSession):
            self.session._sess._unload()
        else:
            del self.session

class AxModelInferStatic:
    def __init__(self, axmodel_path, provider_options=None):
        self.session = InferEngine(axmodel_path, provider_options=provider_options)

    def __call__(self, inputs, shape_group=None):

        if shape_group is None:
            outputs = self.session.run(None, inputs)
        else:
            outputs = self.session.run(None, inputs, shape_group=shape_group)
        return outputs

    def _unload(self):
        self.session._unload()

class AxModelInferDynamic:
    def __init__(self, axmodel_path, provider_options=None):
        self.axmodel_path = axmodel_path
        self.provider_options = provider_options
    def __call__(self, inputs, shape_group=None):

        session = InferEngine(self.axmodel_path, provider_options=self.provider_options)
        if shape_group is None:
            outputs = session.run(None, inputs)
        else:
            outputs = session.run(None, inputs, shape_group=shape_group)

        return outputs


class AxModelInfer:
    def __init__(self, axmodel_path, run_dynamic=False, provider_options=None):
        print(f"init model:{axmodel_path}")
        if run_dynamic:
            self.model = AxModelInferDynamic(axmodel_path, provider_options=provider_options)
        else:
            self.model = AxModelInferStatic(axmodel_path, provider_options=provider_options)

    def __call__(self, inputs, shape_group=None):
        try:
            outputs = self.model(inputs, shape_group)
        except Exception as e:
            if hasattr(self.model, "axmodel_path"):
                print(f"axmodel_path:{self.model.axmodel_path}")
            print(e)
            raise e

        return outputs

    def run(self, outputs, inputs, shape_group=None):
        return self(inputs, shape_group)

    def _unload(self):
        self.model._unload()

def lazyforward(func):
    @wraps(func)  
    def wrapper(self, *args, **kwargs):
        if self.lazy_load:
            self._load()

        result = func(self, *args, **kwargs)  
        
        if self.lazy_load:
            self._unload()

        return result
    return wrapper

class AxLMInfer:
    def __init__(
        self, cfg, model_dir, model_name, prefill_len, lastN, chunk_len=-1, run_dynamic=False, lazy_load=True, provider_options=None
    ):

        self.cfg = cfg
        self.model_dir = model_dir
        self.model_name = model_name
        self.prefill_len = prefill_len
        self.chunk_len = chunk_len if chunk_len>0 else prefill_len
        self.lastN = lastN
        self.run_dynamic = run_dynamic
        self.lazy_load = lazy_load and (not run_dynamic)

        self.num_hidden_layers = self.cfg.num_hidden_layers
        self.hidden_size = self.cfg.hidden_size
        self.provider_options = provider_options
        if not self.lazy_load:
            self._load()
        
    def _load(self):
        self.prefill_decoder_sessins = []
        for i in range(self.num_hidden_layers):
            
            session = AxModelInfer(
                f"{self.model_dir}/{self.model_name}_p{self.chunk_len}_l{i}_together.axmodel",
                self.run_dynamic,
                provider_options = self.provider_options
            )
            self.prefill_decoder_sessins.append(session)
        
        self.post_process_session = AxModelInfer(
            f"{self.model_dir}/{self.model_name}_post.axmodel", self.run_dynamic, provider_options=self.provider_options
        )

    def _unload(self):
        for session in self.prefill_decoder_sessins:
            session._unload()
        self.post_process_session._unload()

        self.prefill_decoder_sessins = None
        self.post_process_session = None
    
    def forward(self,*args, **kwargs):
        raise NotImplementedError("forward funciton was not implemented")

    @lazyforward
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

        

