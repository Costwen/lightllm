import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.cogvlm.layer_weights.transformer_layer_weight import CogVLMTransformerLayerWeight
from lightllm.models.cogvlm.infer_struct import CogVLMInferStateInfo

class CogVLMTransformerLayerInfer(LlamaTransformerLayerInfer):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return
    
    def _get_qkv(self, input_emb, cache_k, cache_v, infer_state: CogVLMInferStateInfo, layer_weight:CogVLMTransformerLayerWeight):
        vision_token_mask, language_token_mask = infer_state.mask
        
        vision_q = torch.mm(input_emb.view(-1, self.embed_dim_), layer_weight.vision_q_weight_)
        vision_k = torch.mm(input_emb.view(-1, self.embed_dim_), layer_weight.vision_k_weight_)
        print(vision_k.shape)
        print(cache_k.shape)
        vision_v = torch.mm(input_emb.view(-1, self.embed_dim_), layer_weight.vision_v_weight_)
        language_q = torch.mm(input_emb.view(-1, self.embed_dim_), layer_weight.language_q_weight_)
        language_k = torch.mm(input_emb.view(-1, self.embed_dim_), layer_weight.language_k_weight_)
        language_v = torch.mm(input_emb.view(-1, self.embed_dim_), layer_weight.language_v_weight_)
        q = vision_q.view(-1, self.tp_q_head_num_, self.head_dim_) * vision_token_mask + language_q.view(-1, self.tp_q_head_num_, self.head_dim_) * language_token_mask

        rotary_emb_fwd(q, infer_state.position_cos, infer_state.position_sin)


        torch.mm(input_emb.view(-1, self.embed_dim_), layer_weight.k_weight_,
                    out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        torch.mm(input_emb.view(-1, self.embed_dim_), layer_weight.v_weight_,
                    out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return q, cache_k, cache_v
    
    def _get_o(self, input, infer_state:CogVLMInferStateInfo, layer_weight:CogVLMTransformerLayerWeight)->torch.Tensor:
        vision_token_mask, language_token_mask = infer_state.mask
        vison_o_tensor = torch.einsum('bln,nd->bld', input, layer_weight.vision_o_weight_)
        language_o_tensor = torch.einsum('bln,nd->bld', input, layer_weight.language_o_weight_)
        o_tensor = vison_o_tensor * vision_token_mask + language_o_tensor * language_token_mask
        return o_tensor


    def _ffn(self, input, infer_state:CogVLMInferStateInfo, layer_weight:CogVLMTransformerLayerWeight)->torch.Tensor:
        vision_token_mask, language_token_mask = infer_state.mask
        vision_gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.vision_gate_proj)
        torch.nn.functional.silu(vision_gate_out, inplace=True)
        vision_up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.vision_up_proj)
        vision_ffn1_out = vision_gate_out * vision_up_out
        vision_gate_out, vision_up_out = None, None
        vision_ffn2_out = torch.mm(vision_ffn1_out, layer_weight.vision_down_proj)

        language_gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.language_gate_proj)
        torch.nn.functional.silu(language_gate_out, inplace=True)
        language_up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.language_up_proj)
        language_ffn1_out = language_gate_out * language_up_out
        language_gate_out, language_up_out = None, None
        language_ffn2_out = torch.mm(language_ffn1_out, layer_weight.language_down_proj)
        ffn2_out = vision_ffn2_out * vision_token_mask + language_ffn2_out * language_token_mask

        return ffn2_out
