import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight


class CogVLMTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)

    def load_hf_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        # attention params
        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_


        if f"transformer.h.{self.layer_num_}.attn.c_attn.weight" in weights:
            qkv_weights = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.weight"]
            split_size = qkv_weights.shape[0] // 3
            q_weights, k_weights, v_weights = torch.split(qkv_weights, split_size, dim=0)
            
            self.q_weight_ = q_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))
            self.k_weight_ = k_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = self._cuda(self.k_weight_.transpose(0, 1))
            self.v_weight_ = v_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = self._cuda(self.v_weight_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.vision_expert_query_key_value.weight" in weights:
            vision_qkv_weights = weights[f"model.layers.{self.layer_num_}.self_attn.vision_expert_query_key_value.weight"]
            split_size = vision_qkv_weights.shape[0] // 3
            vision_q_weights, vision_k_weights, vision_v_weights = torch.split(vision_qkv_weights, split_size, dim=0)
            self.vision_q_weight_ = vision_q_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.vision_q_weight_ = self._cuda(self.vision_q_weight_.transpose(0, 1))
            self.vision_k_weight_ = vision_k_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.vision_k_weight_ = self._cuda(self.vision_k_weight_.transpose(0, 1))
            self.vision_v_weight_ = vision_v_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.vision_v_weight_ = self._cuda(self.vision_v_weight_.transpose(0, 1))
        
        if f"model.layers.{self.layer_num_}.self_attn.language_expert_query_key_value.weight" in weights:
            language_qkv_weights = weights[f"model.layers.{self.layer_num_}.self_attn.language_expert_query_key_value.weight"]
            split_size = language_qkv_weights.shape[0] // 3
            language_q_weights, language_k_weights, language_v_weights = torch.split(language_qkv_weights, split_size, dim=0)
            self.language_q_weight_ = language_q_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.language_q_weight_ = self._cuda(self.language_q_weight_.transpose(0, 1))
            self.language_k_weight_ = language_k_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.language_k_weight_ = self._cuda(self.language_k_weight_.transpose(0, 1))
            self.language_v_weight_ = language_v_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.language_v_weight_ = self._cuda(self.language_v_weight_.transpose(0, 1))


        if f"model.layers.{self.layer_num_}.self_attn.vision_expert_dense.weight" in weights:
            self.vision_expert_dense = weights[f"model.layers.{self.layer_num_}.self_attn.vision_expert_dense.weight"][:,split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.vision_expert_dense = self._cuda(self.vision_expert_dense.transpose(0, 1))
        
        if f"model.layers.{self.layer_num_}.self_attn.language_expert_dense.weight" in weights:
            self.language_expert_dense = weights[f"model.layers.{self.layer_num_}.self_attn.language_expert_dense.weight"][:,split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.language_expert_dense = self._cuda(self.language_expert_dense.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])

        # ffn params
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.language_mlp.up_proj.weight" in weights:
            self.language_up_proj = weights[f"model.layers.{self.layer_num_}.mlp.language_mlp.up_proj.weight"][:, split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.language_up_proj = self._cuda(self.language_up_proj.transpose(0, 1))
        
        if f"model.layers.{self.layer_num_}.mlp.language_mlp.gate_proj.weight" in weights:
            self.language_gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.language_mlp.gate_proj.weight"][:, split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.language_gate_proj = self._cuda(self.language_gate_proj.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.mlp.language_mlp.down_proj.weight" in weights:
            self.language_down_proj = weights[f"model.layers.{self.layer_num_}.mlp.language_mlp.down_proj.weight"][:, split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.language_down_proj = self._cuda(self.language_down_proj.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.mlp.vision_mlp.up_proj.weight" in weights:
            self.vision_up_proj = weights[f"model.layers.{self.layer_num_}.mlp.vision_mlp.up_proj.weight"][:, split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.vision_up_proj = self._cuda(self.vision_up_proj.transpose(0, 1))
        
        if f"model.layers.{self.layer_num_}.mlp.vision_mlp.gate_proj.weight" in weights:
            self.vision_gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.vision_mlp.gate_proj.weight"][:, split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.vision_gate_proj = self._cuda(self.vision_gate_proj.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.mlp.vision_mlp.down_proj.weight" in weights:
            self.vision_down_proj = weights[f"model.layers.{self.layer_num_}.mlp.vision_mlp.down_proj.weight"][:, split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.vision_down_proj = self._cuda(self.vision_down_proj.transpose(0, 1))

        return
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.vision_q_weight_,
                   self.language_q_weight_,
                   self.vision_k_weight_,
                   self.language_k_weight_,
                   self.vision_v_weight_,
                   self.language_v_weight_,
                   self.vision_expert_dense,
                   self.language_expert_dense,
                   self.ffn_norm_weight_,
                   self.language_up_proj,
                   self.language_gate_proj,
                   self.language_down_proj,
                   self.vision_up_proj,
                   self.vision_gate_proj,
                   self.vision_down_proj
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 
