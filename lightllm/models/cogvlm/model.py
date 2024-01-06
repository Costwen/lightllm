import json
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.cogvlm.layer_infer.pre_layer_infer import CogVLMMultimodalPreLayerInfer
from lightllm.models.cogvlm.layer_infer.transformer_layer_infer import CogVLMTransformerLayerInfer
from lightllm.models.cogvlm.layer_weights.transformer_layer_weight import CogVLMTransformerLayerWeight
from lightllm.models.cogvlm.infer_struct import CogVLMInferStateInfo

# Warp of the origal tokenizer
class CogVLMTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # (image_size // patch_size) ** 2 + 2: (490 // 14) ** 2 + 2 = 1227
        self.image_length = 1227

    # only change the impl of the encode func:
    def encode(self, prompt):
        # split prompt by <image>, and merge parts by [pad_id] * 1227
        ids_chunks = self.tokenizer(prompt).input_ids
        input_ids = ids_chunks[:1] + [self.tokenizer.pad_token_id] * self.image_length + ids_chunks[1:]

        return {"input_ids": input_ids, "lengths": self.image_length, "offsets": [1]}

    def __getattr__(self, name):
        if name != 'encode':
            return getattr(self.tokenizer, name)
        return self.encode



class CogVLMTpPartModel(LlamaTpPartModel):

    # infer class
    pre_layer_infer_class = CogVLMMultimodalPreLayerInfer
    transformer_layer_infer_class = CogVLMTransformerLayerInfer
    transformer_weight_class = CogVLMTransformerLayerWeight
    infer_state_class = CogVLMInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
