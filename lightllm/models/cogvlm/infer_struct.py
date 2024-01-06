import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo

class CogVLMInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None

    def init_some_extra_state(self, model, input_ids : torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy() # b x 1
            position_ids = []
            for i in range(len(b_seq_len_numpy)):
                position_id = [0, 1] + [2] * 1225 + [3]
                position_id += list(np.arange(4, b_seq_len_numpy[i]-1224))
                position_ids.append(np.array(position_id))

            vision_token_mask = torch.zeros_like(input_ids, dtype=torch.float16).view(-1, 1)
            vision_token_mask[2:1227] = 1.0
            language_token_mask = (1.0 - vision_token_mask).to(torch.float16)
            self.mask = (vision_token_mask, language_token_mask)

            position_ids = torch.from_numpy(np.concatenate(position_ids, axis=0)).cuda()
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1225
            vision_token_mask = torch.zeros_like(input_ids, dtype=torch.float16).view(-1, 1)

            vision_token_mask[2:1227] = 1.0
            language_token_mask = (1.0 - vision_token_mask).to(torch.float16)
            self.mask = (vision_token_mask, language_token_mask)
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.b_req_idx[0], 0].item()
            # b_loc[0, max_len_in_batch - 1].item()

        return

