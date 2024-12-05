#origin
# Copyright 2024 Zhenwei Shao and MILVLG team.
# Licensed under the Apache License, Version 2.0.

# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from flashsloth.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, LEARNABLE_TOKEN, LEARNABLE_TOKEN_INDEX
from flashsloth.model.pooling import build_pooling

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)
            self.pooling = build_pooling('attention', input_dim=1152, pooling_size=3, device=self.vision_tower.device, dtype=self.vision_tower.dtype)
            # self.pooling = build_pooling('average', pooling_size=3, device=self.vision_tower.device)
            # hack
            # [Edited by zhenwei - 2024-02-02 20:36]
            is_meta = getattr(nn.Linear(1, 1, bias=False).weight, 'is_meta', False)
            if is_meta:
                fake_dict = {}
                for n, p in self.mm_projector.named_parameters():
                    fake_dict[n] = torch.zeros_like(p, device='cpu')
                from transformers.modeling_utils import _load_state_dict_into_meta_model
                _load_state_dict_into_meta_model(
                    self.mm_projector,
                    fake_dict,
                    fake_dict.keys(),  # left for now but could be removed, see below
                    '',
                    fake_dict.keys(),
                )
                # self.mm_projector.to('cuda' if torch.cuda.is_available() else 'cpu')

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        self.pooling = build_pooling('attention', input_dim=1152, pooling_size=3, device=self.vision_tower.device, dtype=self.vision_tower.dtype)
        # self.pooling = build_pooling('average', pooling_size=3, device=self.vision_tower.device)


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features_origin = image_features
        image_features = self.get_model().pooling(image_features)
        image_features = self.get_model().mm_projector(image_features)  
        return image_features, image_features_origin

    def extract_question_token_indices(self, labels, batch_indices, image_token_len, modal, version="phi2"):
        """
        extract indices of all question tokens in the input sequence.
        """
        if len(batch_indices) < 20:
            version = "phi2"
        else:
            version = "plain"
            
        if version == "plain":
            question_token_ranges = []
            for idx, (cur_labels, cur_batch_indices, num ) in enumerate(zip(labels, batch_indices, modal)):
                question_token_ranges.append([(image_token_len + 1, batch_indices[idx][0])])
        else:
            question_token_ranges = []  
            for _, (cur_labels, cur_batch_indices, num ) in enumerate(zip(labels, batch_indices, modal)):
                cur_question_ranges = []
                #first question token is after the image token and before the first learnable token
                if num == 1:#single modal
                    first_question_start = 32
                elif num==2:    #multi modal
                    first_question_start = 32 + image_token_len + 1
                if len(cur_batch_indices) == 0:
                    print("cur_batch_indices", cur_batch_indices)
                    first_question_end = first_question_start
                else:
                    first_question_end = cur_batch_indices[0] 
                if first_question_end < first_question_start:
                    print("first_question_start", first_question_start)
                    print("first_question_end", first_question_end)
                    print(batch_indices)
                # assert first_question_end >= first_question_start
                cur_question_ranges.append((first_question_start, first_question_end))
                #subsequent question tokens are after the answer token and before the next learnable token
                learnable_idx_counter = 1
                for i in range(len(cur_labels) - 1):
                    if cur_labels[i] != IGNORE_INDEX and cur_labels[i + 1] == IGNORE_INDEX:
                        question_start = i + 3
                        try:
                            question_end = cur_batch_indices[learnable_idx_counter]
                        except IndexError:
                            print(f"learnable_idx_counter {learnable_idx_counter} exceeds cur_batch_indices length {len(cur_batch_indices)}")
                            break
                        learnable_idx_counter += 1
                        cur_question_ranges.append((question_start, question_end))
                if len(cur_question_ranges) > len(cur_batch_indices):
                    cur_question_ranges = cur_question_ranges[:len(cur_batch_indices)]
                elif len(cur_question_ranges) < len(cur_batch_indices):
                    last_range = cur_question_ranges[-1] if cur_question_ranges else (0, 0)
                    while len(cur_question_ranges) < len(cur_batch_indices):
                        cur_question_ranges.append(last_range)
                question_token_ranges.append(cur_question_ranges)
        return question_token_ranges

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, learnable_tokens, model_version='phi2'
    ):
        dot_tokens = self.get_model().embed_tokens(torch.full((learnable_tokens.size(0),), 764, device=input_ids.device, dtype=input_ids.dtype))
        learnable_tokens = learnable_tokens + dot_tokens
        modal = [2]
        vision_tower = self.get_vision_tower()
        if model_version == 'phi2':
            if past_key_values is not None:
                target_shape = past_key_values[0][0].shape[2] + 1
                attention_mask = torch.ones(
                    (attention_mask.shape[0], target_shape),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                return input_ids[:, -1:], position_ids, attention_mask, past_key_values, None, labels, [], None, learnable_tokens.shape[0], modal, None
            if vision_tower is None or images is None or input_ids.shape[1] == 1:
                return input_ids, None, None, past_key_values, None, None, [], None, learnable_tokens.shape[0], modal, None
        else: 
            if vision_tower is None or images is None or input_ids.shape[1] == 1:
                if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                    target_shape = past_key_values.seqlen_offset + 1
                    attention_mask = torch.cat((attention_mask, torch.ones(
                        (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                return input_ids, position_ids, attention_mask, past_key_values, None, labels, [], None, learnable_tokens.shape[0], modal

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, image_features_origin = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features_origin = torch.split(image_features_origin, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
            image_features_origin = [x.flatten(0, 1).to(self.device) for x in image_features_origin]
            image_features = torch.stack(image_features, dim=0)
            image_features_origin = torch.stack(image_features_origin, dim=0)
        else:
            image_features, image_features_origin = self.encode_images(images)
            image_features = image_features.to(self.device)
            image_features_origin = image_features_origin.to(self.device)

        batch_indices = []
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        print(image_features.shape)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        modal =[]
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_learnables = (cur_input_ids == LEARNABLE_TOKEN_INDEX).sum()
            num_specials = num_images + num_learnables
            image_token_indices_origin = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() 
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] 
            learnable_token_indices = torch.where(cur_input_ids == LEARNABLE_TOKEN_INDEX)[0].tolist() #[43]
            all_special_indices = sorted(image_token_indices_origin+ learnable_token_indices)
            image_token_len = image_features.shape[1] - 1 
            learnable_token_len = learnable_tokens.shape[0] -1  
            offset = 0
            new_indices= []
            for i, idx in enumerate(all_special_indices):
                if idx in learnable_token_indices:
                    new_indices.append(idx + offset) 
                if idx in image_token_indices:
                    offset += image_token_len
                if idx in learnable_token_indices:
                    offset += learnable_token_len
            batch_indices.append(new_indices)
            special_token_indices = sorted(image_token_indices + learnable_token_indices)
            cur_input_ids_no_special = []
            cur_labels = labels[batch_idx]
            cur_labels_no_special = []
            for i in range(len(special_token_indices) - 1):
                cur_input_ids_no_special.append(cur_input_ids[special_token_indices[i]+1:special_token_indices[i+1]])
                cur_labels_no_special.append(cur_labels[special_token_indices[i]+1:special_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_no_special]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_no_special))
            cur_input_embeds_no_special = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_specials + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_special[i])
                cur_new_labels.append(cur_labels_no_special[i])
                if i < num_specials:
                    if special_token_indices[i+1] in image_token_indices:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif special_token_indices[i+1] in learnable_token_indices:
                        cur_new_input_embeds.append(learnable_tokens)
                        cur_new_labels.append(torch.full((learnable_tokens.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    else:
                        ValueError("token indices error")
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            if num_images == 0 :
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds = torch.cat([cur_new_input_embeds, cur_image_features[0:0]], dim=0)
                cur_image_idx += 1
                modal.append(1)
            else:
                modal.append(2)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        question_token_ranges = self.extract_question_token_indices(new_labels, batch_indices, image_token_len+1, modal)

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, batch_indices, image_features_origin, learnable_tokens.shape[0], modal, question_token_ranges

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if tokenizer.convert_tokens_to_ids(LEARNABLE_TOKEN) == tokenizer.unk_token_id:
            tokenizer.add_tokens([LEARNABLE_TOKEN], special_tokens=True)
            print(f"Added {LEARNABLE_TOKEN} to tokenizer.")
        else:
            print(f"{LEARNABLE_TOKEN} already exists in the tokenizer.")
        token_id = tokenizer.convert_tokens_to_ids(LEARNABLE_TOKEN)
        print(f"Token ID for {LEARNABLE_TOKEN}: {token_id}")
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
