import json
import torch
import copy
import torch
import os
import torch.nn as nn
from functools import partial, wraps
from easydict import EasyDict as edict
from torch.utils.checkpoint import checkpoint
from utils.dtype import DTYPE_MAPPING
from models.visual_encoder import build_vision
from models.llm import build_llm
from models.adapter import build_adapter, build_projector
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from utils.io import partial_load_from_checkpoints
from transformers import PreTrainedModel, PretrainedConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.utils import logging  
import torch.distributed as dist
from deepspeed.sequence.layer import DistributedAttention
import deepspeed
from transformers.integrations.deepspeed import (  # noqa
    is_deepspeed_zero3_enabled,
    # is_deepspeed_sp_enabled,
)


logging.set_verbosity_info()  # Turn on this for debug mode
logger = logging.get_logger("transformers")


global_configs = None


default_generate_params = {
    "do_sample": True,
    "keyword": None
}


default_lora_config = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


default_evaluation_config = {
    "do_eval": True,
    "method": "cocoevalcap",
    "resep": True,
    "visualize": {
        "enable": False,
        "type": "gif",
        "visualize_root": "./visualizations"
    }
}


class VideoLLMConfig(PretrainedConfig):
    model_type = "videollm"


class VideoLLMModel(PreTrainedModel): 
    supports_gradient_checkpointing = True
    def __init__(
        self,
        llm: edict = {
            "freeze_llm": True,
            "dtype": "fp16",
            "use_lora": False,
            "lora_config": default_lora_config
        },
        use_visual: bool = True,  # flag of using visual_encoder/adapter forwarding
        use_flash_attention: bool = False,
        visual: edict = {"freeze_vit": False},
        adapter: edict = {},
        projector: edict = {"type": "linear"},
        dtype: str = None,
        pretrained: str = "",  # partial pretrained model path
        ckpt_rename_parameters: dict = {"module.": ""},  # key mapping dictionary
        valid_prefix=None,
        log_source_loss: bool = False,
        generate_params: edict = default_generate_params,
        evaluation: edict = default_evaluation_config,
        config: PretrainedConfig = VideoLLMConfig(),
        add_special_token: bool = False,
        lazy_load: bool = False,
    ):
        super().__init__(config) # for PretrainedModel
        self.llm_config = edict(llm) 
        self.lazy_load = lazy_load
        self.use_visual = use_visual
        self.visual = edict(visual)
        self.torch_dtype = dtype
        self.adapter_config = edict(adapter)
        self.projector_config = edict(projector)
        self.pretrained = pretrained
        self.ckpt_rename_parameters = ckpt_rename_parameters
        self.valid_prefix = valid_prefix
        self.log_source_loss = log_source_loss
        self.generate_params = edict(generate_params)
        self.evaluation = edict(evaluation)
        self.use_flash_attention = use_flash_attention
        self.add_special_token = add_special_token
        self.setup()  # freeze params in PreTrainedModel.__init__() might be overrided
        self.enable_input_require_grads = self.llm.enable_input_require_grads
        self.config = self.llm.config  # this is a hack to make deepspeed happy. FIXME

    def setup(self):
        
        # setup llm
        self._setup_llm()
        
        # setup vision model
        if self.use_visual:
            # DO NOT call deepspeed.Init() here
            self._setup_visual_encoder()
            self._setup_adapter()
            self._setup_projector()

            if self.add_special_token:
                embed_std = 1 / torch.sqrt(torch.tensor(self.llm.config.hidden_size))
                self.special_token = nn.Parameter(torch.randn(self.llm.config.hidden_size) * embed_std)
            
            if self.adapter_config.freeze_adapter:
                for name, param in self.llm_proj.named_parameters():
                    param.requires_grad = False               
                self.llm_proj = self.llm_proj.eval()
                self.llm_proj.train = lambda self, mode=True: self
                logger.info("freeze llm_proj")

        # use peft-lora training
        if self.llm_config.use_lora:
            from peft import LoraConfig, get_peft_model 
            lora_config = {**default_lora_config, **self.llm_config.lora_config}
            lora_config = LoraConfig(**lora_config)
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.logger.info_trainable_parameters()

        # load pretrained model
        if self.pretrained:
            local_path = self.pretrained
            logger.info(f"Load pretrained model from: {local_path}")
            if not self.lazy_load:
                state_dict = partial_load_from_checkpoints(
                    local_path, 
                    map_location="auto", ckpt_rename_parameters=self.ckpt_rename_parameters, valid_prefix=self.valid_prefix
                )
                msg = self.load_state_dict(state_dict, strict=False)
                logger.info("State Loaded.")
                logger.info(f"Missing keys: {msg.missing_keys}")
                logger.info(f"Unexpected keys: {msg.unexpected_keys}")
            else:
                from safetensors.torch import load
                file_paths = partial_load_from_checkpoints(
                    local_path, 
                    map_location="auto", ckpt_rename_parameters=self.ckpt_rename_parameters, valid_prefix=self.valid_prefix
                )
                for file_path in file_paths:
                    print(f"loading checkpoint from {file_path}")
                    with open(file_path, "rb") as f:
                        data = f.read()
                    loaded = load(data)
                    msg = self.load_state_dict(loaded, strict=False)
                logger.info("State Loaded.")

        if self.torch_dtype:
            logger.info(f"Converting model to {DTYPE_MAPPING[self.torch_dtype]}.")
            self.to(DTYPE_MAPPING[self.torch_dtype])
            logger.info("Done.")
        

    def _setup_visual_encoder(self):       
        self.visual_encoder = build_vision(**self.visual)
        if self.visual.freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False               
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = lambda self, mode=True: self
            logger.info("freeze vision encoder")

    def _setup_llm(self):
        
        # text encoder & load pretrained model
        self.llm, self.tokenizer = build_llm(**self.llm_config)

        # freeze llm if needed
        if self.llm_config.use_lora or self.llm_config.freeze_llm:
            logger.info(
                f"llm.use_lora={self.llm_config.use_lora}, llm.freeze_llm={self.llm_config.freeze_llm}")
            for name, param in self.llm.named_parameters():
                param.requires_grad = False
        
    def _setup_adapter(self):
        self.adapter = build_adapter(**self.adapter_config)
        
        if self.adapter_config.freeze_adapter:
            for name, param in self.adapter.named_parameters():
                param.requires_grad = False               
            self.adapter = self.adapter.eval()
            self.adapter.train = lambda self, mode=True: self
            logger.info("freeze adapter")
    
    def _setup_projector(self):
        if not self.projector_config.input_hidden_size:  
            hidden_size = self.adapter.hidden_size
        else:
            hidden_size = self.projector_config.input_hidden_size

        self.llm_proj = build_projector(
            self.projector_config.type,
            input_hidden_size=hidden_size, 
            output_hidden_size=self.llm.config.hidden_size
        )

    def _encode_vision(self, images, n_frames):
        
        if images.size(0) > 0:
            image_embeds = self.visual_encoder(images)
        else:
            # NOTE: This is a hacking for deepspeed training
            # we feeding a dummy image tensor (1, 3, H, W)
            # into vision_encoder when training a pure-text batch
            images = images.new_zeros((1, *images.shape[1:]))
            image_embeds = self.visual_encoder(images)[0:0]
        adapter_out = self.adapter(image_embeds, n_frames=n_frames)
        vision_embeds = [self.llm_proj(feature) for feature in adapter_out]
        attention_mask = [torch.ones(feature.size()[:-1], dtype=torch.long).to(feature.device) for feature in vision_embeds]
        vision_targets = [torch.ones(feature.size(), dtype=torch.long).to(feature.device).fill_(-100) for feature in attention_mask]

        return vision_embeds, attention_mask, vision_targets

    def _concat_embedding(self, vision_encode_out, batch, vision_placeholder_index, left_padding=False):
        """ concat vision and text
        """

        vision_embeds, vision_atts, vision_targets = vision_encode_out

        input_embeds = []
        attention_mask = []
        targets = []

        for cur_batch_idx, cur_input_ids in enumerate(batch["input_ids"]):
            cur_vision_embeds = vision_embeds[cur_batch_idx]
            cur_vision_attn = vision_atts[cur_batch_idx]
            cur_vision_targets = vision_targets[cur_batch_idx]
            cur_attn_masks = batch["attention_mask"][cur_batch_idx]

            image_token_indices = torch.where(cur_input_ids == vision_placeholder_index)[0]
            cur_image_num = len(image_token_indices)
            image_token_indices = list(image_token_indices) + [cur_input_ids.shape[0]]

            cur_input_embeds = []
            cur_attention_mask = []
            cur_target = []

            # convert text before 1st <image> to embedding
            image_token_index = image_token_indices[0]

            cur_input_embeds.append(
                self.llm.get_input_embeddings()(cur_input_ids[:image_token_index]),
            )
            cur_attention_mask.append(
                cur_attn_masks[:image_token_index],
            )
            if "labels" in batch:
                cur_target.append(
                    batch["labels"][cur_batch_idx, :image_token_index],
                )

            if batch.get("vison_placeholder_mode", 0) == 1:
                assert cur_image_num <= 1, "multiple video input is not supported"
                cur_vision_embeds = cur_vision_embeds.unsqueeze(0)
                cur_vision_attn = cur_vision_attn.unsqueeze(0)
                cur_vision_targets = cur_vision_targets.unsqueeze(0)
            assert cur_image_num == len(cur_vision_embeds), \
                f"Size mismatch! cur_image_num: {cur_image_num}, len(cur_vision_embeds): {len(cur_vision_embeds)} {len(cur_vision_embeds)} \
                    in {batch['prompt'][cur_batch_idx]} & {batch['gt'][cur_batch_idx]} & {batch['input_ids'][cur_batch_idx]}"
            # convert each <image> xxx group into embedding
            text_embedding = self.llm.get_input_embeddings()(cur_input_ids.relu())
            for i in range(0, cur_image_num):
                image_token_index = image_token_indices[i]
                cur_input_embeds.extend([
                    cur_vision_embeds[i],
                    text_embedding[image_token_index+1:image_token_indices[i+1]]
                ])
                cur_attention_mask.extend([
                    cur_vision_attn[i],
                    cur_attn_masks[image_token_index+1:image_token_indices[i+1]]
                ])
                if "labels" in batch:
                    cur_target.extend([
                        cur_vision_targets[i],
                        batch["labels"][cur_batch_idx, image_token_index+1:image_token_indices[i+1]],
                    ])

            input_embeds.append(torch.cat(cur_input_embeds))
            attention_mask.append(torch.cat(cur_attention_mask))
            if "labels" in batch:
                targets.append(torch.cat(cur_target))

        # padding
        n_tokens = [embed.shape[0] for embed in input_embeds]

        max_token = max(n_tokens)

        for i in range(len(input_embeds)):
            if max_token > n_tokens[i]:
                self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                pad_token = torch.tensor([self.pad_id] * (max_token - n_tokens[i]))
                pad_embedding = self.llm.get_input_embeddings()(pad_token.to(batch["attention_mask"][i].device))
                pad_attention = torch.zeros(pad_embedding.shape[0], dtype=torch.long).to(batch["attention_mask"][i].device)
                pad_targets = torch.ones(pad_attention.size(), dtype=torch.long).to(batch["attention_mask"][i].device).fill_(-100)

                if left_padding:
                    input_embeds[i] = torch.cat([pad_embedding, input_embeds[i]])
                    attention_mask[i] = torch.cat([pad_attention, attention_mask[i]])
                    if "labels" in batch:
                        targets[i] = torch.cat([pad_targets, targets[i]])
                else:
                    input_embeds[i] = torch.cat([input_embeds[i], pad_embedding])
                    attention_mask[i] = torch.cat([attention_mask[i], pad_attention])
                    if "labels" in batch:
                        targets[i] = torch.cat([targets[i], pad_targets])

        inputs_embeds = torch.stack(input_embeds, dim=0).type(self.llm.dtype)
        attention_mask = torch.stack(attention_mask, dim=0)

        if len(targets) > 0:
            targets = torch.stack(targets, dim=0)

        return inputs_embeds, attention_mask, targets


    def forward(self, **batch):
        if self.use_visual:
            # get vision token
            vision_placeholder_index = batch.pop("vision_placeholder_index")
            
            # get vision features
            images, n_frames = batch["frames"], batch["n_frames"]
            # print("input_ids: ", batch["input_ids"].device)
            vision_encode_out = self._encode_vision(images, n_frames)
            # print("input_ids2: ", batch["input_ids"].device)
            inputs_embeds, attention_mask, targets = self._concat_embedding(
                vision_encode_out, batch, vision_placeholder_index)

        else:
            inputs_embeds = self.llm.get_input_embeddings()(batch["input_ids"])
            attention_mask = batch["attention_mask"]
            targets = batch["labels"]
        
        # input to llm
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True,
        )
        if batch.get("dpo", False):
            return outputs, targets
        else:
            return outputs

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _enable_gradient_checkpointing(self):
        for model in (self.visual_encoder, self.llm):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def predict(self, batch):
        
        extra_generate_params = dict(
            eos_token_id=self.tokenizer.eos_token_id,
            **self.generate_params
        )

        if self.use_visual:

            with torch.cuda.amp.autocast(
                enabled=(self.device != torch.device("cpu"))
            ):
                # get vision token
                vision_placeholder_index = batch.pop("vision_placeholder_index")

                # get vision features
                images, n_frames = batch["frames"], batch["n_frames"]
                vision_encode_out = self._encode_vision(images, n_frames)

                inputs_embeds, attention_mask, _ = self._concat_embedding(
                    vision_encode_out, batch, vision_placeholder_index, left_padding=False)

        else:
            inputs_embeds = self.llm.get_input_embeddings()(batch["input_ids"])
            attention_mask = batch["attention_mask"]

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **extra_generate_params
        )
        
        # parse result text
        output_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        num_out_per_input = len(output_text) // len(batch["gt"])
        # loop over the batches
        prompts, predicts, gts = [], [], []
        for i, (pred, gt, prompt) in enumerate(zip(output_text, batch["gt"], batch["prompt"])):
            prompts.append(prompt)
            pred_this_sample = output_text[i*num_out_per_input:(i+1)*num_out_per_input]
            if len(pred_this_sample) == 1:
                pred_this_sample = pred_this_sample[0]
            predicts.append(pred_this_sample)
            gts.append(gt)
        return dict(
            prompt = prompts,
            predict = predicts,
            gt = gts
        )
