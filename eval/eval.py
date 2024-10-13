import os
import os.path as osp
import io
import numpy as np
import json
import copy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data.video_llm_data import VideoLLMPredictProcessor
from models.video_llm import VideoLLMModel
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler


@dataclass
class ModelArguments:
    model: Optional[dict] = field(default_factory=dict)


@dataclass
class DataArguments:
    data: Optional[dict] = field(default_factory=dict)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    beta: float = field(default=0.1)
    remove_unused_columns: bool = field(default=False)
    visual_encoder_lr_scale: float = field(default=1.0)


class LocalDataset(Dataset):
    
    def __init__(self, image_folder, anno_path, processor=None):
        self.image_folder = image_folder
        self.processor = processor
        f = open(anno_path, "r")
        self.anns = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        
        item = copy.deepcopy(self.anns[idx])

        if isinstance(item["frames"], list):
            paths = [osp.join(self.image_folder, frame) for frame in item["frames"]]
            frames = [Image.open(p).convert("RGB") for p in paths]
        else:
            path = osp.join(self.image_folder, item["frames"])
            frames = [Image.open(path).convert("RGB")]
        item["frames"] = frames
        item["image_folder"] = self.image_folder
        output = self.processor.transform(item)
        return output


class VideoLLMEvaluator:
    def __init__(self, model, data_args, **kwargs):
        super().__init__(**kwargs)
        self.data_args = edict(data_args.data)
        self.dataloader = self.get_dataloader(self.data_args.predict)
        self.model = model.cuda().eval()

    def get_dataloader(self, config) -> DataLoader:
        
        df_config = config.data_fetch
        dp_config = config.data_preprocess
        dp_config.update({"meta_keys": ["source", "id", "question", "gt"]})
        processor = VideoLLMPredictProcessor(**dp_config)
        
        dataset = LocalDataset(
            image_folder=df_config.image_folder, 
            anno_path=df_config.anno_path, 
            processor=processor
        )

        num_workers = df_config.num_workers
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=df_config.batch_sizes[0], sampler=sampler,
                        collate_fn=processor.batch_transform, num_workers=num_workers, shuffle=False)
        return loader

    def predict(self, save_path):
        f = open(save_path, "a") 
        for index, batch in tqdm(enumerate(self.dataloader)):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()

            outputs = self.model.predict(batch)
            if "frames" in batch:
                batch.pop("frames")

            outputs.update({"vid": batch['vid'], 'id': batch['id'], 'question': batch['question']})
            for key in outputs:
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = batch[key].cpu().numpy().tolist()

            dict_of_list = outputs
            list_of_dict = [{key: values[i] for key, values in dict_of_list.items()} for i in range(len(list(dict_of_list.values())[0]))]
            
            for line in list_of_dict:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    import sys
    torch.distributed.init_process_group(backend='nccl')
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(sys.argv[-1], allow_extra_keys=True)
    model_params = model_args.model
    model_params["pretrained"] = training_args.output_dir
    model = VideoLLMModel(**model_params)
    evaluater = VideoLLMEvaluator(model=model, data_args=data_args)
    save_filename = osp.basename(edict(data_args.data).predict.data_fetch.anno_path)
    save_folder = osp.join(training_args.output_dir, "infer_results")
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, save_filename)
    evaluater.predict(save_path=save_path)

