import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import random
import math


class DynamicSpatialPooling(nn.Module):
    def __init__(self, 
            num_features=1408, 
            in_token_num=576,
            out_token_num=144,
            min_out_token_num=16,
            max_video_tokens=2048,
            fix_randam=False,
            **kwargs
        ):
        super().__init__()
        self.in_token_num = in_token_num
        self.out_token_num = out_token_num
        self.min_out_token_num = min_out_token_num
        self.max_video_tokens = max_video_tokens
        self.fix_randam = fix_randam
        self.hidden_size = num_features 

    def forward(self, image_embeds, n_frames):
        image_embeds_list = image_embeds.split(n_frames, dim=0)
        # Compute temporal tokens as the mean along the time axis
        ret_tokens = []
        for image_embeds_per_video in image_embeds_list:
            video_raw_token = image_embeds_per_video
            frame_num = len(video_raw_token)

            if self.training and not self.fix_randam:
                token_num = random.randint(self.min_out_token_num, self.out_token_num)
            else:
                token_num = self.min_out_token_num
                
            upper_limit = self.max_video_tokens // frame_num
            if self.training and not self.fix_randam:
                token_num = min(upper_limit, token_num)
            else:
                token_num = min(upper_limit, self.out_token_num)
            if token_num == self.in_token_num:
                ret_tokens.append(video_raw_token)
                continue
            hw = int(video_raw_token.shape[-2] ** 0.5)
            x = rearrange(video_raw_token, "b (h w) d -> b d h w", h=hw, w=hw)
            out_hw = max(1, int(math.sqrt(token_num)))
            x = F.adaptive_avg_pool2d(x, (out_hw, out_hw))
            x = rearrange(x, "b d h w -> b (h w) d")
            ret_tokens.append(x)
        return ret_tokens

if __name__ == '__main__':
    model = DynamicSpatialPooling()
    model.training = False
    x = torch.randn((32, 576, 1408))
    y = model(x, [32])
    for i in y:
        print(i.shape)