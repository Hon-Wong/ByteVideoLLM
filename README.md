# ByteVideoLLM
Welcome to the official repository for the upcoming work: ByteVideoLLM.

## Release
- [2024/10/21] 🔥 **ByteVideoLLM-14B** achieves 1st performance among 13B/14B models on VideoMME leaderboard, check [[VideoMME](https://video-mme.github.io/home_page.html#leaderboard)]!
- [2024/10/15] 🔥 **ByteVideoLLM-14B** and the inference code are released. Check [[Checkpoint](https://huggingface.co/Hon-Wong/ByteVideoLLM-14B)]!
  
## Overview
ByteVideoLLM aims to strike a fine balance between performance and token consumption in video LLM.

## Features
In the coming weeks, we will be releasing the following components:

- Data: Approximately 1 million high-quality synthetic data points meticulously gathered by our team for Video QA.
- Model Checkpoint: Pre-trained model checkpoints of different scales.
- Training Code: Codebase to replicate the experiments conducted.
We are continuously enhancing the model's performance.

## Inference

We provide the eval script to ensure the reproducibility of our results.

Firstly, you need to process the annotations of LaSOT into [json format](#data-preparation), which is consistent with the format of the training set.

Secondly, Refer to `config.sample_config.yaml`, fill the correct data path into `data.predict.data_fetch`, and then start the command.

```
deepspeed --master_port={PORT} eval/eval.py {YOUR_CONFIG_PATH}
```

### Data Preparation

If you want to use your own data, please process it into the following annotation format

```json
{
    "source": ,
    "id": ,
    "vid": ,
    "metadata": ,
    "vqa": [
      {
        "from": "human", "value": "[YOUR_QUESTION]"
      },
      {
        "from": "gpt", "value": "[YOUR_ANSWER]"
      },
      ...
    ]
}
```

Stay tuned for updates and the release of these valuable resources!
