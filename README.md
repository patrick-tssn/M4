<h1 align="center">Multi-modal Multiplexing Modeling</h1>
<p align="center">
    <a href="https://arxiv.org/abs/xxxx.xxxxx">
            <img alt="Build" src="http://img.shields.io/badge/cs.CV-arXiv%3Axxxx.xxxxx-B31B1B.svg">
    </a>
    <a href="https://huggingface.co/ColorfulAI/M4-7B">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-M47B-yellow">
    </a>
    <a href="https://huggingface.co/datasets/ColorfulAI/M4-IT">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Dataset-M4IT-yellow">
    </a>
</p>

<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://<CONFERENCE>) -->

![image](assets/framework.png)

## Updates

<!-- - [] Paper Release, check it on [Arxiv](https://arxiv.org/pdf/xxxx.xxxxx.pdf).  -->

- **First Release [M4](https://github.com/patrick-tssn/M4)**. M4 enables multiplexed modeling capabilities for a video language model at minimal cost.

**Table of Contents**

- [M4](#m4)
  - [Introduction](#introduction)
  - [M4-IT](#dataset)
- [Train](#training)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Start Training](#training-1)
  - [Usage](#usage)

## M4

### Introduction

We introduce Multimodal Multiplexing Modeling (M4), a framework that enhances real-time interactive reasoning with minimal fine-tuning on pre-trained MLLMs.

- **M4-IT Dataset**: A synthetic instruction finetuning dataset with components interleaved image-text instruction, noise instruction, and stop instruction.
- **M4 Model**: Enhances proactive response generation, assesses new queries against noise, by enabling parallel decoding.

### M4-IT Dataset

Building on the [LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data), we crafted a small video-free synthetic instruction finetuning dataset, M4-IT, with the assistance of GPT-4o. M4-IT comprises four components:

- the original instruction, which is a data replay from the instruction data of our base model
- interleaved image-text instruction, which is created by reordering the question and image components of the original instruction
- noise instruction, where GPT-4 is prompted to automatically generate statements that do not require a response
- stop instruction, where GPT-4 is prompted to generate stop phrases for the stop instruction

In addition, to assist with audio instruction tuning, we convert user queries into audio using [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), with a randomly selected [VoiceAssistant](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) as a prompt.

**Data Statistics**

The M4-IT dataset comprises a total of 9,963 instructions. The distribution across different categories is as follows:

| Category   | Count |
| ---------- | ----- |
| Original   | 2,624 |
| Interleave | 2,376 |
| Noise      | 2,563 |
| Stop       | 2,500 |

Data sample

```json
    {
        "id": "000000240632",
        "image": "000000240632.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n"
            },
            {
                "from": "human",
                "value": "<speech>\n" # provide the bounding box coordinates of the region that the given sentence describes
            },
            {
                "from": "gpt",
                "value": "[0.280,0.194,0.628,0.824]"
            },
            {
                "from": "human",
                "value": "<speech>\n" # Could I stop you for a second?
            },
            {
                "from": "gpt",
                "value": "<|im_end|>"
            }
        ],
        "speech": [
            "000000240632_0.wav",
            "000000240632_1.wav"
        ]
    },
```

If you are interested in the process of the construction of audio instruction, you can refer to the scripts in `preprocess/tts`

## Training

### Installation

This codebase is tested on CUDA 11.8 and A800-80G.
```bash
conda create -n intersuit python=3.10 -y && conda activate intersuit
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e "intersuit/.[train]"
pip install packaging &&  pip install ninja && pip install flash-attn==2.5.0 --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```


#### Data Preparation

Download [M4-IT](https://huggingface.co/datasets/ColorfulAI/M4-IT) and organize it in the following format. To enhance audio instruction-following performance, you may also download [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) and sample a portion of this dataset based on your computational resources.

```
intersuit/inputs                  
    ├── images/ # images
      └── llava-next/
        ├── ...
        └── xxxx.jpg
    ├── speech/
      ├── voiceassistant/
        ├── ...
        └── xxxx.wav
      └── interinst/
        ├── ...
        └── xxxx.wav
    └── texts/
      ├── voiceassistant.json
      ├── m4-it-qwen.json
      └── m4-it-qwen-audio.json
```

#### Training

Our training logic is no different from the original visual instruction tuning.


```bash
cd intersuit
# finetune on m4-it
bash scripts/finetune_m4.sh
# finetune on m4-it-audio
bash scripts/finetune_m4_audio.sh
```

#### Usage

(i) proactive reasoning




(ii) multiplexing modeling



## Citation

If you find our work helpful, please consider citing it.

```bibtex
@article{omnimmi,
    title={OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts},
    author={Wang, Yuxuan and Wang, Yueqian and Chen, Bo and Wu, Tong and Zhao, Dongyan and Zheng, Zilong},
    journal={arxiv},
    year={2025}
}
```
