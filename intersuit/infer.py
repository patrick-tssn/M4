import os
from PIL import Image
import numpy as np
import torchaudio
import torch
from decord import VideoReader, cpu
import whisper
# fix seed
torch.manual_seed(0)

from intersuit.model.builder import load_pretrained_model
from intersuit.mm_utils import tokenizer_image_speech_tokens, process_images
from intersuit.constants import IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX

import ChatTTS
chat = ChatTTS.Chat()
chat.load(source='local', compile=True)

import warnings
warnings.filterwarnings("ignore")

model_path = "checkpoints/M4-LongVA-7B-Qwen2"
video_path = "local_demo/assets/water.mp4"
max_frames_num = 16 # you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0", attn_implementation="eager")

# original query
query = "Give a detailed caption of the video as if I am blind."
prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>{query}\n<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer_image_speech_tokens(prompt, tokenizer, IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
pad_token_ids = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
attention_masks = input_ids.ne(pad_token_ids).to(input_ids.device)

# new query
new_query = "How many people in the video?"
new_query = "Okay, I see."
new_query = "Sorry to interrupt."
new_query_pos = 10 # which token encounter the new query
new_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{new_query}\n<|im_end|>\n<|im_start|>assistant\n"
new_input_ids = tokenizer_image_speech_tokens(new_prompt, tokenizer, IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

#video input
vr = VideoReader(video_path, ctx=cpu(0))
total_frame_num = len(vr)
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
frame_idx = uniform_sampled_frames.tolist()
frames = vr.get_batch(frame_idx).asnumpy()
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.bfloat16)


with torch.inference_mode():
    output_ids = model.generate_parallel(input_ids, 
                                attention_mask=attention_masks,
                                images=[video_tensor], 
                                modalities=["video"], 
                                new_query=new_input_ids,
                                new_query_pos=new_query_pos,
                                query_str=query,
                                new_query_str=new_query,
                                tokenizer=tokenizer,
                                **gen_kwargs)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
