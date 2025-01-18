import argparse
import copy
import math
import warnings
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta
from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
from PIL import Image
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from transformers import AutoConfig

from transformers import TextIteratorStreamer, TextStreamer
from threading import Thread

import socket
from threading import Thread

import requests

from transformers.cache_utils import (
    DynamicCache,
    SinkCache,
    StaticCache,
    SlidingWindowCache,
    QuantoQuantizedCache,
    QuantizedCacheConfig,
)

from torch.nn.attention.flex_attention import create_mask, create_block_mask
import torch


def transition_attention_mask_pt(b, h, q_idx, kv_idx, prefix_length, channel, device):
    q_idx = torch.tensor(q_idx, dtype=torch.long, device=device)
    kv_idx = torch.tensor(kv_idx, dtype=torch.long, device=device)
    
    # Validate indices
    if q_idx.max() >= len(channel) or kv_idx.max() >= len(channel):
        raise ValueError("Index out of bounds. Ensure q_idx and kv_idx are within the valid range.")
    
    # Allow attention to prefix positions
    prefix_mask = kv_idx.unsqueeze(0) < prefix_length
    # Allow attention within the same channel
    channel_tensor = torch.tensor(channel, dtype=torch.long, device=device)
    block_mask = channel_tensor[q_idx].unsqueeze(-1) == channel_tensor[kv_idx]
    # Causal mask to prevent attending to future positions
    causal_mask = q_idx.unsqueeze(-1) >= kv_idx
    # Combine masks: prefix or block should be true, and must satisfy causal constraints
    combined_mask = (prefix_mask | block_mask) & causal_mask
    return combined_mask


torch.backends.cuda.matmul.allow_tf32 = True

warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

try:
    from intersuit.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
    )
    from intersuit.conversation import conv_templates, SeparatorStyle
    from intersuit.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
        KeywordsStoppingCriteria,
    )
    from intersuit.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug(
        "LongVA is not installed. Please install LongVA to use this model.\nError: %s" % e
    )
    raise NotImplementedError("no valid longva")

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


# Function to start a socket server to receive new queries continuously
def start_query_server(host="127.0.0.1", port=65432, buffer_size=1024):
    """
    Starts a socket server to listen for new queries from a client continuously.
    """
    def listen_for_queries():
        # Create and configure the server socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((host, port))
            server_socket.listen(5)  # Allow up to 5 simultaneous connections
            print(f"Query server running on {host}:{port}... Waiting for client connections.")
            
            while True:  # Keep the server running indefinitely
                conn, addr = server_socket.accept()  # Accept a client connection
                print(f"Connected by {addr}")
                # Handle client requests in a new thread
                client_thread = Thread(target=handle_client, args=(conn, buffer_size), daemon=True)
                client_thread.start()

    def handle_client(conn, buffer_size):
        """
        Handle communication with a connected client.
        """
        global NEW_QUERY
        with conn:
            while True:
                try:
                    data = conn.recv(buffer_size)  # Receive query data from client
                    if not data:
                        break  # Exit the loop if no data is received (client disconnected)
                    query = data.decode("utf-8").strip()
                    print(f"Received new query from client: {query}")
                    # Update the global variable with the new query
                    NEW_QUERY = query
                    # Send acknowledgment to the client
                    conn.sendall(b"Query received.\n")
                except Exception as e:
                    print(f"Error handling client: {e}")
                    break

    # Use a separate thread to keep the server running in the background
    server_thread = Thread(target=listen_for_queries, daemon=True)
    server_thread.start()

# Initialize global variable to store the new query
NEW_QUERY = None

# Start the query server in the background
start_query_server()


class LongVA:
    """
    LongVA Model
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/LongVA-7B-DPO",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[
            bool
        ] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "average",
        token_strategy: Optional[
            str
        ] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame in the context
        video_decode_backend: str = "decord",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }

        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        model_name = (
            model_name
            if model_name is not None
            else get_model_name_from_path(pretrained)
        )

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    pretrained,
                    None,
                    model_name,
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    pretrained,
                    None,
                    model_name,
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert (
            self.batch_size_per_gpu == 1
        ), "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu
                    * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )
                eval_logger.info(
                    "Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0"
                )

            if (
                accelerator.distributed_type == DistributedType.FSDP
                or accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(
                f"Using {accelerator.num_processes} devices with tensor parallelism"
            )
            self._rank = 0
            self._word_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        print(f"spare_frames: {spare_frames.shape}")
        return spare_frames  # (frames, height, width, channels)

    # def stream_generate_until(self, requests: dict, gen_kwargs: dict):
    def output(self, requests: dict, gen_kwargs: dict):

        print("start generation")
        
        question_input = []

        visuals = requests["visuals"]
        context = requests["context"]
        task_type = requests["task_type"]
        
        print(f"################# requests ######################\n{requests}")
        print(f"################# gen_kwargs ######################\n{gen_kwargs}")

        if task_type == "text":
            image_tensor = None

        # encode, pad, and truncate contexts for this batch
        elif task_type == "image":  # For image task
            image_tensor = process_images(visuals, self._image_processor, self._config)
            if type(image_tensor) is list:
                image_tensor = [
                    _image.to(dtype=torch.float16, device=self.device)
                    for _image in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

        elif task_type == "video":  # For video task
            image_tensor = []
            max_frames = gen_kwargs.get("sample_frames", self.max_frames_num)
            if "sample_frames" in gen_kwargs:
                gen_kwargs.pop("sample_frames")

            try:
                if self.video_decode_backend == "decord":
                    frames = self.load_video(visuals, max_frames)
                
                elif self.video_decode_backend == "opencv":
                    video_data = []
                    cap = cv2.VideoCapture('http://10.1.101.4:1234')
                    for i in range(8):
                        ret, frame = cap.read()
                        if not ret:
                            raise ValueError(f"video error at 10.1.101.4:1234")
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_data.append(frame)
                    cap.reslease()
                    video_data = np.stack(video_data, dim=0) # (T, H, W, C)

                frames = (
                    self._image_processor.preprocess(frames, return_tensors="pt")[
                        "pixel_values"
                    ]
                    # .half()
                    .bfloat16()
                    .cuda()
                )
                image_tensor.append(frames)
            except Exception as e:
                eval_logger.error(f"Error {e} in loading video")
                image_tensor = None

            task_type = "video"

        if (
            image_tensor is not None
            and len(image_tensor) != 0
            and DEFAULT_IMAGE_TOKEN not in context
        ):
            """
            Three senarios:
            1. No image, and there for, no image token should be added.
            2. image token is already specified in the context, so we don't need to add it.
            3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
            4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
            """
            if task_type == "image":
                image_tokens = (
                    [DEFAULT_IMAGE_TOKEN] * len(visuals)
                    if isinstance(visuals, list)
                    else [DEFAULT_IMAGE_TOKEN]
                )
            elif task_type == "video":
                image_tokens = (
                    [DEFAULT_IMAGE_TOKEN] * len(frames)
                    if self.token_strategy == "multiple"
                    else [DEFAULT_IMAGE_TOKEN]
                )

            image_tokens = " ".join(image_tokens)
            question = image_tokens + "\n" + context
            # question = context + image_tokens + "\n"
        else:
            question = context

        # This is much safer for llama3, as we now have some object type in it
        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()

        for prev_conv in requests["prev_conv"]:
            conv.append_message(conv.roles[0], prev_conv[0])
            conv.append_message(conv.roles[1], prev_conv[1])

        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)

        prompt_question = conv.get_prompt()
        question_input.append(prompt_question)

        # preconfigure gen_kwargs with defaults
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        input_ids_list = [
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            for prompt in question_input
        ]
        pad_token_ids = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        input_ids = self.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_token_ids
        ).to(self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        
        # print("##"*20)
        # print(stop_str)
        # print(stopping_criteria)
        # print("##"*20)
        
        if task_type == "image":
            gen_kwargs["image_sizes"] = [
                visual.size for visual in visuals
            ]  # (width, height)
        elif task_type == "video":
            gen_kwargs["modalities"] = ["video"]
            gen_kwargs["stopping_criteria"] = [stopping_criteria]
            self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
            self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

        # These steps are not in LLaVA's original code, but are necessary for generation to work
        # TODO: attention to this major generation step...
        if "image_aspect_ratio" in gen_kwargs.keys():
            gen_kwargs.pop("image_aspect_ratio")

        max_context_length = getattr(self.model.config, "max_position_embeddings", 2048)
        num_image_tokens = (
            question.count(DEFAULT_IMAGE_TOKEN)
            * self.model.get_vision_tower().num_patches
        )

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
        )

        gen_kwargs["max_new_tokens"] = min(
            gen_kwargs["max_new_tokens"],
            max_context_length - input_ids.shape[-1] - num_image_tokens,
        )

        if gen_kwargs["max_new_tokens"] < 1:
            print("yield error")
            # yield json.dumps(
            #     {
            #         "text": question
            #         + "Exceeds max token length. Please start a new conversation, thanks.",
            #         "error_code": 0,
            #     }
            # ).encode() + b"\0"
            return

        
        # thread = Thread(
        #     target=self.model.generate,
        #     kwargs=dict(
        #         inputs=input_ids,
        #         attention_mask=attention_masks,
        #         pad_token_id=pad_token_ids,
        #         images=image_tensor,
        #         use_cache=self.use_cache,
        #         streamer=streamer,
        #         **gen_kwargs,
        #     ),
        # )
        # thread.start()
        # generated_text = ""
        # for new_text in streamer:
        #     generated_text += new_text
        #     if generated_text.endswith(stop_str):
        #         generated_text = generated_text[: -len(stop_str)]
        #     yield json.dumps(
        #         {"text": generated_text, "error_code": 0}
        #     ).encode() + b"\0"
        
        with torch.inference_mode():
            # cont = self.model.generate(
            #     input_ids,
            #     attention_mask=attention_masks,
            #     pad_token_id=pad_token_ids,
            #     images=image_tensor,
            #     use_cache=self.use_cache,
            #     **gen_kwargs,
            # )
            streamer = TextStreamer(self.tokenizer)
            new_query = "Can yo8u describe the video?"
            new_query = "Oh. This video is about cooking."
            # new_query = "Oh well, it is what it is. No point in overthinking it."
            new_query = "How to make a cup of wine?"
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], new_query)
            conv.append_message(conv.roles[1], None)
            new_prompt_question = [conv.get_prompt()]
            new_input_ids_list = [
                tokenizer_image_token(
                    new_query, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                for new_query in new_prompt_question
            ]
            new_input_ids = self.pad_sequence(
                new_input_ids_list, batch_first=True, padding_value=pad_token_ids
            ).to(self.device)
            
            speeches = None
            speech_lengths = None
            
            cont = self.model.generate_parallel(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                use_cache=self.use_cache,
                new_query=new_input_ids,
                new_query_str=new_query,
                query_str=question,
                tokenizer=self.tokenizer,
                **gen_kwargs,
            )
            
            # greedy decode
            if image_tensor is not None:
                if speeches is None:
                    (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, image_tensor, gen_kwargs["modalities"], image_sizes=gen_kwargs["image_sizes"])
                else:
                    (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_av(inputs, position_ids, attention_mask, None, None, image_tensor, gen_kwargs["modalities"], image_sizes=gen_kwargs["image_sizes"], speeches=speeches, speech_lengths=speech_lengths)
            else:
                inputs_embeds = self.model.get_model().embed_tokens(inputs)
            
            new_query = None
            new_query_str = None
            query_str = None
            # Update: Retrieve new query from client
            global NEW_QUERY
            if NEW_QUERY is not None:
                print(f"Using new query from client: {NEW_QUERY}")
                new_query_str = NEW_QUERY
                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], new_query)
                conv.append_message(conv.roles[1], None)
                new_prompt_question = [conv.get_prompt()]
                new_input_ids_list = [
                    tokenizer_image_token(
                        new_query, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    for new_query in new_prompt_question
                ]
                new_query = self.pad_sequence(
                    new_input_ids_list, batch_first=True, padding_value=pad_token_ids
                ).to(self.device)
                NEW_QUERY = None  # Reset after using
            
            
            if query_str is not None:
                query_str = query_str.split("<image>\n")[-1]
                print(f"Human: {query_str}")
            if new_query is not None:
                new_inputs_embeds = self.model.get_model().embed_tokens(new_query)
            else:
                new_inputs_embeds = None
            bsz, seq = inputs_embeds.shape[:-1]

            # get channel ids 10: system; 14: 13: system+template
            prefix_length = 13 + 32 * 144 + 1 # system + video # prefix prompt + frame
            channel = [0] * prefix_length + [1] * (seq - prefix_length)
            # # construct new attention masks by flexattention
            def prefix_mask(b, h, q_idx, kv_idx):
                return kv_idx <= prefix_length
            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            def transition_attention_mask(b, h, q_idx, kv_idx):
                prefix_mask = kv_idx <= prefix_length
                causal_mask = q_idx >= kv_idx
                channel_tensor = torch.tensor(channel, dtype=torch.long, device=new_inputs_embeds.device)
                block_mask = channel_tensor[q_idx] == channel_tensor[kv_idx]
                return (prefix_mask | block_mask) & causal_mask

            
            past_key_values = DynamicCache()
            max_cache_length = past_key_values.get_max_length()
            cache_position = torch.arange(seq, dtype=torch.int64, device=inputs_embeds.device)
            max_new_tokens = gen_kwargs["max_new_tokens"]
            for idx in range(max_new_tokens):
                
                # discern noise
                if idx == -1:
                    print()
                    print(f"Human: {new_query_str}")
                    # # encounter new query 
                    # new_attention_mask = create_mask(transition_attention_mask, 1, 1, bsz, seq)
                    
                    outputs_ids = torch.cat([outputs_ids, new_query], dim=-1)
                    
                    ori_length = attention_mask.shape[1]
                    
                    # # (1) naive mask: mask previous input query
                    # attention_mask[:, prefix_length:] = 0 # mask prev attention
                    # cur_length = attention_mask.shape[1] - prefix_length
                    # # print(attention_mask.shape)
                    # attention_mask = torch.cat([attention_mask[:,:-1], attention_mask.new_ones((new_inputs_embeds.shape[0], new_inputs_embeds.shape[1]))], dim=-1)
                    # (2) parallel mask
                    seq = ori_length + new_inputs_embeds.shape[1]
                    channel += [channel[-1]+1] * new_inputs_embeds.shape[1]
                    dtype_min = torch.finfo(new_inputs_embeds.dtype).min
                    
                    # # 1. flexattention
                    # attention_mask_4d = create_mask(transition_attention_mask, 1, 1, seq, seq)
                    # attention_mask_4d = attention_mask_4d[:, :, -new_inputs_embeds.shape[1]:, :]
                    # attention_mask_4d = torch.where(attention_mask_4d, torch.tensor(0.0, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype), torch.tensor(dtype_min, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype))
                    # 2. torch implementation
                    attention_mask_4d = transition_attention_mask_pt(1, 1, list(range(seq)), list(range(seq)), prefix_length, channel, new_inputs_embeds.device).unsqueeze(0).unsqueeze(0)
                    attention_mask_4d = attention_mask_4d[:, :, -new_inputs_embeds.shape[1]-1:, :]
                    
                    
                    attention_mask_4d = torch.where(
                        attention_mask_4d,
                        torch.tensor(0.0, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype),
                        torch.tensor(dtype_min, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype)
                    )
                    # attention_mask_4d = torch.where(attention_mask_4d, torch.tensor(0.0, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype), torch.tensor(dtype_min, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype))
                    
                    cache_position = torch.arange(cache_position[-1], cache_position[-1]+new_inputs_embeds.shape[1]+1, dtype=torch.int64, device=inputs_embeds.device)
                    
                    # print("*"*24)
                    # print(new_inputs_embeds.shape)
                    # print(attention_mask_4d.shape)
                    # print("*"*24)
                    
                    outputs = super().forward(
                        attention_mask=attention_mask_4d,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        inputs_embeds=torch.cat([inputs_embeds, new_inputs_embeds],dim=1),
                        use_cache=True,
                    )
                    
                    
                    
                    temperature = 1
                    next_prob = torch.softmax(outputs.logits[:,-1]/temperature, dim=-1)
                    next_entropy = -torch.sum(next_prob*torch.log(next_prob+1e-5), dim=-1)
                    next_threshold = 0.09
                    next_alpha = 0.3
                    noise_threshold = torch.minimum(
                        torch.ones_like(next_entropy) * next_threshold,
                        torch.exp(-next_entropy) * next_alpha
                    )
                    noise_prob = next_prob[:, 151644] # |<im_start>|
                    # HACK batchsize must be 1
                    if noise_prob[0] > noise_threshold[0]:
                        print("Assistant: ", end='', flush=True)
                        # # (1) naive mask: if noise: continue generate, give up the query, revert attention
                        # attention_mask[:, prefix_length:prefix_length+cur_length] = 1 # revert prev attention
                        # attention_mask[:, prefix_length+cur_length:-1] = 0 # mask noise attention
                        # attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                        # cache_position = cache_position[-1:] + 1
                        # (2) parallel mask: revert mask and crop KVCache
                        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(bsz, 1)], dim=-1)
                        past_key_values.crop(ori_length)
                        cache_position = cache_position[:1] + 1
                        channel += [channel[-1]-1]
                        next_token_ids = outputs.logits[:, :1].argmax(-1)
                        if tokenizer is not None:
                            print(tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
                        outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
                        inputs_embeds = self.get_model().embed_tokens(next_token_ids)
                        
                        
                    else:
                        # if not noise: change channel
                        # print("*******start new topic*******")
                        print("Assistant: ", end='', flush=True)
                        next_token_ids = outputs.logits[:, -1:].argmax(-1)
                        outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
                        if tokenizer is not None:
                            print(tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
                        if streamer is not None:
                            streamer.stream(output_ids=next_token_ids)
                        inputs_embeds = self.get_model().embed_tokens(next_token_ids)
                        # # (1) naive mask: 2d mask
                        # attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                        # cache_position = cache_position[-1:] + 1
                        # channel += [channel[-1]]
                        # (2) paprallel mask for simplicity: mask previous query TODO: consider previous query
                        attention_mask[:, prefix_length:] = 0 # mask prev attention
                        attention_mask = torch.cat([attention_mask[:,:-1], attention_mask.new_ones((new_inputs_embeds.shape[0], new_inputs_embeds.shape[1]+1))], dim=-1)
                        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                        cache_position = cache_position[-1:] + 1
                        channel += [channel[-1]]
                        
                
                else:
                    outputs = super().forward(
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                    )
                    
                    # greedy sample
                    next_token_ids = outputs.logits[:, -1:].argmax(-1)
                    if idx == 0:
                        outputs_ids = next_token_ids
                    else:
                        outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
                    if self.tokenizer is not None:
                        # if idx == 0: print("Assistant: ", end='', flush=True)
                        # print(self.tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
                        yield self.tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True)
                    # if streamer is not None:
                    #     streamer.stream(output_ids=next_token_ids)
                    inputs_embeds = self.model.get_model().embed_tokens(next_token_ids)
                    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                    cache_position = cache_position[-1:] + 1
                    channel += [channel[-1]]
                    
                if next_token_ids.squeeze(0)[0].item() == self.tokenizer.eos_token_id:
                    # print()
                    # print("="*20,"COMPLETE","="*20)
                    break
            
            
            

        outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        
        # print("=="*20)
        # print(outputs)
        # print("=="*20)
        
        # with torch.inference_mode():
        #     print(input_ids)
        #     print(self.tokenizer.convert_ids_to_tokens(input_ids[0]))
        #     outputs = self.model.forward(
        #         input_ids,
        #         images=image_tensor,
        #         modalities=["video"],
        #         # output_attentions=True,
        #         output_hidden_states=True,
        #         return_dict=True,
        #         use_cache=True,
        #     )
        #     print(outputs.keys())
        #     print(outputs["logits"].shape) # B L vocab_size
        #     print(outputs["hidden_states"][-1].shape) # B L 3584
        #     print(outputs["attentions"][-1].shape) # B NH L L
        #     print(outputs["past_key_values"])
            
                
        
        # return outputs
        
        # except Exception as e:
        #     raise e
        

def image_demo(model, args):
# Get the directory of the current script file

    # Construct the path to the visual file relative to the current script file
    visual_path = args.image_path
    image = Image.open(visual_path).convert("RGB")
    
    input_visuals = [image]
    input_context = args.question
    task_type = "image"
    gen_kwargs = {"max_new_tokens": 1024, "temperature": 0, "do_sample": False}
    query = {
        "visuals": input_visuals,
        "context": input_context,
        "task_type": task_type,
        "prev_conv": [],
    }
    try:
        prev = 0
        for x in model.stream_generate_until(query, gen_kwargs):
            output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)

        print("\n")
    except Exception as e:
        print(e)

def video_demo(model, args):
    visual_path = args.video_path
    input_visuals = [visual_path]
    input_context = args.question
    task_type = "video"
    gen_kwargs = {"max_new_tokens": 1024, "temperature": 0, "do_sample": False, "sample_frames": args.num_sampled_frames}
    query = {
        "visuals": input_visuals,
        "context": input_context,
        "task_type": task_type,
        "prev_conv": [],
    }
    # try:
    prev = 0
    
    print("start...")
    outputs = model.stream_generate_until(query, gen_kwargs)
    print(outputs)
    
    # for x in model.stream_generate_until(query, gen_kwargs):
    #     output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
    #     print(output[prev:], end="", flush=True)
    #     prev = len(output)

    print("\n")
    # except Exception as e:
    #     print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--num_sampled_frames", type=int, default=32)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    # model = LongVA(pretrained="checkpoints/longva7b-llavanextsub10k-qwen2-noise", model_name="llava_qwen", device_map=args.device)
    # model = LongVA(pretrained="checkpoints/longva7b-llavanextsub10k-qwen2-noise", model_name="llava_qwen", device_map=args.device, attn_implementation="flash_attention_2")
    # model = LongVA(pretrained="checkpoints/LongVA-Qwen2-7B-Instruct", model_name="llava_qwen", device_map=args.device)
    
    model = LongVA(pretrained="checkpoints/longva7b-llavanextsub10k-qwen2-ORNS1111", model_name="llava_qwen", device_map=args.device, attn_implementation="eager")
    # model = LongVA(pretrained="checkpoints/longva7b-llavanextsub10k-qwen2-ORNS1111", model_name="llava_qwen", device_map=args.device, attn_implementation="flash_attention_2")
    # model = LongVA(pretrained="checkpoints/longva7b-llavanextsub10k-qwen2-ORNS1111", model_name="llava_qwen", device_map=args.device)
    
    # if args.image_path:
    #     image_demo(model, args)
    # if args.video_path:
    #     video_demo(model, args)
    
    visual_path = args.video_path
    input_visuals = [visual_path]
    input_context = args.question
    task_type = "video"
    gen_kwargs = {"max_new_tokens": 1024, "temperature": 0, "do_sample": False, "sample_frames": args.num_sampled_frames}
    query = {
        "visuals": input_visuals,
        "context": input_context,
        "task_type": task_type,
        "prev_conv": [],
    }
    # try:
    prev = 0
    
    print("start...")
    outputs = model.output(query, gen_kwargs)
    # print(outputs)
    
    # for x in model.stream_generate_until(query, gen_kwargs):
    #     output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
    #     print(output[prev:], end="", flush=True)
    #     prev = len(output)

    print("\n")
    # except Exception as e:
    #     print(e)