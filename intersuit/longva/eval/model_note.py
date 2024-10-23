import math
import random
import os
import argparse
import json

import torch
import transformers
from tqdm import tqdm

from longva.vid_utils import load_video
from longva.model.builder import load_pretrained_model
from longva.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from longva.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from longva.conversation import conv_templates, SeparatorStyle

OPTIONS = ["A", "B", "C", "D", "E"]
# OPTIONS = ["(1)", "(2)", "(3)", "(4)", "(5)"]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_random_chunk(lst, n, k, seed=42):
    random.seed(seed)
    lst = lst[:]
    random.shuffle(lst)
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=4096)

    return parser.parse_args()

def get_model_output(model, image_processor, tokenizer, video, qs, args):
    
    # preprompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    # postprompt = "<|im_end|>\n<|im_start|>assistant\n"

    if os.path.isdir(video):
        video = load_video(video, video_decode_backend='frame', fps=1)
    elif video.endswith(".gif"):
        video = load_video(video, video_decode_backend='gif', fps=1)
    else:
        video = load_video(video, fps=1, max_frames=64)
    # video = load_video(video, fps=1)
    video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].to(args.device, dtype=torch.float16)
    
    
    # print(video_tensor.shape)
    # qs = preprompt + DEFAULT_IMAGE_TOKEN + qs + postprompt
    image_tokens = DEFAULT_IMAGE_TOKEN
    
    # qs = image_tokens + '\n' + qs
    qs = qs + image_tokens
    
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    qs = conv.get_prompt()
    input_ids = tokenizer_image_token(qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # post-hoc 
    
    
    with torch.inference_mode():
        # output_ids = model.generate(
        #     input_ids,
        #     images=[video_tensor],
        #     # images=video_tensor,
        #     modalities=["video"],
        #     do_sample=False,
        #     max_new_tokens=1024,
        #     use_cache=True,
        #     stopping_criteria=[stopping_criteria]
        #     )
        
        outputs = model.forward(
            input_ids,
            images=[video_tensor],
            modalities=["video"],
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        # analysis



    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # outputs = outputs.strip()

    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name, device_map="auto")
    model.eval()
    model.tie_weights()
    model = model.to(args.device)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    # gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_questions = get_random_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    # gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)
    gt_answers = get_random_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    acc = 0
    
    for sample in tqdm(gt_questions):
        video_name = sample['video']
        idx = sample['video'].split(".mp4")[0]
        answer = gt_answers[index]['answer']
        if "type" in gt_answers[index]:
            typeid = gt_answers[index]["type"]
        else:
            typeid = None
        index += 1

        # llava prompt
        question = sample['question']
        # question += "\n"

        answer = OPTIONS[answer]
        sample_set = {'id': idx, 'question': question, 'answer': answer}
        if typeid: sample_set["type"] = typeid

        # Load the video file
        temp_path = os.path.join(args.video_dir, video_name)
        if os.path.exists(temp_path):
            video_path = temp_path
            # try:
            # Run inference on the video and add the output to the list
            output = get_model_output(model, image_processor, tokenizer, video_path, question, args).split('.')[0]
            if output == answer: acc += 1
            sample_set['pred'] = output
            output_list.append(sample_set)
            # except Exception as e:
            #     print(f"Error processing video file '{video_name}': {e}")
            ans_file.write(json.dumps(sample_set) + "\n")


    ans_file.close()
    
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
