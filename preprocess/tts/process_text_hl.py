import json
import os
import ChatTTS
import torch
import torchaudio
from num2words import num2words
import re
from tqdm import tqdm
import argparse
chat = ChatTTS.Chat()
chat.load(source="local", compile=True)
#chat.normalizer.register("en", normalizer_en_nemo_text())
#chat.normalizer.register("zh",normalizer_zh_tn())

import multiprocessing
from multiprocessing import Pool

def convert_numbers_to_words(text):
    """将文本中的所有数字转换为对应的英文单词。"""
    def replace(match):
        num = match.group(0)
        # 将数字转换为英文单词
        return num2words(num)
 
    # 使用正则表达式查找数字并替换
    return re.sub(r'\b\d+\b', replace, text)

def clean_text(text):
    """清理文本中的 '<image>\n' 和换行符 '\n'。"""
    text = text.replace("-", "to")
    return text.replace("<image>\n", "").replace("\n", " ")

def text_to_speech(item_id, idx, text, output_dir="/ceph/home/songchun01/hengli/patrick/data/speech"):
    """将文本转为语音并保存为指定路径的音频文件。"""
    os.makedirs(output_dir, exist_ok=True)
    audio_filename = f"{item_id}_{idx}.wav" 
    audio_path = os.path.join(output_dir, audio_filename)
    # 提取目录路径
    directory = os.path.dirname(audio_path)
    
    # 确保目录存在
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.exists(audio_path):
        return audio_filename
    wav = chat.infer(text)

    try:
        torchaudio.save(audio_path, torch.from_numpy(wav).unsqueeze(0), 24000)
    except:
        torchaudio.save(audio_path, torch.from_numpy(wav), 24000)
    
    
    return audio_filename

def process_conversation(items):
    """处理单个 JSON 项目，将 'human' 对话转为音频并更新 `speech` 列表。"""
    speech_files = []
    for _, item in tqdm(enumerate(items), total=len(items)):
        count = 0
        for i, convo in enumerate(item["conversations"]): 
            if convo.get("from") == "human":
                # 将文本转为音频文件

                clean_value = clean_text(convo["value"])
                clean_value = convert_numbers_to_words(clean_value)
                audio_filename = text_to_speech(item["id"], count, clean_value)
                count+=1
                # 记录音频文件路径
                speech_files.append(audio_filename)
        # 添加 `speech` 字段
        if len(speech_files) == 0:
            continue
        item["speech"] = speech_files
        torch.cuda.empty_cache()
    return item

def split_into_batches(data, num_batches):
    # 计算每个批次的大小
    batch_size = len(data) // num_batches
    remainder = len(data) % num_batches  # 处理余数

    batches = []
    start = 0
    for i in range(num_batches):
        # 如果有余数，当前批次多一个元素
        end = start + batch_size + (1 if i < remainder else 0)
        batches.append(data[start:end])
        start = end

    return batches

#with open("./updated_data.json", "w", encoding="utf-8") as f:
#    json.dump(merged_result, f, ensure_ascii=False, indent=4)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Process data batches.")
    parser.add_argument('--batch_id', type=int, required=True, help="ID of the batch to process.")
    parser.add_argument('--num_batch', type=int, required=True, help="ID of the batch to process.")
    args = parser.parse_args()

    with open("llava-next-sub-10k-ORNS1111.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    num_batches = args.num_batch
    batches = split_into_batches(data, num_batches)
    # 处理指定的批次
    batch = batches[args.batch_id]
    new = process_conversation(batch)

if __name__ == "__main__":
    main()
