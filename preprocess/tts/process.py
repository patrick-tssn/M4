import os
import json
from tqdm import tqdm
import random
random.seed(43)

items = json.load(open("llava-next-sub-10k-ORNS1111.json"))

missing = []
cnt =0

for _, item in tqdm(enumerate(items), total=len(items)):
    count = 0
    speech_files = []
    max_speech = 1
    speech_indices = random.sample(range(len(item["conversations"])), max_speech)
    for i, convo in enumerate(item["conversations"]): 
        if convo.get("from") == "human":
            # 将文本转为音频文件
            output_dir = "/scratch/nlp/data/interinst/orns"
            item_id = item["id"]
            audio_filename = f"{item_id}_{count}.wav" 
            audio_path = os.path.join(output_dir, audio_filename)
            if not os.path.exists(audio_path):
                # print("no existing file: ", audio_path)
                # raise ValueError("error")
                missing.append(item_id)
                speech_files.append("")
            
            # 记录音频文件路径
            speech_files.append(audio_filename)
            if count in speech_indices:    
            # if count == 0:
                if convo["value"].startswith("<image>\n"):
                    convo["value"] = "<image>\n<speech>\n"
                elif convo["value"].endswith("<image>\n"):
                    convo["value"] = "<speech>\n<image>\n"
                else:
                    convo["value"] = "<speech>\n"
            
            count+=1
            
    item["speech"] = speech_files
    cnt += count
    assert len(speech_files) == len(item["conversations"])//2
    
# with open(f"llava-next-sub-10k-ORNS1111-speech-{max_speech}_start.json", "w") as jp:
with open(f"llava-next-sub-10k-ORNS1111-speech-{max_speech}.json", "w") as jp:
    json.dump(items, jp, indent=4)
print(len(missing), cnt)