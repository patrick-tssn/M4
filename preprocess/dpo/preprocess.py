import jsonlines

vid_dir = [""]

with jsonlines.open("sft_dpo_17k.json") as jpl:
    for dct in jpl:
        vid = dct["video"]
        