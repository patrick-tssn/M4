# OminousLLM


Inspired by GPT-4o

## Roadmap

- data collection (with shenzhi)
    - data download: youtube with [100 keywords](preprocess/keywords.md)
- multi-task instruction tuning (with shenzhi)
    1. temporal sentence grouding in dialogue (timestamp -> dialogue). f"{timestamp}: {asr}"
    2. temporal sentence grouding in caption: (timestamp -> caption). f"{timestamp}: {caption}"
    3. caption to dialogue. f"When the description of the video clip is {caption}, what's the speech: {asr}"
    4. dialogue to caption. f"Please describe the video clip when someone is saying {asr}: {caption}"

    