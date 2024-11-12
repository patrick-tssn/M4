## audio codec

- dac: [1024, 165]
https://github.com/descriptinc/descript-audio-codec

- mimi: [8, 28]
https://github.com/kyutai-labs/moshi/tree/main/moshi

- snac: [20], [40], [80], [160]
https://github.com/hubertsiuzdak/snac


## moshi training data

audio training data
- 7M hours audio + whisper -> single stream data
- fisher dataset: 2000 hours phone conversation
- 170 hours multi-party natural and scripted conversations

speech-text instruction data
- LLM generated text dialogue + TTS

## moshi benchmark

- latency

- audio-language modeling: text -> audio; compare the likelihood 
    - textless NLP: compare the likelihood of an existing word and and invalid variant 
    - Spoken StoryCloze: compare commonsense 5-sentence stories
    - Spoken Topic-StoryCloze: randomly sampled sentences
- quality of generated dialogues
    - perplexity from DialoGPT
- Streaming ASR and TTS
- Quantilization: 
    - audio quality: MOSNet

## moshi highlight

K * S total timesteps
-> S steps for temporal transformer
-> K steps for depth transformer

--> t, q  t in T = 12.5 * sec (samples) q in Q = 8

acoustic delay: semantic token

inner monologue: PAD, EPAD (before start of a word from pad) for no-response