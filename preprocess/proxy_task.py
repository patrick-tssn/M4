from dataclasses import dataclass

@dataclass
class Tasks:
    tasks = [
        "t2d",
        "t2c",
        "c2d",
        "d2c"
    ]
    t2d = {
        "past": [
            "what does the person said before?", 
            "Can you retell what the speaker said before?",
            "What were the speaker talking about a while ago?"
        ],
        "currents": [
            "What is the person talking about?",
            "Can you retell what the speaker is talking now?"
        ],
    }
    t2c = {
        "past": [
            "What happended in the video a while ago?",
            "Can you summarize what happended in the video before?",
            "Please descibe the video several minutes ago."
        ],
        "current": [
            "What is happening in the video?",
            "Descirbe the video now."
        ]
    }
    c2d = [
        "According to the description of the video {caption}, what does the speak say?",
        "Based on the caption of the video: {caption}, pleaase response the dialogue happened in the video."
    ]
    d2c = [
        "Based on their talks: {dialogue}, what happended in the video?",
        "Can you describe the video according to their dialogues: {dialogue}?"
    ]