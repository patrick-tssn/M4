from openai import AzureOpenAI, OpenAI
from abc import ABC, abstractmethod
import time
import random


from os import getenv
from dotenv import load_dotenv
load_dotenv()
API_BASE = getenv("API_BASE")
API_KEY = getenv("API_KEY")

REGIONS = {
        "gpt-35-turbo-0125": ["canadaeast", "northcentralus", "southcentralus"],
        "gpt-4-0125-preview": ["eastus", "northcentralus", "southcentralus"],
        "gpt-4-turbo-2024-04-09": ["eastus2", "swedencentral"]
    }

class BaseAPIWrapper(ABC):
    @abstractmethod
    def get_completion(self, user_prompt, system_prompt=None):
        pass

class OpenAIAPIWrapper(BaseAPIWrapper):
    def __init__(self, caller_name="default", api_base="",  key_pool=[], temperature=0, model="gpt-4-turbo-2024-04-09", time_out=5):
        api_base = API_BASE
        key_pool = [API_KEY]
        self.temperature = temperature
        self.model = model
        self.time_out = time_out
        self.api_key = random.choice(key_pool)
        if api_base:
            self.model = model
            region = random.choice(REGIONS[model])
            endpoint = f"{api_base}/{region}"
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )
        else:
            self.model = model
            self.client = OpenAI(
                api_key=self.api_key
            )

    def request(self, usr_question, system_content, image_path=None):
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # {"role": "system", "content": f"{system_content}"},
                {"role": "user", "content": f"{usr_question}"}
            ],
        )

        # resp = response.choices[0]['message']['content']
        # total_tokens = response.usage['total_tokens']
        resp = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        

        return resp, total_tokens
    
    def get_completion(self, user_prompt=None, system_prompt=None, image_path=None, max_try=10):
        gpt_cv_nlp = ""
        total_tokens = 0
        while max_try > 0:
            try:
                gpt_cv_nlp, total_tokens = self.request(user_prompt, system_prompt, image_path)
                res = {
                    "response": gpt_cv_nlp,
                    "tokens": total_tokens
                }
                max_try = 0
                break
            except Exception as e:
                print(f"encounter error: {e}")
                print("fail ", max_try)
                time.sleep(self.time_out)
                max_try -= 1
    
        return gpt_cv_nlp, total_tokens


