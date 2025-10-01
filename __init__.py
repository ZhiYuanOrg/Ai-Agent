from dotenv  import load_dotenv
import os

load_dotenv()
API_KEY=os.getenv("OPENROUTER_API_KEY")

from langchain.llms.base import LLM
import requests
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun

class OpenRouter(LLM):
    model_name: str = "deepseek/deepseek-chat-v3.1:free" 
    temperature: float = 0.7 

    @property
    def _llm_type(self)->str:
        return "openrouter"
    
    #用于调用OpenRouter的模型
    def _call(self,
              prompt:str,
              stop:Optional[List[str]] = None,
              run_manager:Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any)->str:
        #官方文档的调用
        try:
            response = requests.post(
            'https://openrouter.ai/api/alpha/responses',
            headers={
                'Authorization': 'Bearer {}'.format(API_KEY),#认证key
                'Content-Type': 'application/json',#规定格式为json
            },
            json={
                'model': '{}'.format(self.model_name),#传入使用的模型
                'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': '{}'.format(prompt),#这里传入用户的prompt描述
                        },
                    ],
                },
            ],
            }
            )
            return response.json()
        except Exception:
            return "Error: Unable to connect to OpenRouter API"
