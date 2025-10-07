from dotenv import load_dotenv
import os
import requests
from typing import Any, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

class OpenRouter(LLM):
    model_name: str = "deepseek/deepseek-chat-v3.1:free" 
    temperature: float = 0.7 

    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # 委托给invoke方法处理，保持实现一致
        return self.invoke(prompt, stop, run_manager,** kwargs)
    
    def invoke(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = requests.post(
                "https://openrouter.ai/api/alpha/responses",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",#考虑system或者user
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                },
                            ],
                        },
                    ],
                },
            )
            return response.json()["output"][0]["content"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            return f"API请求错误: {str(e)}"
        except Exception as e:
            return f"处理错误: {str(e)}"


if __name__ == "__main__":
    llm = OpenRouter()
    
    prompt = "你是一个无人机的避障路径规划系统,前方的障碍物信息是:1:前方5米处有障碍物大树,左上方坐标为(10, 20),右上方坐标为(15, 20),左下方坐标为(10, 15),右下方坐标为(15, 15)。2:前方8米处有障碍物建筑物,左上方坐标为(20, 30),右上方坐标为(25, 30),左下方坐标为(20, 25),右下方坐标为(25, 25)。请根据以上信息,以类似于1、向前走1m,2、向右转45度,3、向前走2m的格式输出避障路径规划方案,注意避障路径规划方案必须是可行的。注意我只需要最终方案"
    
    # 调用模型（使用invoke方法）
    response = llm.invoke(prompt)
    print("模型响应:")
    print(response)
