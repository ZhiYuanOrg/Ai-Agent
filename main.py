from detector import Detector
from prompt_manager import PromptManager
from __init__ import OpenRouter

#逻辑待完善
if __name__ == "__main__":
    promptManager = PromptManager()
    openRouter=OpenRouter()
    detector=Detector("models/yolo11n.onnx", r"imgs/test.jpg", 0.5, 0.45)


