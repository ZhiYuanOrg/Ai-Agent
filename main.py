from detector import Detector
from prompt_manager import PromptManager
from __init__ import OpenRouter

if __name__ == "__main__":
    promptManager = PromptManager()
    openRouter = OpenRouter()
    detector = Detector("models/yolo11n.onnx", r"imgs/test.jpg", 0.5, 0.45)
    output_img, obstacles, distances, positions = detector.main()
    promptManager.extend_info(obstacles, distances, positions)
    prompt = promptManager.get_prompt()
    print("生成的prompt：")
    print(prompt)
    response = openRouter.invoke(prompt)
    print("AI响应：")
    print(response)


