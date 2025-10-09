'''这个文件用来拼接无人机面前对于障碍物的描述语句'''

#基础的表述prompt
#需要调整
front_prompt="你是一个无人机的避障路径规划系统，前方的障碍物信息是："
base_prompt="{}:前方{}米处有障碍物{},左上方坐标为{},右上方坐标为{},左下方坐标为{},右下方坐标为{}。\n"
back_prompt="请根据以上信息,以类似于1、向前走1m,2、向右转45度,3、向前走2m的格式输出避障路径规划方案,只告诉我最终避障方案避障，不要包含别的，注意避障路径规划方案必须是可行的。"
class PromptManager:
    def __init__(self):
        self.prompt=""
        self.obstacles=[]
        self.distance=[]
        self.positions=[]

    def extend_info(self,obstacle,distance,positions):
        #这个逻辑是伪的
        self.obstacles.extend(obstacle)
        self.distance.extend(distance)
        self.positions.extend(positions)

    def get_prompt(self):
        self.prompt=""
        self.prompt+=front_prompt+"\n"
        for i in range(len(self.obstacles)):
            self.prompt+=base_prompt.format(i+1,
                                    self.distance[i],
                                    self.obstacles[i],
                                    self.positions[i][0],
                                    self.positions[i][1],
                                    self.positions[i][2],
                                    self.positions[i][3])
        self.prompt+=back_prompt
        return self.prompt

#测试代码 
if __name__ == "__main__":
    # 创建PromptManager实例
    prompt_manager = PromptManager()
    
    # 准备测试数据
    obstacles = ["大树", "建筑物", "电线杆"]
    distances = [5, 10, 3]
    positions = [
        [(10, 20), (15, 20), (10, 15), (15, 15)],  # 第一个障碍物的四个坐标
        [(20, 30), (25, 30), (20, 25), (25, 25)],  # 第二个障碍物的四个坐标
        [(5, 15), (8, 15), (5, 12), (8, 12)]       # 第三个障碍物的四个坐标
    ]
    
    # 添加信息
    prompt_manager.extend_info(obstacles, distances, positions)
    
    # 获取生成的prompt
    generated_prompt = prompt_manager.get_prompt()
    
    # 打印结果以便查看
    print("生成的提示词如下：")
    print(generated_prompt)
    
    # 简单验证
    assert len(prompt_manager.obstacles) == 3, "障碍物数量不正确"
    assert len(prompt_manager.distance) == 3, "距离数量不正确"
    assert len(prompt_manager.positions) == 3, "位置信息数量不正确"
    assert front_prompt in generated_prompt, "未包含前置提示"
    assert back_prompt in generated_prompt, "未包含后置提示"
    assert "大树" in generated_prompt, "未包含障碍物信息"
    assert "5米" in generated_prompt, "未包含距离信息"
    
    print("\n测试通过！")