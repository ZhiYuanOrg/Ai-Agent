'''这个文件用来拼接无人机面前对于障碍物的描述语句'''

#基础的表述prompt
bass_prompt="{}:前方{}米处有障碍物{},左上方坐标为{},右上方坐标为{},左下方坐标为{},右下方坐标为{}。\n"

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
        for i in range(len(self.obstacles)):
            self.prompt+=bass_prompt.format(i+1,
                                    self.distance[i],
                                    self.obstacles[i],
                                    self.positions[i][0],
                                    self.positions[i][1],
                                    self.positions[i][2],
                                    self.positions[i][3])
        return self.prompt
 