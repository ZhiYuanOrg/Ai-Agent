# AI Agent项目说明

## 依赖安装

```
pip install -r requirements.txt
```

## 项目构思说明

该项目是用于与无人机进行交互，从无人机返回障碍物信息（类别、距离、坐标），通过PromptManager填入base_prompt,将其通过langchain框架传给AI

## 目前进度

已经写好了与AI模型链接的组件（即搭好了大体langchain框架于init.py）

写好了prompt拼接模块

## 待完善

1、差从yolo接受障碍物信息

2、没有检测是否连接成功与连接效果

3、拓展项：utils中的工具（非紧急）

## 你可能会用到

[OpenRouter官网]（[Responses API Alpha Basic Usage | Simple Text Requests | OpenRouter | Documentation](https://openrouter.ai/docs/api-reference/responses-api-alpha/basic-usage)）



