import yaml
def load_class_names(yaml_path):
    """
    从 YAML 文件加载类别名称

    Args:
        yaml_path (str): YAML 文件路径

    Returns:
        dict: 类别索引到名称的映射字典
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 检查不同的可能的键名
    names_key = None
    for key in ['names', 'nc', 'classes']:
        if key in data:
            names_key = key
            break

    if names_key is None:
        raise ValueError(f"在 YAML 文件中未找到类别名称键。可用的键: {list(data.keys())}")

    # 处理不同的 YAML 格式
    if isinstance(data[names_key], dict):
        # 格式: names: {0: 'person', 1: 'bicycle', ...}
        class_names = {int(k): v for k, v in data[names_key].items()}
    elif isinstance(data[names_key], list):
        # 格式: names: ['person', 'bicycle', ...]
        class_names = {i: name for i, name in enumerate(data[names_key])}
    else:
        raise ValueError(f"不支持的类别名称格式: {type(data[names_key])}")

    return class_names