import os
import yaml
import json
import unicodedata
from datetime import datetime

# 获取 musa 设备
def hardware_detection():
    try:
        import torch_musa
        _ = torch_musa.device_count()
        musa_available = torch_musa.is_available()
        if musa_available:
            print("MUSA device detected: {}".format(torch_musa.get_device_name(0)))
    except:
        musa_available = False

def torch_version():
    try:
        import torch
        print("pytorch version: {}".format(torch.version.__version__))
    except:
        pass

# 配置文件读取
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 获取目录下所有非隐藏文件
def get_all_file(dir_path, exp_name):
    file_path_list = []
    for entry in os.listdir(dir_path):
        file_path = os.path.join(dir_path, entry)
        if os.path.isfile(file_path):
            if not entry.startswith('.') and entry.endswith(exp_name):
                file_path_list.append(file_path)
    # 按创建时间降序排序（最新的文件在前）
    file_path_list = sorted(file_path_list, key=lambda x: os.path.getctime(x), reverse=True)
    
    return file_path_list

# 读取单json文件内容
def load_json_content(json_file):
    content = []
    try:
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
    except Exception as e:
        print(f"聊天历史文件 {json_file} 无法读取：", e)
    return content

# 读取文本信息，似乎没用到
def get_file_text(file_path, model_message):
    filename = os.path.basename(file_path)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            model_message += f"\n\n以下是用户上传的文件 {filename} 的内容：\n{file_content}"
        except Exception as e:
            model_message += f"\n\n用户上传的文件 {filename} 读取失败：\n{str(e)}"
    else:
        model_message += f"\n\n用户上传的文件 {filename} 无法找到"
    file_content_dict[file_path] = model_message
    return filename, file_content_dict

# 文本长度限制
def truncate(text, max_len):
    return text[:max_len] + "..." if len(text) > max_len else text

# 文本长度粗略计算
def rough_token_estimate(text):
    # 经验值：每个字符 ≈ 0.5 ~ 1.5 token
    lines = text.count('\n')
    avg_line_len = len(text) / max(lines, 1)
    if avg_line_len < 40:
        return int(len(text) * 1.5)  # 代码类
    elif avg_line_len > 100:
        return int(len(text) * 0.7)  # 英文长句
    else:
        return int(len(text))       # 中文或混合

# 文本宽度限制，用于标题
def visual_truncate(text, max_width):
    result = ''
    width = 0
    for ch in text:
        # 东亚宽字符或全角字符计为宽度 2
        if unicodedata.east_asian_width(ch) in ('W', 'F'):
            width += 2
        else:
            width += 1
        if width > max_width:
            result += '...'
            break
        result += ch
    return result

# 获取json文件列表和第一条信息的前面部分
def get_json_list(history_dir):
    first_message_list = []
    json_file_list = get_all_file(history_dir, '.json')
    if json_file_list:
        for file_path in json_file_list:
            try:
                content = load_json_content(file_path)
                if content:
                    first_content = content[0]
                    first_message = first_content.get('content')
                    first_message = visual_truncate(first_message, max_width=20)
                    first_message_list.append(first_message)
                else:
                    first_message_list.append("空聊天记录")
            except Exception as e:
                print(f"聊天历史文件 {file_path} 无法读取：", e)
                pass

    return json_file_list, first_message_list

# 上下文截断，只保留系统角色和最新
def smart_truncate_conversation(conversation, tokenizer, max_tokens=32768, min_reserve=2):
    try:
        # 拆分结构：system + middle + reserved
        system_msg = conversation[0] if conversation[0]['role'] == 'system' else None
        middle = conversation[1:-min_reserve] if system_msg else conversation[:-min_reserve]
        reserved = conversation[-min_reserve:]

        # 计算总 tokens
        total = sum(len(tokenizer.encode(msg['content'], add_special_tokens=False)) for msg in conversation)
        print("上下文总 tokens:", total)

        if total <= max_tokens:
            return conversation  # 没超限，直接返回

        # 截断中间部分
        while len(middle) > 0 and total > max_tokens:
            removed = middle.pop(0)
            total -= len(tokenizer.encode(removed['content'], add_special_tokens=False))

        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(middle + reserved)
        return result

    except Exception as e:
        print(f"✘ 发生错误: {e}")


# 模型信息整理
def model_mes_manage(chat_history):
    to_model_messages = []
    last_text_message = None 
    for item in chat_history:
        raw_content = item["content"]
        # 当前是文件路径
        if isinstance(raw_content, tuple):
            file_path = raw_content[0]
            filename = os.path.basename(file_path)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    file_text = f"以下是用户上传的文件 {filename} 的内容：\n{file_content}"
                except Exception as e:
                    file_text = f"用户上传的文件 {filename} 读取失败：\n{str(e)}"
            else:
                file_text = f"用户上传的文件 {filename} 无法找到"
    
            # 合并上一条文本消息
            if last_text_message and last_text_message["role"] == item["role"]:
                last_text_message["content"] += "\n\n" + file_text
            else:
                # 如果没得合并，就单独添加
                to_model_messages.append({"role": item["role"], "content": file_text})
    
            last_text_message = None  # 合并后清空
    
        else:
            # 普通文本消息，暂存，待合并
            last_text_message = {"role": item["role"], "content": raw_content}
            to_model_messages.append(last_text_message)
    return to_model_messages

# 备用的，粗糙的
def alter_yaml(yaml_file, parent, key, value):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)  # 将 YAML 内容解析为 Python 字典/列表

    if parent:
        if parent in data:
            data[parent][key] = value
    else:
        if key in data:
            data[key] = value

    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    repoid = 'Qwen/Qwen3-Embedding-0.6B'  # 替换为实际目录路径
    dir_path = '/root/autodl-tmp/Qwen3-Embedding-0.6B' #'.json'
    download_ms(repoid, dir_path)