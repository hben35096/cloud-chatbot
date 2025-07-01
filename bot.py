import argparse
import gradio as gr
from gradio import utils
import random
import time
import os
import json
import function
import repo_download
from datetime import datetime

parser = argparse.ArgumentParser(description="Cloud chatbot 启动脚本", add_help=True)
parser.add_argument('--port', nargs='?', const=True, help='指定服务端口号')
parser.add_argument('--listen', nargs='?', const=True, help='以 0.0.0.0 作为服务器名称启动 Gradio，允许响应网络请求')
parser.add_argument('--ui-test', action='store_true', help="允许在无显卡模式下启动，以测试UI的功能")
parser.add_argument('--model-set', nargs='?', const=True, help='选择模型并加载对应配置信息')
parser.add_argument('--show-prompt', nargs='?', const=True, help='显示完整上下文')
parser.add_argument('--warning-off', action='store_true', help="关闭 Python 警告")
args = parser.parse_args()

if args.warning_off:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

if args.listen:
    host = "0.0.0.0"
else:
    host = "127.0.0.1"
    

function.hardware_detection()
function.torch_version()


self_dir = os.path.abspath(os.path.dirname(__file__))
history_dir = os.path.join(self_dir, 'histories')
now_json_file = os.path.join(history_dir, 'NewChat.json')

main_config = function.load_config(os.path.join(self_dir, 'main_config.yaml'))
model_config = function.load_config(os.path.join(self_dir, 'model_config.yaml'))
models_dir = os.path.join(self_dir, 'models')

main_name = main_config["main_name"]
main_description = main_config["main_description"]

model_name = list(model_config)[0]
if args.model_set:
    model_card = args.model_set
    if model_card in list(model_config):
        model_name = model_card
    else:
        print(f"--model-set 参数中的模型名称 {model_card} 不正确，将使用默认模型配置")
        print(f"正确填写的应该是 {list(model_config)[0]} 或 {list(model_config)[1]}")

print(f"Load model config：{model_name}")

description = model_config[model_name]["description"]
dtypes = model_config[model_name]["dtypes"]
max_input_tokens = model_config[model_name]["max_input_tokens"]
max_generate_tokens = model_config[model_name]["max_generate_tokens"]
max_msg = model_config[model_name]["max_msg"]
max_file_content = model_config[model_name]["max_file_content"]
system_set = model_config[model_name]["system_set"]


up_file_type = model_config[model_name]["up_file_type"]

json_file_list, first_message_list = function.get_json_list(history_dir) # json_file_dict

file_content_dict = {}
samples = [[t] for t in first_message_list]

model = None
tokenizer = None
intercept = False
dividing_line = "-" * 50

# 用不着的
def dummy(v):
    print(f"你输入了：{v}")
    return v

def role_tip(role):
    tip = f"你刚刚选择了角色 {role}，建议切换到新对话使用。"
    print(tip)
    gr.Info(tip, duration=4)
    
# 这是聊天列表选择后触发的，获取文件内容给机器人和网页缓存
def get_json_content(index):
    global now_json_file
    now_json_file = json_file_list[index]
    json_content = function.load_json_content(now_json_file)
    # 将读取到的记录文件内容，发送给机器人和网页存储
    return json_content, json_content

# 清空后，聊天记录为空，就会自动创建一个新的文件，所以聊天记录并不是真的被清空
def clear_conversation():
    return {}, [], []  

# 给右侧边栏网页存储的，一开始要加载初始值，如是选项则是第一个，如是数字则直接给
column_dict = {
    "model": list(model_config)[0],
    "dtype": dtypes[0],
    "role": list(system_set)[0],
    "max_tokens": max_input_tokens,
    "max_generate": max_generate_tokens,
    "max_msg": max_msg,
    "max_file_token": max_file_content
}
# 聚焦文本输入框时触发，获取 7 个选项，并把所选值给右侧边栏网页存储
def storage_column(model, dtype, role, max_tokens, max_generate, max_msg, max_file_token):
    column_dict = {
        "model": model,
        "dtype": dtype,
        "role": role,
        "max_tokens": max_tokens,
        "max_generate": max_generate,
        "max_msg": max_msg,
        "max_file_token": max_file_token
    }
    return column_dict
# 刷新网页用的，用于获取最新聊天列表
def init_load(storage_state, storage_column_r):
    # 重新获取文件列表
    json_file_list, first_message_list = function.get_json_list(history_dir) 
    new_samples = [[t] for t in first_message_list]

    dict = storage_column_r
    # 原样发送给自己，也把读取到的列表信息发送给会话列表
    return storage_state, gr.update(samples=new_samples), dict["model"], dict["dtype"], dict["role"], dict["max_tokens"], dict["max_generate"], dict["max_msg"], dict["max_file_token"]

def clear_gpu(model):
    import gc, torch_musa
    del model
    gc.collect()
    torch_musa.empty_cache()

def load_model(model_path, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    global model, tokenizer

    model_name = os.path.basename(model_path)

    if model is None or tokenizer is None:
        info = f"⏳ 正在加载模型 {model_name} ..."
        print(info)
        gr.Info(info)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype),
            device_map="auto"
        )
        print("✅ 模型加载完成")

def file_size_warn(file_path, max_file_token):
    warn_info = ""
    intercept = False
    if os.path.getsize(file_path) > 2 * 1024 * 1024:
        warn_info = "❌ 文件过大，请控制在 2MB 以内。"

        intercept = True
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            approx_tokens = function.rough_token_estimate(file_content)
            print(f"上传文件估算 token 数：{approx_tokens}")
            if approx_tokens > max_file_token: # 记得改成 24000
                warn_info = "❌ 文件内容过多，请重新选择文件。"
                intercept = True
        except Exception as e:
            warn_info = "❌ 文件读取失败，请重新选择文件。"
            intercept = True
    return intercept, warn_info
    
# 传给机器人传给历史记录
def respond_user(message, chat_history, max_message, max_file_token):
    print(dividing_line)
    global now_json_file, json_file_list, intercept

    if not chat_history:
        now_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        now_json_file = os.path.join(history_dir, f'{now_time}.json') 

    files = None
    intercept = False
    if isinstance(message, dict):
        text_message = message.get("text", "")
        if files is None:
            files = message.get("files", [])
    else:
        text_message = message
        
    # 各种拦截
    if not text_message.strip() and not files:
        intercept = True
        warn_info = "❌ 空信息"
        
    approx_tokens = function.rough_token_estimate(text_message)
    if approx_tokens > max_message:
        intercept = True
        warn_info = "❌ 一次性发送信息过多，请减少文字或分段发送"
    
    if files:
        intercept, warn_info = file_size_warn(files[0], max_file_token)

    if intercept:
        print(warn_info)
        gr.Info(warn_info, duration=2)

        json_file_list, first_message_list = function.get_json_list(history_dir) # 无奈啊
        samples=[[t] for t in first_message_list]
        return "", chat_history, chat_history, gr.update(samples=samples)

    chat_history.append({"role": "user", "content": text_message})
    
    if files:
        if isinstance(files, list):
            files = tuple(files) # 强行转为元组，倒是给模型前进行统一整理
        chat_history.append({"role": "user", "content": files})
    # print("得到 now_json_file 名称：",now_json_file)

    with open(now_json_file, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)
    # 这是正确的，每次聊天都要重新获取一次聊天列表，有助于更新列表排位
    json_file_list, first_message_list = function.get_json_list(history_dir) 
    samples=[[t] for t in first_message_list]

    return "", chat_history, chat_history, gr.update(samples=samples)

def respond_bot(chat_history, model_select, role_select, dtype_select, do_sample, max_tokens_input, max_tokens_generate, record_model_name):
    global model, tokenizer
    
    role_set = system_set.get(role_select)
    print(f"💻 系统角色设定：{role_select}")
    if not intercept: # 如果信息被拦截拦截了，这里才执行
        global now_json_file, json_file_list
        system_message = [{"role": "system", "content": role_set.strip()}]
        to_model_messages = system_message + function.model_mes_manage(chat_history)
        last_msg = to_model_messages[-1]
        print(f"😎 {last_msg.get('role')} 提问：\n{last_msg.get('content')}")
        if args.ui_test:
            bot_message = random.choice([
                f"（UI测试模式下的模拟回复）🤖 我是通义千问模型 {model_name}",
                f"（UI测试模式下的模拟回复）😎 {description}",
                "（UI测试模式下的模拟回复）🤣 其实是个假机器人，哈哈"
            ]) # 简单模拟机器人，随机一个进行回复
            
            time.sleep(0.8)
            print(f"🤖 模拟AI回复：\n{bot_message}")
            chat_history.append({"role": "assistant", "content": bot_message}) # 模拟的机器人回复
        else:
            # 分析词元并截断，可以的，暂时灰掉
            from transformers import AutoTokenizer
            if do_sample == "启用":
                sample_Enable = True
            else:
                sample_Enable = False
            model_path = os.path.join(models_dir, model_select)
            repo_id = model_config[model_select]["ms_repoid"]
            absence = repo_download.check_repo_wholeness(repo_id, model_path)
            if absence:
                info_t = "检测本地模型不存在或不完整，开始下载模型文件..."
                print(info_t)
                gr.Info(info_t, duration=6)
                wholeness_files = repo_download.download_ms(repo_id, model_path)
                if wholeness_files:
                    print("✅ 模型已下载到：", model_path)
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            reserved = function.smart_truncate_conversation(to_model_messages, tokenizer, max_tokens=max_tokens_input, min_reserve=2)
            # print("截断后信息：\n", reserved)
            if model_select != record_model_name and model is not None:
                clear_gpu(model)
                model=None
                tokenizer=None
            
            load_model(model_path, dtype_select)
            text = tokenizer.apply_chat_template(reserved, tokenize=False, add_generation_prompt=True)
            if args.show_prompt:
                print("📝 传给模型的完整prompt：\n", text)
            
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens_generate,
                do_sample=sample_Enable,
            )
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"AI回复：\n{response}")
            chat_history.append({"role": "assistant", "content": response})
    
        with open(now_json_file, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
            
    json_file_list, first_message_list = function.get_json_list(history_dir) # 这是正确的，每次聊天都要重新获取一次聊天列表，有助于更新列表排位
    samples=[[t] for t in first_message_list]
    

    return "", chat_history, chat_history, gr.update(samples=samples), model_select


with gr.Blocks(theme="soft") as demo:
    storage = gr.BrowserState([])
    storage_column_r = gr.BrowserState(column_dict)
    record_model_name = gr.State()
    # gr.HTML("<h1 style='text-align: center;'>🍓Qwen ChatBot</h1>")
    gr.HTML(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 1.5em;">
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01EfJVFQ1uZPd7W4W6i_!!6000000006051-2-tps-112-112.png" 
           alt="Logo" style="width: 48px; height: 48px;">
      <div>
        <div style="font-size: 24px; font-weight: bold;">{main_name}</div>
        <div style="font-size: 14px; color: gray;">{main_description}</div>
      </div>
    </div>
    """)
    # json_dict = gr.Textbox(visible=False) # 不知道怎么使用了
    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            # 来自 gradio/gradio/chat_interface.py 289行
            new_chat_button = gr.Button(
                "新对话",
                variant="primary",
                size="md",
                icon=utils.get_icon_path("plus.svg"),
            )
            # 左侧边栏对话列表
            sess_list = gr.Dataset(
                components=[gr.Textbox(label='聊天记录', visible=False, lines=1, max_lines=1)],
                samples=samples,
                samples_per_page=14,
                type="index", layout="table", show_label=False
            )
            
        with gr.Column(scale=4):
            # 允许修改用户内容 editable="user"，这个还不知道怎么用，暂时不要
            chatbot = gr.Chatbot(type="messages", label='Qwen ChatBot', show_copy_button=False, height=560)
            msg = gr.MultimodalTextbox(placeholder="输入消息或上传文件...", show_label=False, file_types=up_file_type)
        with gr.Column(scale=1, min_width=280):
            model_select = gr.Dropdown(choices=list(model_config), value=model_name, label="模型",)
            role_select = gr.Dropdown(choices=list(system_set), label="角色选择",)
            dtype_select = gr.Dropdown(choices=dtypes, label="模型精度设置",)
            with gr.Accordion("更多设置", open=False):
                do_sample = gr.Dropdown(choices=["启用", "禁用"], label="随机采样",)
                max_tokens_input = gr.Number(value=max_input_tokens, label='最大上下文长度', info='限制长度以避免爆显存', precision=0, step=64)
                max_tokens_generate = gr.Number(value=max_generate_tokens, label='最大生成token', precision=0, step=64)
                max_message = gr.Number(value=max_msg, label='最大单次发送信息量', precision=0, step=64)
                max_file_token = gr.Number(value=max_file_content, label='最大文件信息量', precision=0, step=64)
    # 1.信息给 chatbot
    submit_event = msg.submit(
        respond_user,
        inputs=[msg, chatbot, max_message, max_file_token],
        outputs=[msg, chatbot, storage, sess_list]
    )

    # 2.再让模型回复
    submit_event.then(
        respond_bot,
        inputs=[chatbot, model_select, role_select, dtype_select, do_sample, max_tokens_input, max_tokens_generate, record_model_name],
        outputs=[msg, chatbot, storage, sess_list, record_model_name]
    )
    # 多模态文本框聚焦触发
    msg.focus(
        fn=storage_column,
        inputs=[model_select, dtype_select, role_select, max_tokens_input, max_tokens_generate, max_message, max_file_token],
        outputs=[storage_column_r]
    )

    role_select.select(fn=role_tip, inputs=[role_select], outputs=None)
    sess_list.select(get_json_content, sess_list, [chatbot, storage])
    new_chat_button.click(fn=clear_conversation, inputs=[], outputs=[msg, chatbot, storage],)

    demo.load(
        init_load,
        inputs=[storage, storage_column_r],
        outputs=[chatbot, sess_list, model_select, dtype_select, role_select, max_tokens_input, max_tokens_generate, max_message, max_file_token]
    )


if __name__ == "__main__":
    demo.launch(server_name=host, server_port=int(args.port),)