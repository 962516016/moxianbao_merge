import glob
import json
import os
import secrets
import random
import shutil
import socket
import zipfile
from datetime import datetime, timedelta
from urllib.parse import unquote

import joblib
import matplotlib.pyplot as plt
import matplotx
import numpy as np
import openai
import pandas as pd
import pymysql
import requests
from flask import Flask, jsonify, request, send_file, render_template, session, redirect
from flask_cors import CORS
from lightgbm import LGBMRegressor, early_stopping
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

import DataDeduplication
import IQR_OutlierCorrection
import KNN_FillNaN
import Normal
import Resample
import env
# from dialog import *
from env import DB_CONFIG
import base64

import datetime
import json
import shutil
import tempfile
import urllib.parse
from datetime import datetime
from easy_ernie import Ernie
import requests
from flask import Flask, render_template, request, session, send_file, make_response
import os
import uuid
from LRU_cache import LRUCache
import threading
import pickle
import asyncio
import yaml
from env import CONFIG
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = 'your_secret_key'
CORS(app)

# 字典存储各种功能对应的符号
api_list = {
    'upload_file': '0',
    'data_analyze': '1',
    'online_predict': '2',
    'gptapi_analyze': '3',
    'download_resfile': '4',
    'getmodel': '5',
    'api': '6'
}
# _________________________________________________________________________AI对话合并专区_________________________________________________
app.config['SECRET_KEY'] = os.urandom(24)
with open(CONFIG, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    if 'HTTPS_PROXY' in config:
        if os.environ.get('HTTPS_PROXY') is None:  # 优先使用环境变量中的代理，若环境变量中没有代理，则使用配置文件中的代理
            os.environ['HTTPS_PROXY'] = config['HTTPS_PROXY']
    if 'PASSWORD' in config:
        PASSWORD = config['PASSWORD']
    else:
        PASSWORD = ""  # 即不使用访问密码
    if 'ADMIN_PASSWORD' in config:
        ADMIN_PASSWORD = config['ADMIN_PASSWORD']
    else:
        ADMIN_PASSWORD = ""
    PORT = config['PORT']
    API_KEY = config['OPENAI_API_KEY']
    CHAT_CONTEXT_NUMBER_MAX = config[
        'CHAT_CONTEXT_NUMBER_MAX']  # 连续对话模式下的上下文最大数量 n，即开启连续对话模式后，将上传本条消息以及之前你和GPT对话的n-1条消息
    USER_SAVE_MAX = config['USER_SAVE_MAX']  # 设置最多存储n个用户，当用户过多时可适当调大

if os.getenv("DEPLOY_ON_RAILWAY") is not None or os.getenv("DEPLOY_ON_ZEABUR"):  # 如果是云部署，需要删除代理
    os.environ.pop('HTTPS_PROXY', None)

API_KEY = os.getenv("OPENAI_API_KEY", default=API_KEY)  # 如果环境变量中设置了OPENAI_API_KEY，则使用环境变量中的OPENAI_API_KEY
PORT = os.getenv("PORT", default=PORT)  # 如果环境变量中设置了PORT，则使用环境变量中的PORT
PASSWORD = os.getenv("PASSWORD", default=PASSWORD)  # 如果环境变量中设置了PASSWORD，则使用环境变量中的PASSWORD
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", default=ADMIN_PASSWORD)  # 如果环境变量中设置了ADMIN_PASSWORD，则使用环境变量中的ADMIN_PASSWORD
if ADMIN_PASSWORD == "":
    ADMIN_PASSWORD = PASSWORD  # 如果ADMIN_PASSWORD为空，则使用PASSWORD

STREAM_FLAG = True  # 是否开启流式推送
USER_DICT_FILE = "all_user_dict_v3.pkl"  # 用户信息存储文件（包含版本）
lock = threading.Lock()  # 用于线程锁

project_info = "## 欢迎使用风电智能分析服务    \n" \
    # " Code From  " \


# "[ChatGPT-Web](https://github.com/LiangYang666/ChatGPT-Web)  \n" \
# "发送`帮助`可获取帮助  \n"


def get_response_from_ChatGPT_API(message_context, apikey,
                                  model="gpt-3.5-turbo-16k", temperature=0.9, presence_penalty=0, max_tokens=2000):
    """
    从ChatGPT API获取回复
    :param message_context: 上下文
    :param apikey: API KEY
    :param model: 模型
    :param temperature: 温度
    :param presence_penalty: 惩罚
    :param max_tokens: 最大token数量
    :return: 回复
    """
    if apikey is None:
        apikey = API_KEY

    header = {"Content-Type": "application/json",
              "Library": 'Venti',
              "Authorization": "Bearer " + apikey}

    data = {
        "model": model,
        "messages": message_context,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens
    }
    url = "https://api.openai.com/v1/chat/completions"
    # url = "https://chat-api.leyoubaloy.xyz/v1"
    try:
        response = requests.post(url, headers=header, data=json.dumps(data))
        response = response.json()
        # 判断是否含 choices[0].message.content
        if "choices" in response \
                and len(response["choices"]) > 0 \
                and "message" in response["choices"][0] \
                and "content" in response["choices"][0]["message"]:
            data = response["choices"][0]["message"]["content"]
        else:
            data = str(response)

    except Exception as e:
        print(e)
        return str(e)

    return data


def get_message_context(message_history, have_chat_context, chat_with_history):
    """
    获取上下文
    :param message_history:
    :param have_chat_context:
    :param chat_with_history:
    :return:
    """
    message_context = []
    total = 0
    if chat_with_history:
        num = min([len(message_history), CHAT_CONTEXT_NUMBER_MAX, have_chat_context])
        # 获取所有有效聊天记录
        valid_start = 0
        valid_num = 0
        for i in range(len(message_history) - 1, -1, -1):
            message = message_history[i]
            if message['role'] in {'assistant', 'user'}:
                valid_start = i
                valid_num += 1
            if valid_num >= num:
                break

        for i in range(valid_start, len(message_history)):
            message = message_history[i]
            if message['role'] in {'assistant', 'user'}:
                message_context.append(message)
                total += len(message['content'])
    else:
        message_context.append(message_history[-1])
        total += len(message_history[-1]['content'])

    print(f"len(message_context): {len(message_context)} total: {total}", )
    return message_context


def handle_messages_get_response(message, apikey, message_history, have_chat_context, chat_with_history):
    """
    处理用户发送的消息，获取回复
    :param message: 用户发送的消息
    :param apikey:
    :param message_history: 消息历史
    :param have_chat_context: 已发送消息数量上下文(从重置为连续对话开始)
    :param chat_with_history: 是否连续对话
    """
    message_history.append({"role": "user", "content": message})
    message_context = get_message_context(message_history, have_chat_context, chat_with_history)
    response = get_response_from_ChatGPT_API(message_context, apikey)
    message_history.append({"role": "assistant", "content": response})
    # 换行打印messages_history
    # print("message_history:")
    # for i, message in enumerate(message_history):
    #     if message['role'] == 'user':
    #         print(f"\t{i}:\t{message['role']}:\t\t{message['content']}")
    #     else:
    #         print(f"\t{i}:\t{message['role']}:\t{message['content']}")

    return response


def get_response_stream_generate_from_ChatGPT_API(message_context, apikey, message_history,
                                                  model="gpt-3.5-turbo", temperature=0.9, presence_penalty=0,
                                                  max_tokens=2000):
    """
    从ChatGPT API获取回复
    :param apikey:
    :param message_context: 上下文
    :param message_history: 消息历史
    :param model: 模型
    :param temperature: 温度
    :param presence_penalty: 惩罚
    :param max_tokens: 最大token数量
    :return: 回复生成器
    """
    if apikey is None:
        apikey = API_KEY

    header = {"Content-Type": "application/json",
              "Library": 'Venti',
              "Authorization": "Bearer " + apikey}

    data = {
        "model": model,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "messages": message_context,
        "stream": True
    }

    print("开始流式请求")
    url = "https://api.openai.com/v1/chat/completions"
    # data_wenxin = message_context[0]['content']
    # ernie = Ernie('***',
    #               '***')
    # sessionId = ernie.newConversation('风电')
    # 请求接收流式数据 动态print
    try:
        response = requests.request("POST", url, headers=header, json=data, stream=True)

        def generate():
            stream_content = str()
            one_message = {"role": "assistant", "content": stream_content}
            message_history.append(one_message)
            i = 0
            for line in response.iter_lines():
                # print(str(line))
                line_str = str(line, encoding='utf-8')
                if line_str.startswith("data:"):
                    if line_str.startswith("data: [DONE]"):
                        asyncio.run(save_all_user_dict())
                        break
                    line_json = json.loads(line_str[5:])
                    if 'choices' in line_json:
                        if len(line_json['choices']) > 0:
                            choice = line_json['choices'][0]
                            if 'delta' in choice:
                                delta = choice['delta']
                                if 'role' in delta:
                                    role = delta['role']
                                elif 'content' in delta:
                                    delta_content = delta['content']
                                    i += 1
                                    if i < 40:
                                        print(delta_content, end="")
                                    elif i == 40:
                                        print("......")
                                    one_message['content'] = one_message['content'] + delta_content
                                    yield delta_content

                elif len(line_str.strip()) > 0:
                    print(line_str)
                    yield line_str

        # def generate0():
        #     tmp = ernie.askStream(data_wenxin, sessionId, '0')
        #     for i in tmp:
        #         if not i['done']:
        #             yield i['answer']
        #         else:
        #             one_message = {"role": "assistant", "content": i['answer']}
        #             message_history.append(one_message)
    except Exception as e:
        ee = e

        def generate():
            yield "request error:\n" + str(ee)

    return generate


def handle_messages_get_response_stream(message, apikey, message_history, have_chat_context, chat_with_history):
    message_history.append({"role": "user", "content": message})
    asyncio.run(save_all_user_dict())
    message_context = get_message_context(message_history, have_chat_context, chat_with_history)
    generate = get_response_stream_generate_from_ChatGPT_API(message_context, apikey, message_history)
    return generate


def check_session(current_session):
    """
    检查session，如果不存在则创建新的session
    :param current_session: 当前session
    :return: 当前session
    """
    if current_session.get('session_id') is not None:
        print("existing session, session_id:\t", current_session.get('session_id'))
    else:
        current_session['session_id'] = uuid.uuid1()
        print("new session, session_id:\t", current_session.get('session_id'))
    return current_session['session_id']


def check_user_bind(current_session):
    """
    检查用户是否绑定，如果没有绑定则重定向到index
    :param current_session: 当前session
    :return: 当前session
    """
    if current_session.get('user_id') is None:
        return False
    return True


def get_user_info(user_id):
    """
    获取用户信息
    :param user_id: 用户id
    :return: 用户信息
    """
    lock.acquire()
    user_info = all_user_dict.get(user_id)
    lock.release()
    return user_info


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     """
#     主页
#     :return: 主页
#     """
#     check_session(session)
#     return render_template('dialog.html')


@app.route('/loadHistory', methods=['GET', 'POST'])
def load_messages():
    """
    加载聊天记录
    :return: 聊天记录
    """
    check_session(session)
    success, message = auth(request.headers, session)
    code = 200  # 200表示云端存储了 node.js改写时若云端不存储则返回201
    if not success:
        return {"code": code, "data": [{"role": "web-system", "content": message}]}
    if session.get('user_id') is None:
        messages_history = [{"role": "assistant", "content": project_info},
                            {"role": "assistant", "content": "#### 当前浏览器会话为首次请求\n"
                                                             "#### 请输入已有用户`id`或创建新的用户`id`。\n"
                                                             "- 已有用户`id`请在输入框中直接输入\n"
                                                             "- 创建新的用户`id`请在输入框中输入`new:xxx`,其中`xxx`为你的自定义id，请牢记\n"
                                                             "- 输入`帮助`以获取帮助提示"}]
    else:
        user_info = get_user_info(session.get('user_id'))
        chat_id = user_info['selected_chat_id']
        messages_history = user_info['chats'][chat_id]['messages_history']
        print(f"用户({session.get('user_id')})加载聊天记录，共{len(messages_history)}条记录")
    return {"code": code, "data": messages_history, "message": ""}


@app.route('/downloadUserDictFile', methods=['GET', 'POST'])
def download_user_dict_file():
    """
    下载用户字典文件
    :return: 用户字典文件
    """
    check_session(session)
    if request.headers.get("admin-password") is None:
        success, message = auth(request.headers, session)
        if not success:
            return "未授权，无法下载"
        user_id = request.headers.get("user-id")
        if user_id is None:
            return "未绑定用户，无法下载"
        select_user_dict = LRUCache(USER_SAVE_MAX)
        lock.acquire()
        select_user_dict.put(user_id, all_user_dict.get(user_id))
        lock.release()
        # 存储为临时文件再发送出去
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, mode='wb') as temp_file:
            # 将 Python 对象使用 pickle 序列化保存到临时文件中
            pickle.dump(select_user_dict, temp_file)
        response = make_response(send_file(temp_file.name, as_attachment=True))
        response.headers["Content-Disposition"] = f"attachment; filename={user_id}_of_{USER_DICT_FILE}"
        response.call_on_close(lambda: os.remove(temp_file.name))
        return response

    else:
        if request.headers.get("admin-password") != ADMIN_PASSWORD:
            return "管理员密码错误，无法下载"
        response = make_response(send_file(USER_DICT_FILE, as_attachment=True))
        response.headers["Content-Disposition"] = f"attachment; filename={USER_DICT_FILE}"
        return response


def backup_user_dict_file():
    """
    备份用户字典文件
    :return:
    """
    backup_file_name = USER_DICT_FILE.replace(".pkl",
                                              f"_buckup_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.pkl")
    shutil.copy(USER_DICT_FILE, backup_file_name)
    print(f"备份用户字典文件{USER_DICT_FILE}为{backup_file_name}")


@app.route('/uploadUserDictFile', methods=['POST'])
def upload_user_dict_file():
    """
    上传用户字典文件 并合并记录
    :return:
    """
    check_session(session)
    file = request.files.get('file')  # 获取上传的文件
    if file:
        if request.headers.get("admin-password") is None:
            success, message = auth(request.headers, session)
            if not success:
                return "未授权，无法合并用户记录"
            user_id = request.headers.get("user-id")
            if user_id is None:
                return "未绑定用户，无法合并用户记录"
            if not file.filename.endswith(".pkl"):
                return "上传文件格式错误，无法合并用户记录"

            # 读取获取的文件
            upload_user_dict = ""
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, mode='wb') as temp_file:
                file.save(temp_file.name)
            # 将 Python 对象使用 pickle 序列化保存到临时文件中
            try:
                with open(temp_file.name, 'rb') as temp_file:
                    upload_user_dict = pickle.load(temp_file)
            except:
                return "上传文件格式错误，无法解析以及合并用户记录"
            finally:
                os.remove(temp_file.name)
            # 判断是否为LRUCache对象
            if not isinstance(upload_user_dict, LRUCache):
                return "上传文件格式错误，无法合并用户记录"
            lock.acquire()
            user_info = all_user_dict.get(user_id)
            lock.release()
            upload_user_info = upload_user_dict.get(user_id)
            if user_info is None or upload_user_info is None:
                return "仅能合并相同用户id的记录，请确保所上传的记录与当前用户id一致"
            backup_user_dict_file()
            for chat_id in upload_user_info['chats'].keys():
                if user_info['chats'].get(chat_id) is None:
                    user_info['chats'][chat_id] = upload_user_info['chats'][chat_id]
                else:
                    new_chat_id = str(uuid.uuid1())
                    user_info['chats'][new_chat_id] = upload_user_info['chats'][chat_id]
            asyncio.run(save_all_user_dict())
            return '个人用户记录合并完成'
        else:
            if request.headers.get("admin-password") != ADMIN_PASSWORD:
                return "管理员密码错误，无法上传用户记录"
            if not file.filename.endswith(".pkl"):
                return "上传文件格式错误，无法上传用户记录"
            # 读取获取的文件
            upload_user_dict = ""
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, mode='wb') as temp_file:
                file.save(temp_file.name)
            # 将 Python 对象使用 pickle 序列化保存到临时文件中
            try:
                with open(temp_file.name, 'rb') as temp_file:
                    upload_user_dict = pickle.load(temp_file)
            except:
                return "上传文件格式错误，无法解析以及合并用户记录"
            finally:
                os.remove(temp_file.name)
            # 判断是否为LRUCache对象
            if not isinstance(upload_user_dict, LRUCache):
                return "上传文件格式错误，无法合并用户记录"
            backup_user_dict_file()
            lock.acquire()
            for user_id in list(upload_user_dict.keys()):
                if all_user_dict.get(user_id) is None:
                    all_user_dict.put(user_id, upload_user_dict.get(user_id))
                else:
                    for chat_id in upload_user_dict.get(user_id)['chats'].keys():
                        if all_user_dict.get(user_id)['chats'].get(chat_id) is None:
                            all_user_dict.get(user_id)['chats'][chat_id] = upload_user_dict.get(user_id)['chats'][
                                chat_id]
                        else:
                            new_chat_id = str(uuid.uuid1())
                            all_user_dict.get(user_id)['chats'][new_chat_id] = upload_user_dict.get(user_id)['chats'][
                                chat_id]
            lock.release()
            asyncio.run(save_all_user_dict())
            return '所有用户记录合并完成'
    else:
        return '文件上传失败'


def auth(request_head, session):
    """
    验证用户身份
    :param request_head: 请求头
    :param session: session
    :return: 验证结果
    """
    user_id = request_head.get("user-id")
    user_id = urllib.parse.unquote(user_id)
    password = request_head.get("password")
    apikey = request_head.get("api-key")

    user_info = get_user_info(user_id)
    if len(PASSWORD) > 0 and password != PASSWORD:
        return False, "访问密码错误，请在设置中填写正确的访问密码"

    if user_info is not None:
        session['user_id'] = user_id
        if apikey is not None and len(apikey) > 1 and apikey != "null" and apikey != "undefined":
            user_info['apikey'] = apikey
        else:
            user_info['apikey'] = None
        return True, "success"
    else:
        if session.get('user_id') is not None:
            del session['user_id']
        return False, "用户不存在，请在设置中填写正确的用户id，或发送new:xxx创建新的用户，其中xxx为你的自定义id"


@app.route('/loadChats', methods=['GET', 'POST'])
def load_chats():
    """
    加载聊天联系人
    :return: 聊天联系人
    """
    check_session(session)
    success, message = auth(request.headers, session)

    if not check_user_bind(session) or not success:
        chats = []
    else:
        user_info = get_user_info(session.get('user_id'))
        chats = []
        for chat_id, chat_info in user_info['chats'].items():
            if chat_info['chat_with_history']:
                mode = "continuous"
            else:
                mode = "normal"
            chats.append(
                {"id": chat_id, "name": chat_info['name'], "selected": chat_id == user_info['selected_chat_id'],
                 "mode": mode, "messages_total": len(user_info['chats'][chat_id]['messages_history'])})
    code = 200  # 200表示云端存储了 node.js改写时若云端不存储则返回201
    return {"code": code, "data": chats, "message": ""}


def new_chat_dict(user_id, name, send_time):
    return {"chat_with_history": False,
            "have_chat_context": 0,  # 从每次重置聊天模式后开始重置一次之后累计
            "name": name,
            "messages_history": [{"role": "assistant", "content": project_info},
                                 {"role": "web-system", "content": f"当前对话的用户id为{user_id}"},
                                 {"role": "web-system", "content": send_time},
                                 {"role": "web-system", "content": f"你已添加了{name}，现在可以开始聊天了。"},
                                 ]}


def new_user_dict(user_id, send_time):
    chat_id = str(uuid.uuid1())
    user_dict = {"chats": {chat_id: new_chat_dict(user_id, "ERNIE智能对话分析", send_time)},
                 "selected_chat_id": chat_id,
                 "default_chat_id": chat_id}

    user_dict['chats'][chat_id]['messages_history'].insert(1, {"role": "assistant",
                                                               "content": "创建新的用户id成功，请牢记该id"})
    return user_dict


def get_balance(apikey):
    head = ""
    if apikey is not None:
        head = "###  用户专属api key余额  \n"
    else:
        head = "### 通用api key  \n"
        apikey = API_KEY

    subscription_url = "https://api.openai.com/v1/dashboard/billing/subscription"
    headers = {
        "Authorization": "Bearer " + apikey,
        "Content-Type": "application/json"
    }
    subscription_response = requests.get(subscription_url, headers=headers)
    if subscription_response.status_code == 200:
        data = subscription_response.json()
        total = data.get("hard_limit_usd")
    else:
        return head + subscription_response.text

    # start_date设置为今天日期前99天
    start_date = (datetime.datetime.now() - datetime.timedelta(days=99)).strftime("%Y-%m-%d")
    # end_date设置为今天日期+1
    end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    billing_url = f"https://api.openai.com/v1/dashboard/billing/usage?start_date={start_date}&end_date={end_date}"
    billing_response = requests.get(billing_url, headers=headers)
    if billing_response.status_code == 200:
        data = billing_response.json()
        total_usage = data.get("total_usage") / 100
        daily_costs = data.get("daily_costs")
        days = min(5, len(daily_costs))
        recent = f"##### 最近{days}天使用情况  \n"
        for i in range(days):
            cur = daily_costs[-i - 1]
            date = datetime.datetime.fromtimestamp(cur.get("timestamp")).strftime("%Y-%m-%d")
            line_items = cur.get("line_items")
            cost = 0
            for item in line_items:
                cost += item.get("cost")
            recent += f"\t{date}\t{cost / 100} \n"
    else:
        return head + billing_response.text

    return head + f"\n#### 总额:\t{total:.4f}  \n" \
                  f"#### 已用:\t{total_usage:.4f}  \n" \
                  f"#### 剩余:\t{total - total_usage:.4f}  \n" \
                  f"\n" + recent


def get_time_str(time):
    year = time.year  # 得到年份
    month = time.month  # 得到月份
    date = time.day  # 得到日期
    hour = time.hour  # 得到小时
    if hour < 10:
        hour = "0" + str(hour)
    minute = time.minute  # 得到分钟
    if minute < 10:
        minute = "0" + str(minute)
    return f"{year}年{month}月{date}日 {hour}:{minute}"


@app.route('/createUser', methods=['GET', 'POST'])
def createUser():
    username = request.args.get("username")
    url_redirect = {"url_redirect": "/predict", "user_id": None}
    user_id = username
    url_redirect["user_id"] = user_id
    if user_id in all_user_dict:
        session['user_id'] = user_id
        return url_redirect

    time = datetime.now()  # 获取当前时间
    time = get_time_str(time)

    user_dict = new_user_dict(user_id, time)
    lock.acquire()
    all_user_dict.put(user_id, user_dict)  # 默认普通对话
    lock.release()
    print("创建新的用户id:\t", user_id)
    session['user_id'] = user_id
    return url_redirect


@app.route('/returnMessage', methods=['GET', 'POST'])
def return_message():
    """
    获取用户发送的消息，调用get_chat_response()获取回复，返回回复，用于更新聊天框
    :return:
    """
    check_session(session)
    request_data = request.get_json()

    success, message = auth(request.headers, session)
    if not success:
        session.clear()

    messages = request_data.get("messages")
    print('message->', messages)
    max_tokens = request_data.get("max_tokens")
    model = request_data.get("model")
    temperature = request_data.get("temperature")
    stream = request_data.get("stream")
    continuous_chat = request_data.get("continuous_chat")
    save_message = request_data.get("save_message")

    send_message = messages[-1].get("content")
    print('send_message', send_message)
    send_time = messages[-1].get("send_time")
    display_time = bool(messages[-1].get("display_time"))
    url_redirect = {"url_redirect": "/", "user_id": None}
    if send_message == "帮助":
        return "### 帮助\n" \
               "1. 输入`new:xxx`创建新的用户id\n " \
               "2. 输入`id:your_id`切换到已有用户id，新会话时无需加`id:`进入已有用户\n" \
               "3. 输入`rename_id:xxx`可将当前用户id更改\n" \
               "4. 输入`查余额`可获得余额信息及最近几天使用量\n" \
               "5. 相关设置也可以在设置面板中进行设置\n" \
               "6. 输入`帮助`查看帮助信息"
    if session.get('user_id') is None:  # 如果当前session未绑定用户
        print("当前会话为首次请求，用户输入:\t", send_message)
        if send_message.startswith("new:"):
            user_id = send_message.split(":")[1]
            url_redirect["user_id"] = user_id
            if user_id in all_user_dict:
                session['user_id'] = user_id
                return url_redirect
            user_dict = new_user_dict(user_id, send_time)
            lock.acquire()
            all_user_dict.put(user_id, user_dict)  # 默认普通对话
            lock.release()
            print("创建新的用户id:\t", user_id)
            session['user_id'] = user_id
            url_redirect["user_id"] = user_id
            return url_redirect
        else:
            user_id = send_message
            user_info = get_user_info(user_id)
            if user_info is None:
                return "用户id不存在，请重新输入或创建新的用户id"
            else:
                session['user_id'] = user_id
                print("已有用户id:\t", user_id)
                # 重定向到index
                url_redirect["user_id"] = user_id
                return url_redirect
    else:  # 当存在用户id时
        if send_message.startswith("id:"):
            user_id = send_message.split(":")[1].strip()
            user_info = get_user_info(user_id)
            if user_info is None:
                return "用户id不存在，请重新输入或创建新的用户id"
            else:
                session['user_id'] = user_id
                url_redirect["user_id"] = user_id
                print("切换到已有用户id:\t", user_id)
                # 重定向到index
                return url_redirect
        elif send_message.startswith("new:"):
            user_id = send_message.split(":")[1]
            if user_id in all_user_dict:
                return "用户id已存在，请重新输入或切换到已有用户id"
            session['user_id'] = user_id
            url_redirect["user_id"] = user_id
            user_dict = new_user_dict(user_id, send_time)
            lock.acquire()
            all_user_dict.put(user_id, user_dict)
            lock.release()
            print("创建新的用户id:\t", user_id)
            return url_redirect
        elif send_message.startswith("delete:"):  # 删除用户
            user_id = send_message.split(":")[1]
            if user_id != session.get('user_id'):
                return "只能删除当前会话的用户id"
            else:
                lock.acquire()
                all_user_dict.delete(user_id)
                lock.release()
                session['user_id'] = None
                print("删除用户id:\t", user_id)
                # 异步存储all_user_dict
                asyncio.run(save_all_user_dict())
                return url_redirect
        elif send_message.startswith("set_apikey:"):
            apikey = send_message.split(":")[1]
            user_info = get_user_info(session.get('user_id'))
            user_info['apikey'] = apikey
            # TODO 前端未存储
            print("设置用户专属apikey:\t", apikey)
            return "设置用户专属apikey成功"
        elif send_message.startswith("rename_id:"):
            new_user_id = send_message.split(":")[1]
            user_info = get_user_info(session.get('user_id'))
            if new_user_id in all_user_dict:
                return "用户id已存在，请重新输入"
            else:
                lock.acquire()
                all_user_dict.delete(session['user_id'])
                all_user_dict.put(new_user_id, user_info)
                lock.release()
                session['user_id'] = new_user_id
                asyncio.run(save_all_user_dict())
                print("修改用户id:\t", new_user_id)
                url_redirect["user_id"] = new_user_id
                return url_redirect
        elif send_message == "查余额":
            user_info = get_user_info(session.get('user_id'))
            apikey = user_info.get('apikey')
            return get_balance(apikey)
        else:  # 处理聊天数据
            user_id = session.get('user_id')
            print(f"用户({user_id})发送消息:{send_message}")
            user_info = get_user_info(user_id)
            chat_id = user_info['selected_chat_id']
            messages_history = user_info['chats'][chat_id]['messages_history']
            chat_with_history = user_info['chats'][chat_id]['chat_with_history']
            apikey = user_info.get('apikey')
            if chat_with_history:
                user_info['chats'][chat_id]['have_chat_context'] += 1
            if display_time:
                messages_history.append({'role': 'web-system', "content": send_time})
            for m in messages:
                keys = list(m.keys())
                for k in keys:
                    if k not in ['role', 'content']:
                        del m[k]
            if not STREAM_FLAG:
                print('here1')
                if save_message:
                    messages_history.append(messages[-1])
                response = get_response_from_ChatGPT_API(messages, apikey)
                if save_message:
                    messages_history.append({"role": "assistant", "content": response})
                asyncio.run(save_all_user_dict())

                print(f"用户({session.get('user_id')})得到的回复消息:{response[:40]}...")
                # 异步存储all_user_dict
                asyncio.run(save_all_user_dict())
                return response
            else:
                print('here2')
                if save_message:
                    messages_history.append(messages[-1])
                asyncio.run(save_all_user_dict())
                if not save_message:
                    messages_history = []
                generate = get_response_stream_generate_from_ChatGPT_API(messages, apikey, messages_history,
                                                                         model=model, temperature=temperature,
                                                                         max_tokens=max_tokens)
                return app.response_class(generate(), mimetype='application/json')


@app.route('/returnMessage1', methods=['GET', 'POST'])
def return_message1():
    """
    获取用户发送的消息，调用get_chat_response()获取回复，返回回复，用于更新聊天框
    :return:
    """
    check_session(session)
    request_data = request.get_json()

    success, message = auth(request.headers, session)
    if not success:
        session.clear()

    messages = request_data.get("messages")
    print('message->', messages)
    max_tokens = request_data.get("max_tokens")
    model = request_data.get("model")
    temperature = request_data.get("temperature")
    stream = request_data.get("stream")
    continuous_chat = request_data.get("continuous_chat")
    save_message = request_data.get("save_message")

    send_message = messages[-1].get("content")
    print('send_message', send_message)
    send_time = messages[-1].get("send_time")
    display_time = bool(messages[-1].get("display_time"))
    url_redirect = {"url_redirect": "/", "user_id": None}
    if send_message == "帮助":
        return "### 帮助\n" \
               "1. 输入`new:xxx`创建新的用户id\n " \
               "2. 输入`id:your_id`切换到已有用户id，新会话时无需加`id:`进入已有用户\n" \
               "3. 输入`rename_id:xxx`可将当前用户id更改\n" \
               "4. 输入`查余额`可获得余额信息及最近几天使用量\n" \
               "5. 相关设置也可以在设置面板中进行设置\n" \
               "6. 输入`帮助`查看帮助信息"
    if session.get('user_id') is None:  # 如果当前session未绑定用户
        print("当前会话为首次请求，用户输入:\t", send_message)
        if send_message.startswith("new:"):
            user_id = send_message.split(":")[1]
            url_redirect["user_id"] = user_id
            if user_id in all_user_dict:
                session['user_id'] = user_id
                return url_redirect
            user_dict = new_user_dict(user_id, send_time)
            lock.acquire()
            all_user_dict.put(user_id, user_dict)  # 默认普通对话
            lock.release()
            print("创建新的用户id:\t", user_id)
            session['user_id'] = user_id
            url_redirect["user_id"] = user_id
            return url_redirect
        else:
            user_id = send_message
            user_info = get_user_info(user_id)
            if user_info is None:
                return "用户id不存在，请重新输入或创建新的用户id"
            else:
                session['user_id'] = user_id
                print("已有用户id:\t", user_id)
                # 重定向到index
                url_redirect["user_id"] = user_id
                return url_redirect
    else:  # 当存在用户id时
        if send_message.startswith("id:"):
            user_id = send_message.split(":")[1].strip()
            user_info = get_user_info(user_id)
            if user_info is None:
                return "用户id不存在，请重新输入或创建新的用户id"
            else:
                session['user_id'] = user_id
                url_redirect["user_id"] = user_id
                print("切换到已有用户id:\t", user_id)
                # 重定向到index
                return url_redirect
        elif send_message.startswith("new:"):
            user_id = send_message.split(":")[1]
            if user_id in all_user_dict:
                return "用户id已存在，请重新输入或切换到已有用户id"
            session['user_id'] = user_id
            url_redirect["user_id"] = user_id
            user_dict = new_user_dict(user_id, send_time)
            lock.acquire()
            all_user_dict.put(user_id, user_dict)
            lock.release()
            print("创建新的用户id:\t", user_id)
            return url_redirect
        elif send_message.startswith("delete:"):  # 删除用户
            user_id = send_message.split(":")[1]
            if user_id != session.get('user_id'):
                return "只能删除当前会话的用户id"
            else:
                lock.acquire()
                all_user_dict.delete(user_id)
                lock.release()
                session['user_id'] = None
                print("删除用户id:\t", user_id)
                # 异步存储all_user_dict
                asyncio.run(save_all_user_dict())
                return url_redirect
        elif send_message.startswith("set_apikey:"):
            apikey = send_message.split(":")[1]
            user_info = get_user_info(session.get('user_id'))
            user_info['apikey'] = apikey
            # TODO 前端未存储
            print("设置用户专属apikey:\t", apikey)
            return "设置用户专属apikey成功"
        elif send_message.startswith("rename_id:"):
            new_user_id = send_message.split(":")[1]
            user_info = get_user_info(session.get('user_id'))
            if new_user_id in all_user_dict:
                return "用户id已存在，请重新输入"
            else:
                lock.acquire()
                all_user_dict.delete(session['user_id'])
                all_user_dict.put(new_user_id, user_info)
                lock.release()
                session['user_id'] = new_user_id
                asyncio.run(save_all_user_dict())
                print("修改用户id:\t", new_user_id)
                url_redirect["user_id"] = new_user_id
                return url_redirect
        elif send_message == "查余额":
            user_info = get_user_info(session.get('user_id'))
            apikey = user_info.get('apikey')
            return get_balance(apikey)
        else:  # 处理聊天数据
            user_id = session.get('user_id')
            print(f"用户({user_id})发送消息:{send_message}")
            user_info = get_user_info(user_id)
            chat_id = user_info['selected_chat_id']
            messages_history = user_info['chats'][chat_id]['messages_history']
            chat_with_history = user_info['chats'][chat_id]['chat_with_history']
            apikey = user_info.get('apikey')
            if chat_with_history:
                user_info['chats'][chat_id]['have_chat_context'] += 1
            if display_time:
                messages_history.append({'role': 'web-system', "content": send_time})
            for m in messages:
                keys = list(m.keys())
                for k in keys:
                    if k not in ['role', 'content']:
                        del m[k]
            if not STREAM_FLAG:
                print('here1')
                if save_message:
                    messages_history.append(messages[-1])
                response = get_response_from_ChatGPT_API(messages, apikey)
                if save_message:
                    messages_history.append({"role": "assistant", "content": response})
                asyncio.run(save_all_user_dict())

                print(f"用户({session.get('user_id')})得到的回复消息:{response[:40]}...")
                # 异步存储all_user_dict
                asyncio.run(save_all_user_dict())
                return response
            else:
                print('here2')
                # if save_message:
                # messages_history.append(messages[-1])
                asyncio.run(save_all_user_dict())
                if not save_message:
                    messages_history = []
                generate = get_response_stream_generate_from_ChatGPT_API(messages, apikey, messages_history,
                                                                         model=model, temperature=temperature,
                                                                         max_tokens=max_tokens)
                return app.response_class(generate(), mimetype='application/json')


async def save_all_user_dict():
    """
    异步存储all_user_dict
    :return:
    """
    await asyncio.sleep(0)
    lock.acquire()
    with open(USER_DICT_FILE, "wb") as f:
        pickle.dump(all_user_dict, f)
    # print("all_user_dict.pkl存储成功")
    lock.release()


@app.route('/selectChat', methods=['GET'])
def select_chat():
    """
    选择聊天对象
    :return:
    """
    chat_id = request.args.get("id")
    check_session(session)
    if not check_user_bind(session):
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_id = session.get('user_id')
    user_info = get_user_info(user_id)
    user_info['selected_chat_id'] = chat_id
    return {"code": 200, "msg": "选择聊天对象成功"}


@app.route('/newChat', methods=['GET'])
def new_chat():
    """
    新建聊天对象
    :return:
    """
    name = request.args.get("name")
    time = request.args.get("time")
    new_chat_id = request.args.get("chat_id")
    check_session(session)
    if not check_user_bind(session):
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_id = session.get('user_id')
    user_info = get_user_info(user_id)
    # new_chat_id = str(uuid.uuid1())
    user_info['selected_chat_id'] = new_chat_id
    user_info['chats'][new_chat_id] = new_chat_dict(user_id, name, time)
    print("新建聊天对象")
    return {"code": 200, "data": {"name": name, "id": new_chat_id, "selected": True,
                                  "messages_total": len(user_info['chats'][new_chat_id]['messages_history'])}}


@app.route('/deleteHistory', methods=['GET'])
def delete_history():
    """
    清空上下文
    :return:
    """
    check_session(session)
    if not check_user_bind(session):
        print("请先创建或输入已有用户id")
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_info = get_user_info(session.get('user_id'))
    chat_id = user_info['selected_chat_id']
    default_chat_id = user_info['default_chat_id']
    if default_chat_id == chat_id:
        print("清空历史记录")
        user_info["chats"][chat_id]['messages_history'] = user_info["chats"][chat_id]['messages_history'][:5]
    else:
        print("删除聊天对话")
        del user_info["chats"][chat_id]
    user_info['selected_chat_id'] = default_chat_id
    return "2"


def check_load_pickle():
    global all_user_dict

    if os.path.exists(USER_DICT_FILE):
        with open(USER_DICT_FILE, "rb") as pickle_file:
            all_user_dict = pickle.load(pickle_file)
            all_user_dict.change_capacity(USER_SAVE_MAX)
        print(f"已加载上次存储的用户上下文，共有{len(all_user_dict)}用户, 分别是")
        for i, user_id in enumerate(list(all_user_dict.keys())):
            print(f"{i} 用户id:{user_id}\t对话统计:\t", end="")
            user_info = all_user_dict.get(user_id)
            for chat_id in user_info['chats'].keys():
                print(f"{user_info['chats'][chat_id]['name']}[{len(user_info['chats'][chat_id]['messages_history'])}] ",
                      end="")
            print()
    elif os.path.exists("all_user_dict_v2.pkl"):  # 适配V2
        print('检测到v2版本的上下文，将转换为v3版本')
        with open("all_user_dict_v2.pkl", "rb") as pickle_file:
            all_user_dict = pickle.load(pickle_file)
            all_user_dict.change_capacity(USER_SAVE_MAX)
        print("共有用户", len(all_user_dict), "个")
        for user_id in list(all_user_dict.keys()):
            user_info: dict = all_user_dict.get(user_id)
            for chat_id in user_info['chats'].keys():
                if "messages_history" in user_info['chats'][chat_id]:
                    for i in range(len(user_info['chats'][chat_id]['messages_history'])):
                        # 将system关键字改为 web-system
                        if "role" in user_info['chats'][chat_id]['messages_history'][i] and \
                                user_info['chats'][chat_id]['messages_history'][i]["role"] == "system":
                            user_info['chats'][chat_id]['messages_history'][i]["role"] = "web-system"

        asyncio.run(save_all_user_dict())

    elif os.path.exists("all_user_dict.pkl"):  # 适配当出现这个时
        print('检测到v1版本的上下文，将转换为v3版本')
        with open("all_user_dict.pkl", "rb") as pickle_file:
            all_user_dict = pickle.load(pickle_file)
            all_user_dict.change_capacity(USER_SAVE_MAX)
        print("共有用户", len(all_user_dict), "个")
        for user_id in list(all_user_dict.keys()):
            user_info: dict = all_user_dict.get(user_id)
            if "messages_history" in user_info:
                user_dict = new_user_dict(user_id, "")
                chat_id = user_dict['selected_chat_id']
                user_dict['chats'][chat_id]['messages_history'] = user_info['messages_history']
                user_dict['chats'][chat_id]['chat_with_history'] = user_info['chat_with_history']
                user_dict['chats'][chat_id]['have_chat_context'] = user_info['have_chat_context']
                all_user_dict.put(user_id, user_dict)  # 更新
        asyncio.run(save_all_user_dict())
    else:
        with open(USER_DICT_FILE, "wb") as pickle_file:
            pickle.dump(all_user_dict, pickle_file)
        print("未检测到上次存储的用户上下文，已创建新的用户上下文")

    # 判断all_user_dict是否为None且时LRUCache的对象
    if all_user_dict is None or not isinstance(all_user_dict, LRUCache):
        print("all_user_dict为空或不是LRUCache对象，已创建新的LRUCache对象")
        all_user_dict = LRUCache(USER_SAVE_MAX)


# _________________________________________________________________________数据库连接_________________________________________________________________________
# 创建 pymysqlpool 连接池
def get_connection():
    connection = pymysql.connect(host=DB_CONFIG['host'], port=DB_CONFIG['port'], user=DB_CONFIG['user'],
                                 password=DB_CONFIG['password'], db=DB_CONFIG['database'])
    return connection


# _________________________________________________________________________功能性函数_________________________________________________________________________
# 原有模型的预测准确度
def res_predict(df_path, model_path):
    df = pd.read_csv(df_path)
    model = joblib.load(model_path)
    output = model.predict(df[['WINDSPEED', 'WINDSPEED2']])
    return output


origin_model_resYD15 = res_predict('模型相似度检测/test.csv', 'usingmodels/model1.pkl')
origin_model_resPOWER = res_predict('模型相似度检测/test.csv', 'usingmodels/model2.pkl')


def getMAE(ndarray1, ndarray2):
    diff = np.abs(ndarray1 - ndarray2)
    mae = np.mean(diff)
    return mae


# 得到相似度
def getSimilarity(a):
    if a > 2000:
        a = 2000
    a = 2000 - a
    a = a / 2000
    a = a * 0.4 + 0.5
    return a


def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
        return ip


def yd15addmodel(df):
    folder_path = './usingmodels'
    file_names = os.listdir(folder_path)
    cnt_yd15 = 0
    output = None
    for file_name in file_names:
        if file_name == 'model1.pkl' or file_name == 'model2.pkl' or file_name[0] == 'a':
            continue
        else:
            cnt_yd15 = cnt_yd15 + 1
            modelpath = folder_path + '/' + file_name
            tmpmodel = joblib.load(modelpath)
            if output is None:
                output = tmpmodel.predict(df[["WINDSPEED", "WINDSPEED2"]].values)
            else:
                output = output + tmpmodel.predict(df[["WINDSPEED", "WINDSPEED2"]].values)
    if output is None:
        return None
    return output * 0.2 / cnt_yd15


def actualaddmodel(df):
    folder_path = './usingmodels'
    file_names = os.listdir(folder_path)
    cnt_actual = 0
    output = None
    for file_name in file_names:
        if file_name == 'model1.pkl' or file_name == 'model2.pkl' or file_name[0] == 'y':
            continue
        else:
            cnt_actual = cnt_actual + 1
            modelpath = folder_path + '/' + file_name
            tmpmodel = joblib.load(modelpath)
            if output is None:
                output = tmpmodel.predict(df[["WINDSPEED", "WINDSPEED2"]].values)
            else:
                output = output + tmpmodel.predict(df[["WINDSPEED", "WINDSPEED2"]].values)
    if output is None:
        return None
    return output * 0.2 / cnt_actual


# 对两个csv文件进行训练和预测
def train(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    datatimelist = df2['DATATIME'].values

    # 预测YD15
    X_train1 = df1[["WINDSPEED", "WINDSPEED2"]]
    y_train1 = df1[["YD15"]]
    X_test1 = df2[["WINDSPEED", "WINDSPEED2"]]
    # gbm
    x_train, x_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.2)
    gbm1 = LGBMRegressor(objective="regression", learning_rate=0.005, n_estimators=1000, n_jobs=-1)
    gbm1 = gbm1.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric="rmse",
                    callbacks=[early_stopping(stopping_rounds=1000)])
    y_pred15 = gbm1.predict(X_test1)
    # print('y_pred15', y_pred15 * 0.9)
    # print(df2['PREYD15'].values * 0.1)
    output1 = df2['PREYD15'].values * 0.7 + y_pred15 * 0.3
    output1_new = yd15addmodel(X_test1)
    if output1_new is not None:
        output1 = output1 * 0.8 + output1_new

    # 预测POWER
    X_train2 = df1[["WINDSPEED", "WINDSPEED2"]]
    y_train2 = df1[["ROUND(A.POWER,0)"]]
    X_test2 = df2[["WINDSPEED", "WINDSPEED2"]]
    # gbm
    x_train, x_test, y_train, y_test = train_test_split(X_train2, y_train2, test_size=0.2)
    gbm2 = LGBMRegressor(objective="regression", learning_rate=0.005, n_estimators=1000, n_jobs=-1)
    gbm2 = gbm2.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric="rmse",
                    callbacks=[early_stopping(stopping_rounds=1000)])
    POWER = gbm2.predict(X_test2)

    output2 = POWER * 0.3 + df2['PREACTUAL'].values * 0.7
    output2_new = yd15addmodel(X_test1)
    if output2_new is not None:
        output2 = output2 * 0.8 + output2_new

    return [datatimelist.tolist(), output1.tolist(), output2.tolist(), df2['YD15'].values.tolist(),
            df2['ACTUAL'].values.tolist()]


def train2(path1, path2):
    df2 = pd.read_csv(path2)
    datatimelist = df2['DATATIME'].values
    return [datatimelist.tolist(), df2['YD15'].values.tolist(),
            df2['ACTUAL'].values.tolist()]


# 对一个数据集进行预测功率
def upload_predict(data):
    null_count = data['YD15'].isnull().sum()
    session['null_count'] = str(null_count)
    model1 = joblib.load("usingmodels/model1.pkl")
    model2 = joblib.load("usingmodels/model2.pkl")
    df = data[-null_count:]
    data_new = df.copy()
    # 新建一列
    df['WINDSPEED2'] = df['WINDSPEED'] * np.cos(np.radians(df['WINDDIRECTION'].values))
    train = df[["WINDSPEED", "WINDSPEED2"]]
    output1 = model1.predict(train.values)
    output2 = model2.predict(train.values)

    output1_new = yd15addmodel(train)
    output2_new = actualaddmodel(train)

    if output1_new is None:
        data_new['YD15'] = output1
    else:
        data_new['YD15'] = output1 * 0.8 + output1_new

    if output2_new is None:
        data_new['ROUND(A.POWER,0)'] = output2
    else:
        data_new['ROUND(A.POWER,0)'] = output2 * 0.8 + output2_new
    data[-null_count:] = data_new
    path = 'userdata/%s/当前结果文件/tmp.csv' % session.get('username')
    data.to_csv(path, index=False)

    return jsonify({
        'DATATIME': data_new['DATATIME'].values.tolist(),
        'PRE_POWER': np.round(data_new['ROUND(A.POWER,0)'].values, decimals=2).tolist(),
        'PRE_YD15': np.round(data_new['YD15'].values, decimals=2).tolist()
    })


# 登录时创建文件夹
def createfolder(username):
    path1 = 'userdata/%s/上传数据集' % username
    path2 = 'userdata/%s/下载结果文件' % username
    path3 = 'static/usertouxiang/%s' % username
    path4 = 'userdata/%s/当前上传数据集' % username
    path5 = 'userdata/%s/当前结果文件' % username
    os.makedirs(path1, exist_ok=True)
    os.makedirs(path2, exist_ok=True)
    os.makedirs(path3, exist_ok=True)
    os.makedirs(path4, exist_ok=True)
    os.makedirs(path5, exist_ok=True)
    source_file = 'static/picture/touxiang.png'
    destination_file = path3 + '/touxiang.png'
    if os.path.exists(destination_file) == False:
        shutil.copy2(source_file, destination_file)


def to_string(a, f):
    res = ""
    if f == 1:
        for i in range(min(len(a), 50)):
            res = res + str(round(a[i], 2)) + ","
    else:
        for i in range(min(len(a), 50)):
            res = res + a[i] + ","
    return res


# 计算文件夹中文件数量
def count_files_in_folder(folder_path):
    file_count = 0
    # 获取指定文件夹下的所有文件路径
    file_paths = glob.glob(os.path.join(folder_path, '*'))
    for path in file_paths:
        if os.path.isfile(path):  # 只计算文件数量，排除文件夹和子文件夹
            file_count += 1
    return file_count


# 获取路径下的所有文件，返回两个列表，文件名列表，修改时间列表，按修改时间排序
def get_file_paths(directory):
    file_paths = []
    time_list = []
    list = []
    for root, directories, files in os.walk(directory):
        for file in files:
            # file_paths.append(os.path.join(file))
            t = os.path.getmtime(directory + '/' + file)
            # 将时间戳转换为datetime对象
            modified_datetime = datetime.fromtimestamp(t)
            # 格式化为年月日时分秒的格式
            formatted_time = modified_datetime.strftime("%Y-%m-%d %H:%M:%S")
            # time_list.append(formatted_time)
            list.append([os.path.join(file), formatted_time])
    sorted_list = sorted(list, key=lambda x: x[1])
    for i in range(len(sorted_list)):
        file_paths.append(sorted_list[i][0])
        time_list.append(sorted_list[i][1])
    return file_paths, time_list


# _________________________________________________________________________数据库SQL语句_________________________________________________________________________

# 查询供电量
def querypowersupply(id):
    connection = get_connection()
    cursor = connection.cursor()
    sql = "select month,value from powersupply where TurbID = '%s';" % id
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    result_list = [[], []]
    for item in result:
        for i in range(2):
            result_list[i].append(item[i])
    return result_list


# 查询某个id的供发电功率及预测
def queryiddata(id):
    connection = get_connection()
    cursor = connection.cursor()
    sql = "select DATATIME,ACTUAL,PREACTUAL,YD15,PREYD15 from datatmp where TurbID = '%s' LIMIT 96;" % id
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    result_list = [[], [], [], [], []]
    for item in result:
        for i in range(5):
            result_list[i].append(item[i])
    return result_list


def queryonedatabyidandtime(id, year, month, day, hour, minute):
    connection = get_connection()
    cursor = connection.cursor()
    sql = "select DATATIME,ACTUAL,PREACTUAL,YD15,PREYD15 from datatmp where TurbID = '%s' and year = '%s' and month = '%s' and day = '%s' and hour = '%s' and minute = '%s';" % (
        id, year, month, day, hour, minute)
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    result_list = [[], [], [], [], []]
    for item in result:
        for i in range(5):
            result_list[i].append(item[i])
    return result_list


# 给密钥到期时间增加time个月份
def addsdktimemonth(username, time):
    connection = get_connection()
    cursor = connection.cursor()
    sql = "UPDATE usertable SET time = DATE_ADD(time, INTERVAL %s MONTH) WHERE username = '%s';" % (time, username)
    print(sql)
    cursor.execute(sql)
    connection.commit()
    flg = cursor.rowcount
    cursor.close()
    connection.close()
    if flg == 1:
        return True
    else:
        return False


# 查询密钥对应的用户名
def query_sdk_username(sdk):
    connection = get_connection()
    cursor = connection.cursor()
    sql = "select username from usertable where sdk = '%s';" % sdk
    cursor.execute(sql)
    res = cursor.fetchall()
    print(res[0][0])
    cursor.close()
    connection.close()
    if res:
        return res[0]
    return 'false'


# 主图数据查询
def query_pre_data(turbid, year, month, day, hour, length):
    connection = get_connection()
    current_date = datetime(int(year), int(month), int(day), int(hour), 0, 0)
    previous_date = current_date - timedelta(days=int(length))
    current_date = current_date.strftime("%y-%m-%d %H:%M")
    previous_date = previous_date.strftime("%y-%m-%d %H:%M")
    cursor = connection.cursor()
    # 使用 SQL 查询语句从数据库中获取满足条件的数据
    sql = "SELECT DATATIME,WINDSPEED,WINDSPEED2,ACTUAL,YD15 FROM datatmp WHERE TurbID=%s AND STR_TO_DATE(DATATIME, " \
          "'%%Y-%%m-%%d %%H:%%i') >= STR_TO_DATE(%s, '%%Y-%%m-%%d %%H:%%i') AND STR_TO_DATE(DATATIME, '%%Y-%%m-%%d " \
          "%%H:%%i') <= STR_TO_DATE(%s, '%%Y-%%m-%%d %%H:%%i')"
    cursor.execute(sql, (turbid, previous_date, current_date))
    # 获取查询结果
    result = cursor.fetchall()
    result_list = [[], [], [], [], []]
    for item in result:
        for i in range(5):
            result_list[i].append(item[i])
    # 关闭连接
    cursor.close()
    connection.close()
    df = pd.DataFrame({
        'DATATIME': result_list[0],
        'WINDSPEED': result_list[1],
        'WINDSPEED2': result_list[2],
        'ROUND(A.POWER,0)': result_list[3],
        'YD15': result_list[4],
    })
    path = 'userdata/%s/train.csv' % session.get('username')
    df.to_csv(path, index=False)
    return [result_list[0], result_list[3], result_list[4]]


def query_preinput_data(turbid, year, month, day, hour, length):
    connection = get_connection()
    current_date = datetime(int(year), int(month), int(day), int(hour), 0, 0)
    following_date = current_date + timedelta(hours=int(length))
    current_date = current_date.strftime("%y-%m-%d %H:%M")
    following_date = following_date.strftime("%y-%m-%d %H:%M")
    cursor = connection.cursor()
    # 使用 SQL 查询语句从数据库中获取满足条件的数据
    sql = "SELECT DATATIME,WINDSPEED,WINDSPEED2,ACTUAL,PREACTUAL,YD15,PREYD15 FROM datatmp WHERE TurbID=%s AND STR_TO_DATE(DATATIME, " \
          "'%%Y-%%m-%%d %%H:%%i') > STR_TO_DATE(%s, '%%Y-%%m-%%d %%H:%%i') AND STR_TO_DATE(DATATIME, '%%Y-%%m-%%d " \
          "%%H:%%i') <= STR_TO_DATE(%s, '%%Y-%%m-%%d %%H:%%i')"
    cursor.execute(sql, (turbid, current_date, following_date))
    # 获取查询结果
    result = cursor.fetchall()
    result_list = [[], [], [], [], [], [], []]
    for item in result:
        for i in range(7):
            result_list[i].append(item[i])
    df = pd.DataFrame({
        'DATATIME': result_list[0],
        'WINDSPEED': result_list[1],
        'WINDSPEED2': result_list[2],
        'ACTUAL': result_list[3],
        'PREACTUAL': result_list[4],
        'YD15': result_list[5],
        'PREYD15': result_list[6]
    })

    path = 'userdata/%s/predict.csv' % session.get('username')
    df.to_csv(path, index=False)
    # 关闭连接
    cursor.close()
    connection.close()
    if result:
        return True
    return False


def query_range(turbid, year, month, day, hour, length):
    connection = get_connection()
    current_date = datetime(int(year), int(month), int(day), int(hour), 0, 0)
    following_date = current_date + timedelta(hours=int(length))
    current_date = current_date.strftime("%y-%m-%d %H:%M")
    following_date = following_date.strftime("%y-%m-%d %H:%M")
    cursor = connection.cursor()
    # 使用 SQL 查询语句从数据库中获取满足条件的数据

    cursor.execute(sql, (turbid, current_date, following_date))
    # 获取查询结果
    result = cursor.fetchall()
    result_list = [[], [], [], [], [], [], []]
    for item in result:
        for i in range(7):
            result_list[i].append(item[i])
    df = pd.DataFrame({
        'DATATIME': result_list[0],
        'WINDSPEED': result_list[1],
        'WINDSPEED2': result_list[2],
        'ACTUAL': result_list[3],
        'PREACTUAL': result_list[4],
        'YD15': result_list[5],
        'PREYD15': result_list[6]
    })

    path = 'userdata/%s/predict.csv' % session.get('username')
    df.to_csv(path, index=False)
    # 关闭连接
    cursor.close()
    connection.close()
    if result:
        return True
    return False


# 验证用户名密码
def sqlverifypassword(password):
    username = session.get('username')
    connection = get_connection()
    cursor = connection.cursor()
    sql = "SELECT * FROM usertable WHERE username=%s AND password=%s"
    cursor.execute(sql, (username, password))
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    if result:
        return True
    else:
        return False


# 修改用户表中用户密码
def sqlchangepassword(password):
    username = session['username']
    connection = get_connection()
    cursor = connection.cursor()
    sql = "UPDATE usertable SET password='%s' WHERE username='%s';" % (password, username)
    cursor.execute(sql)
    flg = cursor.rowcount
    print(flg)
    connection.commit()
    cursor.close()
    connection.close()
    if flg == 1:
        return True
    else:
        return False


# 查询风向数据
def query_winddirection_data(turbid):
    connection = get_connection()
    cursor = connection.cursor()
    sql = "SELECT * FROM winddirection WHERE TurbID=%s"
    cursor.execute(sql, turbid)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result


# 查询某用户的功能分布（功能调用次数分布）
def query_apicount_data(username, api):
    if username == 'admin':
        sql = "SELECT COUNT(*) FROM log WHERE api='%s'" % api
    else:
        sql = "SELECT COUNT(*) FROM log WHERE username='%s' and api='%s'" % (username, api)
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result


# 查询某个用户某天内调用功能分布（使用时段分析）
def query_timeapicount_data(username, year, month, day):
    if month < 10:
        month = '0' + str(month)
    else:
        month = str(month)
    if day < 10:
        day = '0' + str(day)
    else:
        day = str(day)
    date = str(year) + '-' + month + '-' + day + '%'
    if username == 'admin':
        sql = "SELECT api,COUNT(*) AS count FROM log WHERE operate_time LIKE '%s' GROUP BY api" % date
    else:
        sql = "SELECT api,COUNT(*) AS count FROM log WHERE username='%s' and operate_time LIKE '%s' GROUP BY api" % (
            username, date)
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    apimap = {int(item[0]): item[1] for item in result}
    countlist = []
    for i in range(7):
        if i in apimap.keys():
            countlist.append(apimap.get(i))
        else:
            countlist.append(0)
    return countlist


# 获取日志（日志操作记录）
def query_apilist_data(username):
    if username == 'admin':
        sql = "SELECT username,operate_time,api,note FROM log ORDER BY operate_time desc "
    else:
        sql = "SELECT username,operate_time,api,note FROM log WHERE username='%s' ORDER BY operate_time desc " % username
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result


# 新增用户（注册）
def addUser(username, password):
    connection = get_connection()
    cursor = connection.cursor()
    sql = 'INSERT IGNORE INTO usertable (username,password) VALUES (%s,%s)'
    cursor.execute(sql, (username, password))
    connection.commit()
    flg = cursor.rowcount
    cursor.close()
    connection.close()
    if flg == 1:
        return True
    return False


# 验证用户名密码
def verify_user(username, password):
    print('->', username, password)
    if username == '':
        return False
    connection = get_connection()

    # 创建游标对象
    cursor = connection.cursor()
    # 执行查询语句
    sql = "SELECT password FROM usertable WHERE username = '%s'" % username
    print(sql)
    cursor.execute(sql)
    # 获取查询结果
    result = cursor.fetchone()
    # 关闭游标和连接
    cursor.close()
    connection.close()
    # 根据查询结果返回验证结果
    print("测试0801", result)
    print('login->', result)
    if result[0] == password:
        return True
    else:
        return False


# 添加日志
def addlog(username, operate_time, api, note=''):
    connection = get_connection()
    cursor = connection.cursor()
    sql = "INSERT INTO log (username, operate_time, api, note) VALUES (%s,%s,%s,%s);"
    cursor.execute(sql, (username, operate_time, api, note))
    connection.commit()
    flg = cursor.rowcount
    print(flg)
    cursor.close()
    connection.close()
    if flg == 1:
        return True
    return False


# 更新当前用户的sdk
def updatesdk():
    username = session.get('username')
    connection = get_connection()
    cursor = connection.cursor()
    sql = "SELECT sdk FROM usertable WHERE username='%s'" % username
    cursor.execute(sql)
    result = cursor.fetchone()

    if result is not None:
        if result[0] is not None:
            session['sdk'] = result[0]


def updatesdktime():
    username = session.get('username')
    connection = get_connection()
    cursor = connection.cursor()
    sql = "SELECT time FROM usertable WHERE username='%s'" % username
    cursor.execute(sql)
    result = cursor.fetchone()

    if result is not None:
        if result[0] is not None:
            session['sdktime'] = result[0].strftime("%Y年%m月%d日 %H:%M:%S")

    cursor.close()
    connection.close()
    return session.get('sdk')


# _________________________________________________________________________注册登录退出_______________________________________________________________________

pathNoneCheckList = ['/index', '/visual', '/predict', '/offline', '/api', '/admin', '/log_admin', '/personalcenter',
                     '/log']


# 路由安全性过滤与检查
# @app.before_request
# def check_login():
#     path = request.path
#     username = session.get('username')
#     if path in pathNoneCheckList and username is None:
#         print('测试', path, username)
#         return redirect('/login')
#     return


# 注册界面
@app.route('/register')
def to_register():
    return render_template('register.html')


# 提交注册信息并进行注册
@app.route('/register_submit', methods=['POST'])
def register_submit():
    data = request.get_json()
    data = json.dumps(data)
    print(data)
    json_data = json.loads(data)

    username = json_data['username']
    password = json_data['password']
    repassword = json_data['repassword']

    if repassword != password:
        return '两次密码不一致！'
    else:
        if addUser(username, password):
            return '注册成功,点此登录'
        else:
            return '注册失败,请重试！'


# 登录界面
@app.route('/login')
def login():
    return render_template("login.html")


# 退出界面
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


# 登录验证
@app.route('/index', methods=['POST'])
def login_verify():
    username = request.form['username']
    password = request.form['password']
    flg = verify_user(username, password)
    if flg:
        session['username'] = username
        updatesdk()
        updatesdktime()

        # 为该用户建立需要的文件夹
        createfolder(username)

        # 为该用户创建ai对话账户
        print('我在创建ai用户')
        url1 = 'http://' + get_host_ip() + ':5446/createUser?username=' + username
        requests.get(url1)

        if username == 'admin':
            return redirect('/admin')
        else:
            return render_template("index.html", username=username, sdk=session.get('sdk'))
    elif password == '':
        error = '密码不能为空'
        # redirect('/login')
        return render_template('login.html', error=error, username=username)
    else:
        print('error')
        error = '用户名或密码错误'
        # redirect('/login')
        return render_template('login.html', error=error, username=username)


# _________________________________________________________________________主页_________________________________________________________________________

# 跳转到主页，如果未登录跳转登录界面
@app.route('/')
def home():
    return redirect('/login')


# 获取一定时间范围内的数据作为训练数据（主图左）
@app.route('/predict_value', methods=['GET'])
def predict_value():
    # 获取前端传递的查询参数
    turbid = request.args.get('turbid')
    year = request.args.get('year')
    month = request.args.get('month')
    day = request.args.get('day')
    hour = request.args.get('hour')
    length = request.args.get('length')

    # 查询数据库中满足条件的数据
    data = query_pre_data(turbid, year, month, day, hour, length)
    print(data[0])
    # 将查询结果处理为 JSON 格式
    result = jsonify({
        "DATETIME": data[0],
        "ROUND(A.POWER,0)": data[1],
        "YD15": data[2],
    })
    return result


@app.route('/train_predict', methods=['GET'])
def train_predict():
    # 获取前端传递的查询参数
    turbid = request.args.get('turbid')
    year = request.args.get('year')
    month = request.args.get('month')
    day = request.args.get('day')
    hour = request.args.get('hour')
    length = request.args.get('length')

    flg = query_preinput_data(turbid, year, month, day, hour, length)

    if flg:
        path = "userdata/%s/" % session.get('username')
        path1 = path + 'train.csv'
        path2 = path + 'predict.csv'
        res_list = train(path1, path2)
        print('__', res_list)
        return jsonify({
            "DATATIME": res_list[0],
            "PREYD15": res_list[1],
            "PREACTUAL": res_list[2],
            'YD15': res_list[3],
            "ACTUAL": res_list[4]
        })
    else:
        return jsonify({'error': 'error'})


@app.route('/rangeAccess', methods=['GET'])
def range_access():
    # 获取前端传递的查询参数
    turbid = request.args.get('turbid')
    year = request.args.get('year')
    month = request.args.get('month')
    day = request.args.get('day')
    hour = request.args.get('hour')
    length = request.args.get('length')

    flg = query_preinput_data(turbid, year, month, day, hour, length)

    if flg:
        path = "userdata/%s/" % session.get('username')
        path1 = path + 'train.csv'
        path2 = path + 'predict.csv'
        res_list = train(path1, path2)
        print('__', res_list)
        return jsonify({
            "DATATIME": res_list[0],
            "PREYD15": res_list[1],
            "PREACTUAL": res_list[2],
            'YD15': res_list[3],
            "ACTUAL": res_list[4]
        })
    else:
        return jsonify({'error': 'error'})


@app.route('/train_predict2', methods=['GET'])
def train_predict2():
    # 获取前端传递的查询参数
    turbid = request.args.get('turbid')
    year = request.args.get('year')
    month = request.args.get('month')
    day = request.args.get('day')
    hour = request.args.get('hour')
    length = request.args.get('length')

    flg = query_preinput_data(turbid, year, month, day, hour, length)

    if flg:
        path = "userdata/%s/" % session.get('username')
        path1 = path + 'train.csv'
        path2 = path + 'predict.csv'
        res_list = train2(path1, path2)
        print('__', res_list)
        return jsonify({
            "DATATIME": res_list[0],
            'YD15': res_list[1],
            "ACTUAL": res_list[2]
        })
    else:
        return jsonify({'error': 'error'})


# 获取风向数据（风向玫瑰图）
@app.route('/get_winddirection', methods=['GET'])
def get_winddirection():
    # 获取前端传递的查询参数
    turbid = request.args.get('turbid')
    data = query_winddirection_data(turbid)
    res_list_direction = list(data[0])
    # 将查询结果处理为 JSON 格式
    result = jsonify({
        "direction": res_list_direction
    })
    # print(res_list_direction)
    return result


# 跳转主页
@app.route('/index')
def to_index():
    username = session.get('username')
    return render_template('index.html', username=username)


# _________________________________________________________________________可视化大屏_____________________________________________________________________

# 跳转可视化大屏
@app.route('/visual')
def visual():
    username = session.get('username')
    return render_template('visual.html', username=username)


# 根据id读取供发电功率及预测功率
@app.route('/getiddata', methods=['GET'])
def getiddata():
    id = request.args.get('id')
    list = queryiddata(id)
    print('test', list[0][0])
    return jsonify({
        'DATATIME': list[0],
        'ACTUAL': list[1],
        'PREACTUAL': list[2],
        'YD15': list[3],
        'PREYD15': list[4]
    })


# 根据id和日期读取供发电功率及预测功率
@app.route('/queryonedatabyidandtime', methods=['GET'])
def getonedatabyidandtime():
    id = request.args.get('id')
    year = request.args.get('year')
    month = request.args.get('month')
    day = request.args.get('day')
    hour = request.args.get('hour')
    minute = request.args.get('minute')
    list = queryonedatabyidandtime(id, year, month, day, hour, minute)
    # print(list)
    return jsonify({
        'DATATIME': list[0][0],
        'ACTUAL': list[1][0],
        'PREACTUAL': list[2][0],
        'YD15': list[3][0],
        'PREYD15': list[4][0]
    })


# 根据id获取不同月份的供电量
@app.route('/querypowersupply', methods=['GET'])
def getpowersupply():
    id = request.args.get('id')
    print('__________>>', id)
    list = querypowersupply(id)
    print(list)
    return jsonify({
        'month': list[0],
        'values': list[1],
    })


# _________________________________________________________________________在线预测_______________________________________________________________________

# 跳转在线预测界面
@app.route('/predict')
def to_predict():
    username = session.get('username')
    return render_template('predict.html', username=username)


# 上传文件
@app.route('/upload_file', methods=['POST'])
def get_file():
    username = session.get('username')
    if 'file' not in request.files:
        return '未选择文件', 400
    file = request.files['file']
    if file.filename == '':
        return '未选择文件', 400
    now = datetime.now()
    operate_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # 添加日志
    addlog(username=username, operate_time=operate_time, api=api_list['upload_file'], note="上传数据文件")
    df = pd.read_csv(file)
    path = "userdata/%s/上传数据集" % username
    cnt = count_files_in_folder(path)
    filename = '/in' + str(cnt + 1) + '.csv'
    df.to_csv(path + filename, index=False)
    path = 'userdata/%s/当前上传数据集/tmp.csv' % username
    df.to_csv(path, index=False)

    return jsonify({})


# 数据分析
@app.route('/data_analyze', methods=['GET'])
def data_analysis():
    select = request.args.get('select')
    now = datetime.now()
    operate_time = now.strftime("%Y-%m-%d %H:%M:%S")
    csvid = '1'
    tmp_path = 'userdata/%s/当前上传数据集/tmp.csv' % session.get('username')
    df = pd.read_csv(tmp_path)
    turbid = df['TurbID'].iloc[0]
    if turbid == 1:
        csvid = '1'
    else:
        csvid = '2'
    # 添加日志
    addlog(username=session['username'], operate_time=operate_time, api=api_list['data_analyze'], note="数据分析处理")
    print(csvid + '_report' + select + '.html')
    return render_template('/report/' + csvid + '_report' + select + '.html')
    # return render_template('1_report0000.html')


# 对上传的文件进行预测并返回（功率预测）
@app.route('/online_predict')
def file_predict():
    now = datetime.now()
    operate_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # 添加日志
    addlog(username=session['username'], operate_time=operate_time, api=api_list['online_predict'], note="数据训练预测")
    path = 'userdata/%s/当前上传数据集/tmp.csv' % session.get('username')
    df = pd.read_csv(path)
    tmp = upload_predict(df)
    return tmp


# chatgpt分析
@app.route('/gptapi_analyze')
def analyze_wind_power():
    now = datetime.now()
    operate_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # 添加日志
    addlog(username=session['username'], operate_time=operate_time, api=api_list['gptapi_analyze'], note="AI分析")
    # 编辑prompt
    # openai.api_key = GPT_API
    # openai.api_base = "https://chat-api.leyoubaloy.xyz/v1"
    # send a ChatCompletion request to GPT
    path = 'userdata/%s/当前结果文件/tmp.csv' % session.get('username')

    try:
        df = pd.read_csv(path)
        # 在这里处理文件读取成功后的逻辑
        cnt = session.get('null_count')
        if cnt is None:
            cnt = 50
        else:
            cnt = int(cnt)
        res_datatime = df[-cnt:]['DATATIME'].tolist()
        res_windspeed = df[-cnt:]['WINDSPEED'].tolist()
        res_power = df[-cnt:]['YD15'].tolist()

        return jsonify({
            'datatime': to_string(res_datatime, 0),
            'windspeed': to_string(res_windspeed, 1),
            'power': to_string(res_power, 1)
        })
    except FileNotFoundError:
        # 文件不存在的处理逻辑
        return jsonify({
            'error': 'error'
        })

    # messages = [
    #     {"role": "system",
    #      "content": "我希望你扮演一个数据分析师的角色。作为数据分析师，你有深厚的数学和统计知识，并且擅长使用各种数据分析工具和编" +
    #                 "程语言来解析数据。你对风电数据非常熟悉，包括功率、风速与时间的关系。你的职责是分析这些数据，并提供关于可能原因和" +
    #                 "潜在风险的解释。作为数据分析师，你会仔细研究风电数据中功率、风速和时间之间的关系。你会运用统计方法分析数据的趋势和" +
    #                 "模式，以确定功率和风速之间的关联程度。你会考虑不同的时间段和季节对风电发电量的影响，并尝试找出任何异常或异常行为。在" +
    #                 "分析风电数据时，你会注意到一些可能的原因和潜在的风险。作为数据分析师，你的职责还包括向相关团队和管理层提供分析结果和建议。"},
    #     {"role": "user",
    #      "content": "这是一列时间序列，" + to_string(res_datatime, 0) + "这是对应的风速列，" + to_string(res_windspeed,
    #                                                                                                    1) + "这是对应的功率列，" + to_string(
    #          res_power, 1) +
    #                 "请结合时间分析一下风速对于功率的影响。"
    #                 "我需要你结合风速的变化，分析功率的变化情况，给出分析结果，比如某个时间到另一个时间内，风速发生了什么变化，功率又有什么变化，并分析原因，分析的透彻到底，一段话直接说明白，不用画图"},
    # ]
    # response = openai.ChatCompletion.create(
    #     model='gpt-3.5-turbo-0301',
    #     messages=messages,
    #     temperature=0,
    # )
    # print(messages)
    # 获取助手角色的回答
    # assistant_response = response['choices'][0]['message']['content']


# 下载结果文件
@app.route('/download_resfile')
def download_resfile():
    now = datetime.now()
    operate_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # 添加日志
    addlog(username=session['username'], operate_time=operate_time, api=api_list['download_resfile'],
           note="下载预测结果")
    path = "userdata/%s/下载结果文件/" % session.get('username')
    cnt = count_files_in_folder(path)
    file_name = 'res' + str(cnt + 1) + '.csv'
    file_path = path + file_name  # 文件在服务器上的路径

    tmp_path = 'userdata/%s/当前结果文件/tmp.csv' % session.get('username')
    df = pd.read_csv(tmp_path)
    df.to_csv(file_path, index=False)

    return send_file(file_path, download_name=file_name, as_attachment=True)


@app.route('/dialog')
def dialog():
    username = session.get('username')
    print('测试dialog获取ip', 'http://' + get_host_ip() + ":5446/")
    return render_template('dialog.html', username=username, baseURL='http://' + get_host_ip() + ":5446/")


###
# _________________________________________________________________________离线应用_________________________________________________________________________
###
#

###
# _________________________________________________________________________离线应用_________________________________________________________________________

# 跳转离线应用界面
@app.route('/offline')
def offline():
    username = session.get('username')
    # session.get('sdk')
    sdk = session.get('sdk')
    # print('sdk是多少:', sdk)
    return render_template("offline.html", username=username, sdk=sdk)


# 下载离线应用安装包
@app.route('/download_offine_soft')
def download_offine_soft():
    file_path = './offline_soft/风电功率预测系统油专特供版offline.msi'  # 文件在服务器上的路径
    return send_file(file_path, as_attachment=True)


# 将sdk和username对应起来加到数据库中（在离线应用界面申请sdk）
@app.route('/newsdk_offline')
def newsdkoffline():
    sdk = secrets.token_hex(16).__str__()
    username = session.get('username')

    connection = get_connection()
    cursor = connection.cursor()
    time = datetime.now() + timedelta(days=31)
    time = time.strftime("%Y-%m-%d %H:%M:%S")
    sql = "UPDATE usertable SET sdk='%s',time='%s' WHERE username='%s';" % (sdk, time, username)
    cursor.execute(sql)
    flg = cursor.rowcount
    print(flg)
    connection.commit()
    connection.close()
    cursor.close()
    if flg == 0:
        sdk = None
    print('申请sdk测试', sdk)
    session['sdk'] = sdk
    session['sdktime'] = time
    # jsonify({'sdk': sdk})
    return redirect('/offline')


# _________________________________________________________________________API使用文档_______________________________________________________________________

# 跳转API使用文档界面
@app.route('/api')
def to_api():
    username = session.get('username')
    sdk = session.get('sdk')
    return render_template('api.html', username=username, sdk=sdk)


# 将sdk和username对应起来加到数据库中（在api文档界面申请sdk）
@app.route('/newsdk_api')
def newsdkapi():
    sdk = secrets.token_hex(16).__str__()
    username = session.get('username')
    connection = get_connection()
    cursor = connection.cursor()
    time = datetime.now() + timedelta(days=31)
    time = time.strftime("%Y-%m-%d %H:%M:%S")
    sql = "UPDATE usertable SET sdk='%s',time='%s' WHERE username='%s';" % (sdk, time, username)
    print(sql)
    cursor.execute(sql)
    flg = cursor.rowcount
    print(flg)
    connection.commit()
    connection.close()
    cursor.close()
    if flg == 0:
        sdk = None
    print('申请sdk测试', sdk)
    session['sdk'] = sdk
    session['sdktime'] = time

    # jsonify({'sdk': sdk})
    return redirect('/api')


# _________________________________________________________________________windpower风电API_______________________________________________________________________

@app.route('/windpowerapisavepredictfile', methods=['POST'])
def api_predict():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    # 检查身份验证头部中是否包含正确的密钥
    if 'Authorization' not in request.headers or query_sdk_username(request.headers['Authorization']) == 'false':
        return jsonify({'error': '您的密钥未被授权，请更换密钥'})
    else:
        addlog(username=query_sdk_username(request.headers['Authorization']),
               operate_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), api=api_list['api'], note='API调用')

    file = request.files['file']
    # 检查文件类型是否为CSV
    if file.filename.endswith('.csv'):
        show = request.headers['figure']
        train = request.headers['train']
        output = request.headers['file']

        # 保存上传的文件
        file.save(file.filename)
        # 读取CSV文件
        df = pd.read_csv(file.filename)
        # 在此处进行预测操作，可根据自己的需求使用适当的模型或算法
        res_file = pd.DataFrame({})
        if train == 'True':
            null_count = df['YD15'].isnull().sum()
            cnt = null_count
            data_new = df.copy()
            df = df.fillna(0)
            # 新建一列
            df['WINDSPEED2'] = df['WINDSPEED'] * np.cos(np.radians(df['WINDDIRECTION'].values))
            # 预测YD15
            X_train1 = df[["WINDSPEED", "WINDSPEED2"]][:-null_count]
            y_train1 = df[["YD15"]][:-null_count]
            X_test1 = df[["WINDSPEED", "WINDSPEED2"]][-null_count:]
            # gbm
            x_train, x_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.2)
            gbm1 = LGBMRegressor(objective="regression", learning_rate=0.005, n_estimators=1000, n_jobs=-1)
            gbm1 = gbm1.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric="rmse",
                            callbacks=[early_stopping(stopping_rounds=1000)])
            y_pred15 = gbm1.predict(X_test1.values)
            output1 = y_pred15
            # 预测POWER
            X_train2 = df[["WINDSPEED", "WINDSPEED2"]][:-null_count]
            y_train2 = df[["ROUND(A.POWER,0)"]][:-null_count]
            X_test2 = df[["WINDSPEED", "WINDSPEED2"]][-null_count:]
            # gbm
            x_train, x_test, y_train, y_test = train_test_split(X_train2, y_train2, test_size=0.2)
            gbm2 = LGBMRegressor(objective="regression", learning_rate=0.005, n_estimators=1000, n_jobs=-1)
            gbm2 = gbm2.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric="rmse",
                            callbacks=[early_stopping(stopping_rounds=1000)])
            POWER = gbm2.predict(X_test2.values)
            output2 = POWER
            data_new['YD15'][-null_count:] = output1
            data_new['ROUND(A.POWER,0)'][-null_count:] = output2
            res_file = data_new
        else:
            model1 = joblib.load("usingmodels/model1.pkl")
            model2 = joblib.load("usingmodels/model2.pkl")
            null_count = df['YD15'].isnull().sum()
            cnt = null_count
            data_new = df.copy()
            # 新建一列
            df['WINDSPEED2'] = df['WINDSPEED'] * np.cos(np.radians(df['WINDDIRECTION'].values))
            train = df[["WINDSPEED", "WINDSPEED2"]]
            output1 = model1.predict(train.values)
            output2 = model2.predict(train.values)
            data_new['YD15'] = output1
            data_new['ROUND(A.POWER,0)'] = output2
            res_file = data_new
        # 假设预测结果为新的DataFrame对象
        predictions = res_file
        if show == 'True':
            plt.clf()
            df = res_file
            # 在图形上绘制示例图形
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
            plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
            x = df['DATATIME'][-cnt:].tolist()
            y1 = df['YD15'][-cnt:].tolist()
            y2 = df['ROUND(A.POWER,0)'][-cnt:].tolist()
            # mpl_style(dark=False)
            with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
                fig = plt.figure(figsize=(10, 4))
                plt.plot(x, y1, label='YD15')
                plt.plot(x, y2, label='ROUND(A.POWER,0)')
                plt.xlabel('日期')
                plt.ylabel('功率')
                plt.title('预测结果')
                plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
                plt.legend()
                fig.savefig('res_picture.png')

        # 生成预测后的CSV文件
        if output == 'True':
            output_filename = 'predictions.csv'
            predictions.to_csv(output_filename, index=False)

        if output == 'True' and show == 'False':
            return send_file('predictions.csv', as_attachment=True)
        elif output == 'False' and show == 'True':
            return send_file('res_picture.png', as_attachment=True)
        else:
            # 打包文件和图片
            files_to_compress = ['predictions.csv', 'res_picture.png']
            # 压缩后的ZIP文件路径
            zip_file_path = 'res.zip'
            # 创建一个ZIP文件并将文件添加到其中
            with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                for file in files_to_compress:
                    zipf.write(file)
            return send_file(zip_file_path, as_attachment=True)


@app.route('/windpowerapiprocessingdata', methods=['POST'])
def api_processingdata():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    # 检查身份验证头部中是否包含正确的密钥
    if 'Authorization' not in request.headers or query_sdk_username(request.headers['Authorization']) == 'false':
        return jsonify({'error': '您的密钥未被授权，请更换密钥'})
    else:
        addlog(username=query_sdk_username(request.headers['Authorization']),
               operate_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), api=api_list['api'], note='API调用')

    file = request.files['file']
    # 检查文件类型是否为CSV
    if file.filename.endswith('.csv'):
        fill = request.headers['fill']
        outlier = request.headers['outlier']
        normal = request.headers['normal']
        resample = request.headers['resample']
        deduplication = request.headers['deduplication']

        # 保存上传的文件
        file.save(file.filename)
        # 读取CSV文件
        df = pd.read_csv(file.filename)

        if fill == 'True':
            df = KNN_FillNaN.getcsv(df)
        if outlier == 'True':
            df = IQR_OutlierCorrection.getcsv(df)
        if resample == 'True':
            df = Resample.getcsv(df)
        if deduplication == 'True':
            df = DataDeduplication.getcsv(df)
        if normal == 'True':
            df = Normal.getcsv(df)

        df.to_csv('processed_dataset.csv', index=False)
        return send_file('processed_dataset.csv', as_attachment=True)


@app.route('/windpowerapiaianalysis', methods=['POST'])
def api_aianalysis():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    # 检查身份验证头部中是否包含正确的密钥
    if 'Authorization' not in request.headers or query_sdk_username(request.headers['Authorization']) == 'false':
        return jsonify({'error': '您的密钥未被授权，请更换密钥'})
    else:
        addlog(username=query_sdk_username(request.headers['Authorization']),
               operate_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), api=api_list['api'], note='API调用')

    file = request.files['file']
    # 检查文件类型是否为CSV
    if file.filename.endswith('.csv'):
        ai = request.headers['ai']

        # 保存上传的文件
        file.save(file.filename)
        # 读取CSV文件
        df = pd.read_csv(file.filename)

        if ai == 'True':
            res_datatime = df[-172:]['DATATIME'].tolist()
            res_windspeed = df[-172:]['WINDSPEED'].tolist()
            res_power = df[-172:]['YD15'].tolist()

            # 编辑prompt
            openai.api_key = env.GPT_API
            openai.api_base = "https://chat-api.leyoubaloy.xyz/v1"
            # send a ChatCompletion request to GPT
            messages = [
                {"role": "system",
                 "content": "我希望你扮演一个数据分析师的角色。作为数据分析师，你有深厚的数学和统计知识，并且擅长使用各种数据分析工具和编" +
                            "程语言来解析数据。你对风电数据非常熟悉，包括功率、风速与时间的关系。你的职责是分析这些数据，并提供关于可能原因和" +
                            "潜在风险的解释。作为数据分析师，你会仔细研究风电数据中功率、风速和时间之间的关系。你会运用统计方法分析数据的趋势和" +
                            "模式，以确定功率和风速之间的关联程度。你会考虑不同的时间段和季节对风电发电量的影响，并尝试找出任何异常或异常行为。在" +
                            "分析风电数据时，你会注意到一些可能的原因和潜在的风险。作为数据分析师，你的职责还包括向相关团队和管理层提供分析结果和建议。"},
                {"role": "user",
                 "content": "这是一列时间序列，" + to_string(res_datatime, 0) + "这是对应的风速列，" + to_string(
                     res_windspeed,
                     1) + "这是对应的功率列，" + to_string(
                     res_power, 1) + "请结合时间分析一下风速对于功率的影响。"
                                     "我需要你结合风速的变化，分析功率的变化情况，给出分析结果，比如某个时间到另一个时间内，风速发生了什么变化，功率又有什么变化，并分析原因，分析的透彻到底，大约200个汉字，一段话直接说明白，不用画图"},
            ]
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-0301',
                messages=messages,
                temperature=0,
            )
            print(messages)
            # 获取助手角色的回答
            assistant_response = response['choices'][0]['message']['content']
        return jsonify({
            'ans': assistant_response
        })


# _________________________________________________________________________管理员管理模型界面_______________________________________________________________________

# 跳转模型管理
@app.route('/admin')
def to_admin():
    username = session.get('username')
    if username == 'admin':
        return render_template('admin.html', username=username)
    else:
        return render_template('iderror.html')


# 获取模型列表
@app.route('/get_modelname')
def get_getmodels():
    usingmodels_list, tmp = get_file_paths('usingmodels')
    getmodels_list, tmp = get_file_paths('getmodels')
    return jsonify({
        'usingmodels': usingmodels_list,
        'getmodels': getmodels_list
    })


# 把模型从左移到右
@app.route('/add_models_to_pool', methods=['POST'])
def addmodels():
    filenames = request.get_json()
    path_root = 'getmodels/'
    path_pool = 'usingmodels/'
    for i in range(len(filenames)):
        print(path_root + filenames[i])
        print(path_pool + filenames[i])
        shutil.move(path_root + filenames[i], path_pool + filenames[i])
    return 'ok'


# 直接删除模型
@app.route('/delete_models', methods=['POST'])
def deletemodels():
    filenames = request.get_json()
    path_root = 'getmodels/'
    for i in range(len(filenames)):
        os.remove(path_root + filenames[i])
    return 'ok'


# 模型从右移到左
@app.route('/remove_models_from_pool', methods=['POST'])
def removemodels():
    filenames = request.get_json()
    path_root = 'getmodels/'
    path_pool = 'usingmodels/'
    for i in range(len(filenames)):
        print(path_root + filenames[i])
        print(path_pool + filenames[i])
        shutil.move(path_pool + filenames[i], path_root + filenames[i])
    return 'ok'


# _________________________________________________________________________日志____________________________________________________________________________

# 用户日志界面跳转
@app.route('/log')
def log():
    username = session.get('username')
    return render_template('log.html', username=username, log=log)


# 管理员日志界面跳转
@app.route('/log_admin')
def adminlog():
    username = session.get('username')
    return render_template('log_admin.html', username=username, log=log)


# 获取日志列表
@app.route('/get_log')
def get_loglist():
    username = session.get('username')
    reslog = query_apilist_data(username)
    res = []
    for item in reslog:
        res.append(list(item))
    return jsonify({
        'cnt': len(res),
        'log': res
    })


# 向前端返回（功能调用次数分布）数据
@app.route('/get_apicount', methods=['GET'])
def get_apicount():
    username = request.args.get('username')
    # data = query_apicount_data(username,)
    # res_list_apicount = list(data[0])
    res_list_apicount = []
    apilist = ['上传数据', '数据分析', '预测功率', 'AI分析', '下载结果', '上传模型', 'API调用']
    for key in api_list.keys():
        data = query_apicount_data(username, api_list[key])
        res_list_apicount.append(data[0][0])
    result = jsonify({
        "cnt": len(api_list),
        "apicount": res_list_apicount,
        "apilist": apilist
    })
    return result


# 向前端返回（使用时段分析）数据
@app.route('/get_timeapicount')
def get_timeapicount():
    username = request.args.get('username')
    daydata = []
    for i in range(1, 32):
        api_list = query_timeapicount_data(username, 2023, 8, i)
        daydata.append(api_list)
    result = jsonify({
        "daydata": daydata
    })
    return result


# _________________________________________________________________________个人中心_________________________________________________________________________

# 跳转个人中心界面
@app.route('/personalcenter', methods=['GET'])
def to_personalcenter():
    username = session.get('username')
    return render_template('personalcenter.html', username=username, key_amount=session.get('sdktime'))


# 前往个人中心查看密钥查看sdk
@app.route('/check_sdk')
def check_sdk():
    username = session.get('username')
    sdk = session.get('sdk')
    return render_template('personalcenter.html', username=username, check=sdk, key_amount=session.get('sdktime'))


# 点击（查看）密钥
@app.route('/verifypassword', methods=['POST'])
def verifypassword():
    password = request.form.get('password')
    print('___________', password)
    if sqlverifypassword(password):  # 验证密码成功
        sdk = session.get('sdk')
        if sdk is None:
            print('您还未申请sdk!')
            return '您还未申请sdk!'
        else:
            print(sdk)
            return sdk
    else:
        print('密码错误!')
        return '密码错误!'


# 修改密码
@app.route('/changepassword', methods=['POST'])
def changepassword():
    password = request.form.get('password')
    if sqlchangepassword(password):
        return '修改成功'
    else:
        return '修改失败，请重试'


# 将续费时长添加
@app.route('/sdktimeadd', methods=['GET'])
def sdktimeadd():
    time = request.args.get('time')
    print('月份数', time)
    username = session.get('username')
    flg = addsdktimemonth(username, time)
    if flg:
        updatesdktime()
        return jsonify({
            "result": 'success'})
    else:
        return jsonify({
            "result": 'failed'})


# 更换头像
@app.route('/changetx', methods=['POST'])
def changetx():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    path = "static/usertouxiang/%s/touxiang.png" % session.get('username')
    file.save(path)  # 将图片保存到指定路径
    return "Image uploaded successfully"


# 获取用户文件列表
@app.route('/get_userfile')
def get_userfile():
    path1 = "userdata/%s/上传数据集" % session.get('username')
    path2 = "userdata/%s/下载结果文件" % session.get('username')
    uploadcsv_list, time1 = get_file_paths(path1)
    downloadcsv_list, time2 = get_file_paths(path2)
    return jsonify({
        'uploadcsv': uploadcsv_list,
        'uploadcsv_time': time1,
        'downloadcsv': downloadcsv_list,
        'downloadcsv_time': time2
    })


# （个人中心）界面下载历史文件
@app.route('/download_history_csv')
def download_history_csv():
    file_path = request.args.get('path')
    return send_file(file_path, as_attachment=True)


# _________________________________________________________________________离线app_________________________________________________________________________

# 获取他人使用offline程序跑出来的模型
@app.route('/getmodel', methods=['POST'])
def get_model():
    file = request.files['file']
    file.save('模型相似度检测/tmp.pkl')  # 保存到指定位置
    model = joblib.load('模型相似度检测/tmp.pkl')
    df = pd.read_csv('模型相似度检测/test.csv')
    output = model.predict(df[['WINDSPEED', 'WINDSPEED2']])
    # print(output)
    cnt = count_files_in_folder('./getmodels')
    if cnt % 2 == 0:
        tmp = int(cnt / 2)
        mae = getMAE(output, origin_model_resYD15)
        s = getSimilarity(mae)
        s = round(s * 100000, 0) + random.randrange(100, 300)
        s = int(s)

        file_name = "yd15_" + str(tmp + 1) + '_' + str(s) + ".pkl"
    else:
        tmp = int(cnt / 2)
        mae = getMAE(output, origin_model_resPOWER)
        s = getSimilarity(mae)
        s = round(s * 100000, 0) + random.randrange(100, 300)
        s = int(s)
        file_name = "actualpower_" + str(tmp + 1) + '_' + str(s) + ".pkl"

    # 源文件路径
    source_folder = '模型相似度检测'
    source_file = 'tmp.pkl'
    source_path = os.path.join(source_folder, source_file)

    # 目标文件夹路径
    target_folder = 'getmodels'
    target_file = file_name
    target_path = os.path.join(target_folder, target_file)

    # 剪切文件并修改名称
    shutil.copyfile(source_path, target_path)
    return '文件上传成功！'


router_list = [
    '/',
    '/index',
    ''
]
static_list = [
    '/staic',

]


# _________________________________________________________________________大屏_________________________________________________________________________

@app.route('/sum_by_turbid', methods=['GET'])
def sum_by_turbid():
    # connection = get_connection()  # 从连接池获取连接
    # cursor = connection.cursor()
    #
    # try:
    #     # 执行查询语句
    #     cursor.execute("SELECT turbid, SUM(yd15) AS sum_yd15 FROM datatmp GROUP BY TurbID")
    #     result = cursor.fetchall()
    #     resDict = {item[0]: item[1] for item in result}
    #     print('sum:', resDict)
    #
    #     # 返回 JSON 格式的结果
    #       return jsonify(resDict)
    # finally:
    #     cursor.close()
    #     # release(connection)  # 将连接释放回连接池
    return jsonify({
        "11": 875243253.858176,
        "12": 831182995.9716837,
        "13": 1137271137.9981165,
        "14": 621854739.6179744,
        "15": 1962467735.8125446,
        "16": 2256899187.198111,
        "17": 1101151908.2103438,
        "18": 198365430.37620512,
        "19": 1390901559.3073874,
        "20": 495908518.4166046
    })


# _________________________________________________________________________导航栏foot_________________________________________________________________________
@app.route('/navigation.html')
def navigation():
    username = session.get('username')
    return render_template('navigation.html', username=username)


@app.route('/footer.html')
def footer():
    username = session.get('username')
    return render_template('footer.html', username=username)


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
myhost = '0.0.0.0'
myport = 5446
report_port = 40000

if __name__ == '__main__':
    # print(render_template('footer.html').__str__())

    print("持久化存储文件路径为:", os.path.join(os.getcwd(), USER_DICT_FILE))
    all_user_dict = LRUCache(USER_SAVE_MAX)
    check_load_pickle()

    if len(API_KEY) == 0:
        # 退出程序
        print("请在openai官网注册账号，获取api_key填写至程序内或命令行参数中")
        exit()
    app.run(debug=True, host=myhost, port=myport)
