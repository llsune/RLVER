import numpy as np
from openai import OpenAI
import os, socket, httpx, time, json, requests


url = 'https://antchat.alipay.com/v1/chat/completions'
url_api = "https://matrixllm.alipay.com/v1/chat/completions"
api_key = 'PRrd2S3p6U3lYeMIFX25Gnvgp240hcMY'
model_names = {
    'v0.9': 'qmd_chat_v0_9',
    'v0.9.1': 'qmd_chat_v0_9_1',
    'v0.9.2': 'qmd_chat_v0_9_2',
    'v0.9.2.b3': 'qmd_chat_v0_9_2_update',
    'v0.10.exp': 'qmd_chat_v0_10_exp',
    'v1.0': 'qmd_chat_v1_0_0109',
    'exp': 'qmd_chat_v0_10_exp_b3',
    'rag-exp': 'qmd_rag_exp',
    'rag-v0.9.2': 'qmd_rag_v0_9_2_d251202',
    'rag-v0.9.4': 'qmd_rag_v0_9_4_d1209',
    'rag-v0.9.5': 'qmd_rag_v0_9_5',
    'rag-v0.9.6': 'qmd_rag_v0_9_6_d1223',
    'mem0': 'qmd_mem0_service_update'
}
headers_api = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-90139bca10e14f6b9d8c77621ae3b352"
}

def think_split(text):
    think, response = text.replace('<think>', '').split('</think>')
    return think.strip(), response.strip()

def llm_deploy(query, system_prompt='You are a helpful assistant.', version='rag-exp', temperature=0.5, repetition_penalty=1.0, top_p=0.95, max_tokens=512):
    assert (isinstance(query, str) or isinstance(query, list))
    if isinstance(query, str):
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': query}]
    elif isinstance(query, list):
        messages = query

    client = OpenAI(api_key=api_key, base_url="https://antchat.alipay.com/v1")
    if version == 'mem0':
        completion = client.chat.completions.create(
            model=model_names.get(version),
            messages=messages,
            stream=False,
            max_tokens=4096,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    else:
        completion = client.chat.completions.create(
            model=model_names.get(version),
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": repetition_penalty})
    return completion.choices[0].message.content


# model in ['Qwen3-Next-80B-A3B-Instruct', 'DeepSeek-V3.1', 'Qwen3-235B-A22B-Instruct-2507', 'gemini-2.5-pro', 'gemini-3-pro-preview', 'gpt-5-2025-08-07', 'doubao-seed-1.6-250615']
def llm_call(query, model='Qwen3.5-397B-A17B', enable_thinking=False, temperature=0.7, top_p=0.8, max_tokens=32768, api_key=api_key, system_prompt='You are a helpful assistant.'):
    assert (isinstance(query, str) or isinstance(query, list))
    if isinstance(query, str):
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': query}]
    elif isinstance(query, list):
        messages = query

    if model in ['gemini-2.5-pro', 'gemini-3-pro-preview', 'gpt-5-2025-08-07', 'doubao-seed-1.6-250615']:
        data = {"stream": False, "model": model, "messages": messages}
        response = requests.post(url_api, headers=headers_api, json=data)
        result = response.json()
        return result['choices'][0]['message']['content']

    payload = json.dumps({
    "model": model,
    "messages": messages,
    "stream": False,
    "reasoning": {"enabled": enable_thinking},
    "chat_template_kwargs": {"enable_thinking": enable_thinking, "thinking": enable_thinking},
    "temperature": temperature,
    "top_p": top_p,
    "max_tokens": max_tokens,
    })
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        js_data = json.loads(response.text)
        answer = js_data['choices'][0]['message']['content']
        return answer
    except:
        print('ERROR!!! response is as follow:')
        print(response.text)


if __name__ == '__main__':
    query = '你是谁'
    messages = [{'role': 'user', 'content': query}]
    print(llm_call(messages, model='MiniMax-M2.5'))
    print(llm_deploy(messages))