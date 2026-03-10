# 一个简单的命令行聊天界面，通过 OpenAI 兼容的 API 与 MiniMind 模型进行交互
# 支持流式输出（stream=True），可以实时显示模型的回复
# 支持对话历史管理，可以配置是否携带之前的对话内容
# 连接到本地的 Ollama 服务（http://127.0.0.1:8998/v1）

from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:8998/v1"
)
stream = True
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
history_messages_num = 0  # 必须设置为偶数（Q+A），为0则不携带历史对话
while True:
    query = input('[Q]: ')
    conversation_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="minimind",
        messages=conversation_history[-(history_messages_num or 1):],
        stream=stream,
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9
    )
    if not stream:
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        print('[A]: ', end='')
        assistant_res = ''
        for chunk in response:
            print(chunk.choices[0].delta.content or "", end="")
            assistant_res += chunk.choices[0].delta.content or ""

    conversation_history.append({"role": "assistant", "content": assistant_res})
    print('\n\n')
