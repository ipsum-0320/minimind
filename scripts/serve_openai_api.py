# 启动一个 FastAPI 服务，提供 OpenAI 兼容的接口
# 支持模型的加载和推理（支持 LoRA 微调）
# 支持流式和非流式的聊天完成请求
# 提供命令行参数配置模型大小、层数、是否使用 MoE 等
# 默认运行在 http://0.0.0.0:8998

import argparse
import json
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

app = FastAPI()


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_ropex_scaling=args.inference_rope_scaling
            # rope 长度外推。
        ))
        # torch.load。torch.load(f): 将保存的文件重新读取为 Python 对象，它只是把数据加载到内存里，不会自动把权重塞进你的模型实例中。
        # torch.save。torch.save(obj, f): 将任意 Python 对象（通常是 dict 或 Tensor）保存到磁盘。
        # model.load_state_dict。他是一个“填充”过程。它将一个已经加载到内存中的字典，映射到当前 model 实例的各个参数中。
        # model.save_pretrained。Transformers 中的保存方法，model (PreTrainedModel) 或 tokenizer (PreTrainedTokenizer)。
        # model.from_pretrained。支持从本地路径或 Hugging Face Hub 自动下载并加载。
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        if args.lora_weight != 'None':
            # 应用 LoRA。
            apply_lora(model)
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(device), tokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = False
    tools: list = []


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)

# 流式输出。
def generate_stream_response(messages, temperature, top_p, max_tokens):
    try:
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            # model 继承自 PreTrainedModel, GenerationMixin
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

        Thread(target=_generate).start()

        while True:
            text = queue.get()
            if text is None:
                yield json.dumps({
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }, ensure_ascii=False)
                break

            yield json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )),
                media_type="text/event-stream"
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                # 这里的 messages 就是 chat_openai_api.py 里面 
                # client.chat.completions.create 的 messages。
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]
            # apply_chat_template 将原始的对话列表（包含 system, user, assistant 角色）转换成模型能理解的特定字符串格式
            # （如 Llama 3 的 <|begin_of_text|> 或 Qwen 的 <|im_start|>）。
            # add_generation_prompt 在末尾添加提示词，告诉模型“该轮到你说话了”。

            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            # 将格式化后的字符串转换成数字序列（Tokens），转为 PyTorch 张量，并移动到 GPU (device) 上。

            with torch.no_grad():
                # 因为 model 继承了 GenerationMixin 才能使用 generate 来生成，而不是自己手动去写自回归。
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    # max_length 参数定义的是 [输入 + 输出] 的总长度。
                    do_sample=True,
                    # do_sample 的作用是开启随机采样。
                    # 如果是 False，模型进入 贪心搜索 (Greedy Search) 模式。它每次只选概率最大的那个词。
                    # 如果是 True，模型进入 随机采样 模式。它会根据概率分布随机抽一个词。
                    # 配合你代码里的 temperature 和 top_p 使用。只有开启了 do_sample，这两个参数才会生效。
                    attention_mask=inputs["attention_mask"],
                    # 在批处理（Batch Processing）时，为了让不同长度的句子对齐，短句子后面会补一堆无意义的 [PAD] 字符。
                    # attention_mask 是一个由 0 和 1 组成的序列。
                    # 1：表示“真内容”，模型计算注意力时会看它。
                    # 0：表示“填充符”，模型计算时会完全无视它。
                    pad_token_id=tokenizer.pad_token_id,
                    # 规定“空白字符”，也就是 [PAD] 的具体编号。
                    eos_token_id=tokenizer.eos_token_id,
                    # 模型的“句号”。一旦模型输出了这个 ID，generate 就会提前退出循环。
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                # 只保留新生成的 Token，把输入的部分（Prompt）切掉。
                # 将数字序列转回人类可读的文字，同时过滤掉类似 <|endoftext|> 这种模型内部的特殊标记。
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Python 的寻找变量的顺序符合如下规则：
# 1. Local 局部作用域。
# 2. E (Enclosing) - 嵌套（闭包）作用域。
# 3. G (Global) - 全局作用域。
# 4. B (Built-in) - 内置作用域。

# 介绍一下 global，有时候你不仅想读外层的变量，还想改它。这时直接赋值会产生一个新的局部变量，除非你使用关键字 global。
# count = 0
# def increment():
#     global count  # 声明：我要用外面那个 count
#     count += 1
# increment()
# print(count) # 输出 1

# Python 一个容易踩坑的问题是，没有“块级作用域”。
# 在 Python 中，if、for、while 不会产生新的作用域。
# if True:
#     z = "I am global"
# print(z)  # 竟然可以打印出来！z 现在是全局变量

# 所以，这就解释了为什么在 if __name__ == "__main__" 作用域块中的 tokenizer 可以被函数内引用了。
# 因为 if __name__ == "__main__" 没有创建新的作用于，他只是全局作用于的一个分支，因此 tokenizer 实际上是一个全局作用域变量。
# 在很多其他语言（如 C++ 或 Java）中，main 函数里的变量通常是局部的。但在 Python 中，if __name__ == "__main__": 并不是一个函数，它只是一个条件判断语句。在条件语句块中定义的变量，其作用域依然属于该语句所在的层级。既然这个 if 块在文件的最外层，里面定义的变量就是全局变量。

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    device = args.device
    model, tokenizer = init_model(args)
    uvicorn.run(app, host="0.0.0.0", port=8998)
