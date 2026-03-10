# convert_model.py - 模型格式转换工具
# 用于在不同模型格式之间转换：
# PyTorch 格式 ↔ Transformers 格式
# MiniMind 原生格式 → Transformers Llama 格式
# 支持模型精度转换（如 float16）
# 自动保存 tokenizer 和配置文件
# 兼容 Transformers 5.0 的新写法

import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


# MoE模型需使用此函数转换
# 将 MiniMind 原生定义的模型架构转换为 transformers 格式。
# 代码使用了 register_for_auto_class()。这意味着转换后的模型文件夹会包含自定义的 Python 代码，
# 用户可以通过 AutoModel.from_pretrained(..., trust_remote_code=True) 直接加载，保留了 MiniMind 特有的优化（如 MoE 混合专家架构）。


# 原始的 .pth 文件只是模型参数的“快照”，没有结构定义。如果不转换成 transformers 格式，其他人就无法通过一行代码 from_pretrained 调用我们的模型。
# 当我们使用 torch.save(model.state_dict(), 'model.pth') 保存时，我们保存的其实是一个纯数字字典。
# 它只包含：layer1.weight: [0.12, -0.05, ...] 这种权重数值。
# 它不包含：模型有多少层？每一层多宽？是用 Llama 结构还是 MiniMind 结构？
# 后果：如果别人想用我们的 .pth，他必须先手写一段和我们一模一样的 Python 类代码（比如 class MiniMind...），然后才能把这些数字“填进”代码定义的结构里。如果代码丢了，这堆数字就是一堆乱码。


# transformers 格式则是整装待发的成品包装，当你运行代码中的 save_pretrained 时，它会生成一个文件夹，里面通常包含：
# model.safetensors：模型权重（更安全、加载更快版的 .pth）。
# config.json：最重要的“说明书”。它记录了 hidden_size: 512, num_heads: 8 等所有结构参数，甚至记录了该用哪段代码来跑这个模型。
# tokenizer.json：词表文件，告诉模型怎么把汉字/单词转成数字。

# 当我们调用 Transformers 的 AutoModelForCausalLM.from_pretrained 时，背后发生了如下内容：
# 1. 读取 config.json：噢，这是一个 model_type: llama 的模型。
# 2. 自动找“模具”：既然是 Llama，transformers 库会自动在自己的代码库里翻出 LlamaForCausalLM 的定义。
# 3. 灌入权重：把文件夹里的权重数字填到这个定义的“模具”里。
# 4. 就绪：模型直接可以跑了。

# 为什么会有 register_for_auto_class 这行代码？
# 由于 transformers 官方库里原本并没有 MiniMind 这个型号，如果你不注册，from_pretrained 就会报错说“我不认识这个模型”。
# 执行注册后，转换出的 config.json 会多出一行配置，指向你本地的 modeling_minimind.py。
# 结果就是别人即便电脑里没装 MiniMind，只要联网或者文件夹里有你的代码，from_pretrained 就能通过这个“注册信息”自动加载你的自定义结构。

def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    # 转换后的模型文件夹内会包含 modeling_minimind.py 等自定义代码文件。
    # 使用 MiniMindConfig 和 MiniMindForCausalLM。用户调用时必须设置 trust_remote_code=True。
    # 核心用途是支持 MoE (混合专家模型)。因为标准的 Llama 架构并不原生支持 MiniMind 的 MoE 实现，如果我们训练的是 MoE 版本，必须用这个函数。注意原生的 Llama 模型并不是 MoE 模型，而是 Dense 模型，Mixtral 才将 MoE 发扬光大。

    # 这里的 modeling_minimind.py 是什么？
    # 这里的 modeling_minimind.py 实质上就是你本地定义的模型源代码文件，他们就是 MiniMindConfig 和 MiniMindForCausalLM，只不过是被 copy 过去的。
    # 在你运行 save_pretrained 之前，你需要确保你的项目目录下已经有了定义 MiniMindForCausalLM 和 MiniMindConfig 的 Python 文件。
    # 当 python 执行如下代码时：
    # ======
    # MiniMindConfig.register_for_auto_class()
    # MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    # lm_model.save_pretrained(transformers_path)
    # ======
    # Hugging Face 的 transformers 库会做两件事：
    # 1. 在 config.json 中注入索引：它会在配置文件里记录下：“这个模型对应的配置类是 MiniMindConfig，模型类是 MiniMindForCausalLM”。
    # 2. 复制源码：它会寻找定义这些类的源代码文件，并把它们拷贝到输出的 transformers_path 文件夹下。

    # 最终生成的 modeling_xxx.py 文件会包含了启动某个模型的所有代码，即使源代码被分散到了不同的文件中。
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    # 兼容transformers-5.0的写法
    config_path = os.path.join(transformers_path, "tokenizer_config.json")
    json.dump({**json.load(open(config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")

# 在 Pytorch 中，什么时候是只存储参数，什么时候是参数架构一起存储？

# 只存储参数：torch.save(model.state_dict(), path)，这种一般是在模型训练中会执行，例如训练过程中每隔 1000 步存一次档，这是为了节省空间和保持灵活性，通常只存参数。这样的优点是 占用空间最小；非常灵活（你可以把 A 模型的参数加载到结构相似的 B 模型里）。缺点是必须配合完全一致的源代码。代码改了一个变量名或多了一层，参数就加载不进去了（报错 RuntimeError: Unexpected key(s) in state_dict）。

# 参数架构一起存储：torch.save(model, path) 注意没有 .state_dict()，利用 Python 的 pickle 序列化整个模型对象。它把代码和参数都打包成一个文件了。缺点是极度依赖 Python 环境和目录结构。如果你把文件移动到另一个文件夹，或者改了项目的目录名，加载时就会疯狂报错 ModuleNotFoundError。在工业界，这种存法基本是被禁止的。

# 如果使用 save_pretrained 也可以参数架构一起存储。它包括了 transformers 文件夹格式（包含 config.json + 权重）。
# 它不仅存了参数，还存了“构造图纸”。哪怕 transformers 库更新了，只要有 config.json 和 modeling.py，模型就能完美还原。

# 在工业界，在实验阶段用 .pth 存 state_dict，因为你会频繁改代码；在部署/交付阶段，必须转成 transformers 格式，确保在任何地方都能一键 from_pretrained。


# LlamaForCausalLM结构兼容第三方生态
# 将 MiniMind 的权重“套”进标准的 LlamaConfig 结构中。
# 因为 MiniMind 的设计参考了 Llama，所以通过计算 intermediate_size 等参数，可以将它伪装成一个标准的 Llama 模型。
# 转换后，该模型可以无缝兼容所有支持 Llama 的第三方工具（如 vLLM 加速推理、llama.cpp 量化、Ollama 等），无需额外安装自定义代码。
def convert_torch2transformers_llama(torch_path, transformers_path, dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_position_embeddings,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
        tie_word_embeddings=True
    )
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model = llama_model.to(dtype)  # 转换模型权重精度
    llama_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    # 兼容transformers-5.0的写法
    config_path = os.path.join(transformers_path, "tokenizer_config.json")
    json.dump({**json.load(open(config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-Llama 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式 (half精度): {torch_path}")


if __name__ == '__main__':
    lm_config = MiniMindConfig(hidden_size=512, num_hidden_layers=8, max_seq_len=8192, use_moe=False)
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../MiniMind2-Small'
    convert_torch2transformers_llama(torch_path, transformers_path)
    # # convert transformers to torch model
    # convert_transformers2torch(transformers_path, torch_path)
