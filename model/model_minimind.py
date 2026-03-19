from transformers import PretrainedConfig
# 在深度学习中，一个模型由两部分组成：
# 权重（Weights）： 模型的“记忆”，即那些巨大的参数矩阵。
# 配置（Configuration）： 模型的“结构”，比如它有几层、每层有多宽、词汇表有多大等。

# 它的核心作用：
# - 定义结构： 它规定了模型的超参数（比如 hidden_size 隐藏层维度、num_attention_heads 注意力头数等）。
# - 加载与保存： 它可以从本地文件夹或 Hugging Face Hub 上读取 config.json 文件，也可以把当前的配置保存成 JSON。
# - 模型初始化的桥梁： 当我们创建一个模型时，系统会先看这个“说明书”，知道该盖多少层楼、挖多深的基座，然后再把“权重”填进去。

# PretrainedConfig 是所有模型配置类的“祖宗”，它不负责计算，只负责告诉程序这个模型长什么样。使用 PretrainedConfig 是为了更好的兼容性和标准化。

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    # 告诉 Hugging Face 库，这个配置对应的模型类型是 minimind
    # 这里的 model_type 是类变量，所有实例共享这个变量。

    def __init__(
            self,
            dropout: float = 0.0,
            # 随机失活率，防止过拟合。
            bos_token_id: int = 1,
            # 开始标记的 ID，用于标识句子的开始，详情去看 tokenizer.json。
            # <|im_start|> 就是标号为 1。
            eos_token_id: int = 2,
            # 结束标记的 ID，用于标识句子的结束，详情去看 tokenizer.json。
            # <|im_end|> 就是标号为 2。
            hidden_act: str = 'silu',
            # 隐藏层激活函数，常用的有 ReLU、SiLU 等。
            # 这代表模型中主要计算层（MLP/FFN层）的所有隐藏激活函数都统一使用 SiLU。
            hidden_size: int = 512,
            # 隐藏层的维度，即每层有多少个神经元。
            # 在原始的 Transformer 论文（Attention Is All You Need）中，这个参数的名字就叫 $d_{model}$。
            # 论文里为了写公式方便，定义了 $d_{model}$。在 Hugging Face 等代码库里，为了直观表达这是“隐藏层的大小”，
            # 统一采用了 hidden_size 这个变量名。它们指的完全是同一个东西即 Embedding（嵌入层）的维度，
            # 也是每一层 Transformer Block 输入和输出的向量维度。
            intermediate_size: int = None,
            # 中间层的维度，即前馈网络的隐藏层维度。
            # 在 Transformer 的每个 Block 中，数据经过 Self-Attention 后，
            # 会进入一个 FFN（前馈神经网络） 层。这个过程通常是“先扩容，再压缩”。
            # 输入维度是 $d_{model}$（我们的 512）。
            # 升维（中间层）维度扩充到 intermediate_size。
            # 降维重新压回到 $d_{model}$（512）。
            max_position_embeddings: int = 32768,
            # 最大位置嵌入长度，即输入序列的最大长度，或者说上下文的长度。
            num_attention_heads: int = 8,
            # 注意力头数，即多头注意力机制中的头数。
            num_hidden_layers: int = 8,
            # 隐藏层数，即模型有多少层。
            # num_hidden_layers 指的就是 Transformer Decoder Block 的堆叠层数。
            num_key_value_heads: int = 2,
            # 键值头数，即多头注意力机制中键和值的头数。
            # 这是 GQA 中的参数。
            vocab_size: int = 6400,
            # 词汇表大小，即模型能识别的单词数量。
            rms_norm_eps: float = 1e-05,
            # RMSNorm 的一个超参数，用于数值稳定性。
            rope_theta: int = 1000000.0,
            # 外推缩放参数，用于控制外推长度。
            inference_rope_scaling: bool = False,
            # 是否使用外推缩放。
            # 假设我们的模型是在 2048 长度下训练的，现在我们硬要它处理 32768 长度，直接跑肯定会“变傻”，因为它没见过那么大的位置编号。
            # False： 按照常规逻辑推理。
            # True： 启动 YaRN (Yet another RoPE extensioN) 等缩放算法。
            # 它通过数学手段，把 32k 的位置信息“挤”到模型原本熟悉的 2k 范围内。
            # 打个比方：本来一把尺子只有 20 厘米刻度，通过缩放，我们让模型认为这把尺子其实有 320 厘米，只不过每个刻度变密了。
            flash_attn: bool = True,
            # 是否使用 FlashAttention。
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            # 是否使用 MoE（Mixture of Experts）。
            num_experts_per_tok: int = 2,
            # 每个 token 选择的专家数量。
            # 对于每一个字（Token），模型会通过一个路由器（Router）从 4 个专家里选出 2 个 最合适的来干活。
            n_routed_experts: int = 4,
            # 总的专家数量。
            n_shared_experts: int = 1,
            # 共享专家
            # 这是一个比较先进的设计（类似 DeepSeek）。无论路由怎么选，这个共享专家总是参与计算。它负责捕捉所有 Token 共有的通用知识，而路由专家负责捕获特定领域的知识。
            scoring_func: str = 'softmax',
            # 评分函数，默认为'softmax'。
            # 路由器给 4 个专家打分，用 softmax 把分数归一化到 0-1 之间，然后选分最高的。
            aux_loss_alpha: float = 0.01,
            # 这是一个系数。如果专家之间任务分配不均，就会产生一个惩罚项（Loss）。这个值越大，模型就越努力地让每个专家都有活干。
            seq_aux: bool = True,
            # 意味着在整个序列（Sequence）的维度上去平衡专家的负载，而不是只看单个位置。
            norm_topk_prob: bool = True,
            # 假设选出的两个专家得分是 0.8 和 0.4，标准化后会让它们加起来等于 1（比如变成 0.67 和 0.33），从而保持信号强度的稳定。
            **kwargs
            # 其他参数
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            # 对应高频部分。对于相邻很近的词，YaRN 认为不应该过度拉伸，否则模型会分不清谁是谁。
            "beta_slow": 1,
            # 对应低频部分。对于跨度很大的长文本信息，YaRN 认为这部分应该多拉伸一些。
            "factor": 16,
            # 这是缩放因子。因为 $32768 / 2048 = 16$。它告诉模型：我们要把位置编码的“尺子”拉长 16 倍。
            "original_max_position_embeddings": 2048,
            # 这是模型训练时真实使用的最大长度基准。
            "attention_factor": 1.0,
            # 这是一个修正系数。当序列变长时，注意力分数（Attention Score）的熵会发生变化。
            # 这个参数用来平衡注意力机制的能量，防止因为序列变长导致注意力变得过于集中或分散。
            "type": "yarn" # 指定缩放算法的类型——yarn
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# 规范化层
class RMSNorm(torch.nn.Module):
    # 这段代码实现了 RMSNorm (Root Mean Square Layer Normalization)，这是一种在 Llama 等大语言模型中广泛使用的归一化技术。它的核心思想是：与其像 LayerNorm 那样进行“平移和缩放”，不如只进行“缩放”。
    # LayerNorm 执行了两步操作：平移（减去均值 $\mu$）和 缩放（除以标准差 $\sigma$）。
    # RMSNorm 只执行了一步操作：缩放（除以均方根）。它直接去掉了减去均值（Centering）的过程，也没有偏置项 $\beta$。

    # LayerNorm 的可学习参数有两个，而 RMSNorm 的可学习参数只有一个。
    # 因此前者计算开销较高（需计算均值和方差），后者计算开销较低（计算更简单，速度快）。
    def __init__(self, dim: int, eps: float = 1e-5):
        # 初始化函数。dim 是输入的维度（通常是隐藏层大小），eps 是一个极小的正数，防止除以零。
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        # 定义一个可学习的缩放参数 $\gamma$（Scale）。初始化全为 1。注意：RMSNorm 通常没有 Bias（偏移参数）。
        # nn.Parameter 初始化为 1 是为了在训练开始时，让 RMSNorm 处于“透明状态”，即不改变经过归一化后的信号强度。
    def _norm(self, x):
        # 定义核心的归一化逻辑。
        # x.pow(2).mean(-1, keepdim=True)，计算最后一个维度上每个元素的平方的平均值（均方值）。在 PyTorch 中，x.pow(2) 是一个 Element-wise（逐元素） 操作。
        # torch.rsqrt(... + self.eps)，对均方值加上 eps 后开根号再取倒数，即 \frac{1}{\sqrt{Mean(x^2) + \epsilon}}。

        # 整个 RMSNorm 的过程为：
        # 1. 假设输入的是 x = \begin{bmatrix} 
                            # 3 & 4 & 0 \\
                            # 1 & 1 & 1 
                            # \end{bmatrix}
        # 2. 那么就会按照行去计算均值，得到结果。
        # 3. 然后按照广播进行逐元素的乘法（这里是乘一个倒数，因此就是乘法）。
        # 这里的 torch.rsqrt 就是开根号再取倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # keepdim=True 的核心作用就是保持维度对齐，从而触发 PyTorch 的广播机制（Broadcasting），rsqrt 结果是 (32, 128, 1)。

    def forward(self, x):
        # x.float() 用来强制转换为 float32 以保证计算精度，防止半精度（float16）溢出。
        # self._norm(...) 用来执行归一化。
        # self.weight * ...：将归一化后的结果乘以可学习参数 weight。
        # .type_as(x)：将结果转换回输入 x 的原始数据类型（如 bfloat16）。
        return self.weight * self._norm(x.float()).type_as(x)

# RoPE 位置编码预计算
# 这段代码不仅是给每个词打上“位置标签”，它还是一套动态调整方案：
# - 短文本：按标准 RoPE 运行。
# - 长文本：激活 YaRN 逻辑，通过修改频率分布（ramp）和缩放注意力（attn_factor），让模型在不重新训练的情况下，看懂比以前更长的文章。
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    # 这段代码是 RoPE（旋转位置嵌入） 的核心实现，并且集成了 YaRN (Yet another RoPE extension method) 策略。
    # 为模型生成一套“刻度尺”，让模型知道 Token 之间的相对距离，同时通过 YaRN 技术让模型能够处理比训练时更长的文本。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        # 如果提供了 rope_scaling，说明模型想要支持长文本外推。
        # 提取 YaRN 相关的超参数
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), 
            rope_scaling.get("beta_slow", 1.0), 
            rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            # 如果当前请求的长度超过了原始训练长度
            # 不仅仅是简单地压缩频率（Linear Scaling），而是根据维度的高低，对不同频段进行不通程度的“拉伸”。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 计算 YaRN 的边界。
            # 由于 RoPE 不同维度的震荡频率不同，YaRN 定义了：
            # - 高频部分（低维度）：不改变，保留精确的相对位置感。
            # - 低频部分（高维度）：进行全量的插值（除以 factor）。
            # - 中间部分：通过 ramp 线性过渡。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
            # 生成一个渐变系数 ramp。根据这个系数调整 freqs。低频维度被压缩（除以 factor），从而在有限的旋转范围内装下更长的序列。


    t = torch.arange(end, device=freqs.device)
    # 生成 [0, 1, 2, ..., end-1] 的时间步
    freqs = torch.outer(t, freqs).float()
    # # 计算外积，得到 (sequence_length, dim//2) 的相位矩阵
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    # 计算 $\cos$ 和 $\sin$。
    # torch.cat: 将矩阵在最后一个维度复制一遍（拼接）。这是为了后续能直接与词向量进行点乘和按位相乘（常用技巧是 $[x, y] \rightarrow [x \cos - y \sin, y \cos + x \sin]$）。
    # attn_factor: YaRN 引入的缩放因子，用于在长文本扩展后修正注意力得分的熵，防止模型注意力变得涣散。
    return freqs_cos, freqs_sin

# 应用位置编码
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 这段代码实现了 RoPE (Rotary Positional Embedding) 的核心逻辑。它通过复数旋转的数学技巧，将位置信息注入到向量中。
    def rotate_half(x):
        # 这是一个辅助函数，用于将输入向量的维度“切半”并重组。
        # 下面是 rotate_half 的具体操作。
        # 由于使用了 ...（Ellipsis）和 dim=-1，无论前面有多少个维度（不管是 bsz、seq_len 还是 heads），这个操作只针对每个头内部的向量进行旋转准备。
        # x.shape[-1] 就是 head_dim，代码的执行步骤为：
        # 1. 计算中点：x.shape[-1] // 2 结果是 4 // 2 = 2。
        # 2. 取后半部分并取负：-x[..., 2:] 得到 [-3, -4]。
        # 3. 取前半部分：x[..., :2] 得到 [1, 2]。
        # 4. 拼接（Cat）：将两者按最后维度拼接。

        # 如果是一个形状为 (batch, heads, seq, 64) 的张量，它会保持前三个维度不动。
        # 只在最后一个维度（长度 64）上，把后 32 个元素取负并搬到前 32 个位置，把前 32 个元素搬到后 32 个位置。
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
        # dim 决定了“缝合”的方向，而 ... 决定了“哪些地方不动”。
        # ... (Ellipsis)：代表“不管前面有多少层，统统原样保留”。这是一个极其强大的占位符。在 Python 和 PyTorch 中，它的意思是：自动补全中间所有的维度索引。
        # 假设 x 的形状是 (Batch, Seq, Head, Dim)，也就是 (1, 10, 8, 128)，如果我们写 x[..., :64]，PyTorch 会自动把 ... 替换成 0:1, 0:10, 0:8。它的意思是：“前面的 Batch、Seq、Head 我都要了，我只切最后一个维度（Dim）的前 64 个”。
        # 如果没有 ...，我们可能得写成 x[:, :, :, :64]，万一哪天我们把模型改成了 5 维张量（比如加了并行组），我们的代码就报错了。

        # 我在这里详细介绍一下 torch.cat 的作用：
        # torch.cat（concatenate 的缩写）是 PyTorch 中最常用的张量拼接函数。它的核心作用是将多个张量沿着指定的维度“首尾相接”地拼成一个更大的张量。
        # torch.cat(tensors, dim=0, out=None)，tensors: 一个张量序列（如 [a, b, c]）。注意： 除了拼接的那个维度，其他所有维度的形状必须完全一致。dim: 沿着哪个维度拼接。默认为 0。
        # A = torch.tensor([[1, 2, 3],
        #                 [4, 5, 6]])
        # B = torch.tensor([[7, 8, 9],
        #                 [10, 11, 12]])
        # res0 = torch.cat([A, B], dim=0)
        # 结果形状: (4, 3) -> 行数变多了
        # tensor([[ 1,  2,  3],
        #         [ 4,  5,  6],
        #         [ 7,  8,  9],
        #         [10, 11, 12]])
        # res1 = torch.cat([A, B], dim=1)
        # 结果形状: (2, 6) -> 列数变多了
        # tensor([[ 1,  2,  3,  7,  8,  9],
        #         [ 4,  5,  6, 10, 11, 12]])

        # 这里简单介绍一下 torch.stack()，它不是在现有维度上拼接，而是“凭空”创造一个新维度，把张量叠放进去。 

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    # 它利用三角函数的特性，将 $q$ 和 $k$ 向量在空间中旋转一个角度，从而注入位置信息。
    # q_embed 和 k_embed 就不再是普通的特征向量了，而是“带了指南针”的向量。它们知道自己在句子里的哪个位置，也知道别的词离自己有多远。

    # 为什么只有 QKV 需要旋转？
    # 位置编码是为了计算“关系”，而 V 代表的是“内容”。
    # Q 和 K 旋转是为了在匹配过程中引入“距离感”（比如让模型知道这个词就在我旁边）。
    # V 不旋转是为了保证提取到的“信息内容”是纯净且稳定的。
    return q_embed, k_embed

# GQA 的键值复制
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    # 获取当前的批大小、序列长度、KV 头数以及每个头的维度。
    # 此时 x 的形状是 (Batch, Seq, KV_Heads, Dim)。
    if n_rep == 1:
        # 如果 n_rep 是 1，说明 Q 头和 KV 头的数量已经相等（即标准的 Multi-Head Attention），直接返回原张量，不做任何多余操作。
        return x
    return (
        # x[:, :, :, None, :] 等用于 x.unsqueeze(3)。
        # 假设输入 x 是 KV 张量，形状为 (bsz, seq_len, num_kv_heads, head_dim)，
        # 执行 x[:, :, :, None, :] 后，形状变为 (bsz, seq_len, num_kv_heads, 1, head_dim)。

        # 在上一行代码 x[:, :, :, None, :] 之后，张量的形状是：
        # (bs, slen, num_kv_heads, 1, head_dim)
        # 当我们执行 .expand(bs, slen, num_kv_heads, n_rep, head_dim) 时：
        # PyTorch 发现第 4 维（索引为 3）原来是 1，而我们要求变成 n_rep（比如 4）。
        # 于是，它在逻辑上把这一个维度扩充了 4 倍。

        # torch.repeat：会实打实地申请新的内存空间，把数据拷贝 4 份。torch.expand：完全不复制数据。它只是修改了张量的“步长”（Stride）。它极其节省内存，操作几乎是瞬间完成的。
        # 从换元法的角度考虑，他是从 1 * head_dim 变成了 n_rep * head_dim。
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        # 现在 x 的 size 变成了 (bs, slen, num_key_value_heads, n_rep, head_dim)，做了 reshape 之后变为：
        # (bs, slen, num_key_value_heads * n_rep, head_dim)。

        # 这里有点类似于 view，下面讲一下 view 和 reshape 的区别：
        # view 的核心是，不移动任何数据，只改变对内存的“解释方式”。
        # 它要求张量在内存中必须是连续的（Contiguous）。
        # 它直接修改张量的 Shape 和 Stride（步长）元数据。
        # 如果我们对张量做过 transpose(1, 2) 或 permute，内存里的数据顺序已经和逻辑顺序不一致了。此时调用 view 会报错。

        # reshape 的核心是：只要能变，怎么都行。
        # 其逻辑是先尝试调用 view（如果内存连续，速度极快）。
        # 如果内存不连续（比如转置过），它不会报错，而是默默地在后台执行 .contiguous().view(...)。
        # 那么，如果内存不连续，reshape 会**克隆（Clone）**一份数据到新的内存地址。这意味着它可能会产生额外的显存开销。
    )

# 多头注意力（支持 FlashAttention）
class Attention(nn.Module):
    # 这段代码是一个基于 PyTorch 实现的 多头自注意力机制（Multi-Head Attention），它特别支持了 GQA（Grouped-Query Attention） 分组查询注意力以及 Flash Attention 加速。
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # self.num_key_value_heads = ...: 确定 KV 头的数量。如果配置文件没指定，就和 Query 头的数量一致（即标准多头注意力 MHA）。
        # args.num_key_value_heads 就是 KV 的头数。
        # args.num_attention_heads 就是 Q 的头数。

        # 在传统的 MHA (Multi-Head Attention) 中，Q、K、V 的头数是 1:1:1。而在 GQA (Grouped-Query Attention) 中每一个 Token 依然拥有完整的、与其 num_heads 对应的 Q。但是，多个 Q 头会共用同一个 KV 头。
        assert args.num_attention_heads % self.num_key_value_heads == 0
        # 确保 Query 头的数量能被 KV 头的数量整除，这是实现 GQA 的前提。
        # 同时，这里还需要保证 hidden size 能够被 num_attention_heads 整除。
        self.n_local_heads = args.num_attention_heads
        # 记录 Query (Q) 的总头数。
        self.n_local_kv_heads = self.num_key_value_heads
        # 记录 Key 和 Value (KV) 的总头数。在 GQA 架构中，这个值通常小于 Q 的头数。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 计算 分组倍数。比如有 8 个 Q 头，2 个 KV 头，那么 n_rep = 4，意味着每 4 个 Q 头共用 1 个 KV 头。
        self.head_dim = args.hidden_size // args.num_attention_heads
        # 计算 每个头的维度。例如 hidden_size=512，有 8 个头，那么每个头维度就是 64。
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        # Wq：定义 Q 的投影矩阵。输出的总维度是 $num\_heads \times head\_dim$（通常等于 hidden_size）。
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # Wk：定义 K 的投影矩阵。注意：它的输出维度由 num_key_value_heads 决定，比 Q 短。
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # Wv：定义 V 的投影矩阵。同样，为了实现 GQA，其维度与 K 保持一致。
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        # Wo：定义输出投影层。它将所有头拼接后的结果映射回 hidden_size 维度，用于残差连接。
        self.attn_dropout = nn.Dropout(args.dropout)
        # 用于注意力权重矩阵（Softmax 之后）的 Dropout。
        self.resid_dropout = nn.Dropout(args.dropout)
        # 用于最终输出投影（o_proj 之后）的 Dropout，增强模型泛化能力。
        self.dropout = args.dropout
        # 将 dropout 概率存为变量，方便后续在调用 Flash Attention 函数时直接传参。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 1. 检查 PyTorch 是否自带高效的 scaled_dot_product_attention 函数。
        # 2. 检查配置 args.flash_attn 是否开启。
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        #  x: 输入张量，形状通常为 (Batch, Seq_Len, Hidden_Size)。
        # position_embeddings: 预计算好的旋转位置编码（RoPE）所需的 cos 和 sin 值。
        # past_key_value: 推理加速用的 KV 缓存。如果是训练阶段，通常为 None。
        # use_cache: 布尔值，决定是否返回更新后的 KV 缓存。
        # attention_mask: 掩码，用于遮盖 Padding 部分或防止模型看到未来信息。

        bsz, seq_len, _ = x.shape
        # 获取当前的 Batch Size (bsz) 和 序列长度 (seq_len)。第三个维度是 hidden_size，这里用 _ 忽略，因为投影层已经定义好了。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 将输入 x 分别通过三个线性层，映射到 Q、K、V 空间。
        # 注意由于采用了 GQA，xq 的维度通常比 xk 和 xv 大（因为 Q 的头数更多）。

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # 维度重塑，这一步是将扁平的输出向量**“切分”成多个头**。
        # 转换后的维度为：(Batch, Seq_Len, Num_Heads, Head_Dim)。
        # 这里的 n_local_heads (Q) 和 n_local_kv_heads (K/V) 体现了 GQA 的非对称性。

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # 调用外部函数 apply_rotary_pos_emb。
        # 它利用复数旋转的思想，将 cos 和 sin 注入到 xq 和 xk 中。
        # RoPE 让模型能够通过 Q 和 K 向量之间的相对夹角来感知 Token 之间的相对距离，而不是依赖绝对位置。
        # 只有 Q 和 K 需要位置编码来计算注意力得分，V 存储的是内容信息，不需要旋转。

        # kv_cache实现
        if past_key_value is not None:
            # 检查是否传入了之前步骤缓存的 K 和 V。
            # 在生成式对话中，模型每产生一个新 Token，不需要重新计算前面所有 Token 的 K 和 V，直接从缓存里拿即可。
            # past_key_values = [
            # --- 第一层 (Layer 0) ---
            #     (
            #         torch.randn(1, 4, 3, 8), # Key 张量: [Batch, Seq, Heads, Dim]
            #         torch.randn(1, 4, 3, 8)  # Value 张量: [Batch, Seq, Heads, Dim]
            #     ),
            #     # --- 第二层 (Layer 1) ---
            #     (
            #         torch.randn(1, 4, 3, 8), # Key 张量
            #         torch.randn(1, 4, 3, 8)  # Value 张量
            #     )
            # ]
            # 这里的 past_key_value 就是 (
            #     torch.randn(1, 4, 3, 8), # Key 张量: [Batch, Seq, Heads, Dim]
            #     torch.randn(1, 4, 3, 8)  # Value 张量: [Batch, Seq, Heads, Dim]
            # ),
            # ⭐️⭐️⭐️ 理解高维张量时可以采用换元法。
            xk = torch.cat([past_key_value[0], xk], dim=1)
            # 把新的 Key 接在旧的 Key 后面，形成一个包含“过去+现在”完整上下文的 Key 矩阵。
            xv = torch.cat([past_key_value[1], xv], dim=1)
            # 把当前新 Token 的 Value 拼接到旧的 Value 缓存后面。
        past_kv = (xk, xv) if use_cache else None
        # 如果 use_cache 为真，则将更新后的（包含了当前 Token 的）完整 K 和 V 返回，供下一个时间步使用。
        # 这里的 past_kv 只在推理时生效。

        # 这里通过维度交换和数据复制，把张量调整成符合矩阵乘法要求的格式。
        xq, xk, xv = (
            xq.transpose(1, 2),
            # xq 的形状是 (bsz, seq_len, n_heads, head_dim)。
            # 第 1 维是序列长度，第 2 维是头数。
            # transpose(1, 2) 交换了这两维。
            # 结果形状 (bsz, n_heads, seq_len, head_dim)。
            # PyTorch 的批量矩阵乘法（@）操作的是张量的最后两维，因此我们需要交换。
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            # 完成这里的 repeat_kv 之后，输出向量就变成了 (bsz, seq_len, n_heads, head_dim)，
            # 然后进行 transpose(1,2) 就变成了 (bsz, n_heads, seq_len, head_dim)。
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention 虽快，但它对输入非常“挑剔”。这一长串判断是在确认：当前场景是否满足开启高速通道的条件？
            # 1. self.flash 代表 PyTorch 自带高效的 scaled_dot_product_attention 函数且用户开启了 flash_attn。
            # 2. 如果序列长度只有 1（比如推理时逐字生成），Flash Attention 的并行优势发挥不出来，普通的矩阵乘法反而可能更快。
            # 3. past_key_value is None 意味着这是全量并行计算（训练或 Prefill 阶段），而不是增量生成阶段。目前的 Flash Attention 主要是为大规模并行计算设计的。
            # 4. attention_mask is None or torch.all(attention_mask == 1)。检查是否没有自定义的复杂掩码。如果所有的 mask 都是 1（即全部可见），或者干脆没有 mask，那么就可以使用预设的模式。

            # 这里的 Attention mask 其实就是 Padding Mask。这里的限制条件是说，当这行代码执行时，逻辑上已经确认了除了因果掩码外，没有其他需要屏蔽的位置。既然没有 Padding，那就不需要把额外的 Mask 传给算子。
            # 为什么这里要限制 Attention Mask 为 None 或者无 Padding 呢？
            # 这是因为 FlashAttention 这种极速计算最讨厌“不规则”的东西。
            # 如果每个 Batch 的 Padding 长度都不一样，算子处理起来会变复杂。
            # 所以很多实现（包括 Llama 原生代码）会倾向于如果有复杂的 Padding Mask，就走普通的手动路径；如果场景简单（全是有效词 + 只有因果掩码），就走 Flash 路径。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            # 这是 PyTorch 2.0+ 引入的神级函数。它把原本需要好几行代码完成的动作，全部打包进了一个底层的 C++/CUDA 算子中。
            # 它为什么快？普通的注意力计算（手动写的那种）会产生巨大的中间变量注意力分数 Score。
            # Flash Attention 利用 Tiling（分块） 技术，在 SRAM（GPU 内部的高速缓存）里就把计算做完了，根本不往显存里存那张大表。
            # 当我们设置 is_causal=True 时，我们告诉算子：请帮我自动做一个下三角掩码（Casual Mask），让每个词只能看到它之前的词。
            # 这省去了我们手动创建一个 (seq, seq) 矩阵并填入 -inf 的麻烦。
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 此时 Q 和 K 的形状都是 (bs, n_heads, seq_len, head_dim)
            # 将 K 的最后两维转置，变成 (head_dim, seq_len)，然后进行矩阵乘法。
            # 得到 scores，形状为 (bs, n_heads, seq_len, seq_len)。这张图里的每一个点 (i, j) 代表了第 i 个词对第 j 个词的“关注度”。
            # / math.sqrt(self.head_dim) 是缩放，防止点积数值过大导致 Softmax 后的梯度消失（变得太尖锐）。

            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            # torch.full((seq_len, seq_len), float("-inf"), device=scores.device) 会生成一个全是 -inf 的方阵，如下所示：
            # [[-inf, -inf],
            # [-inf, -inf]]
            # torch.triu(..., diagonal=1) 之后，只保留主对角线以上的部分，其余变 0。
            # [[0, -inf],
            #  [0,   0]]
            # 加上 scores 之后，左下角（含对角线）加上了 0：原得分保持不变。右上角加上了 -inf：原得分变成负无穷。

            # 可以说，torch.triu 这个函数的本质就是保留“上三角”，把“剩下”的部分变成 0。
            # 这里之所以设计为 diagonal=1，就是为了看到自己，如下所示：
            # [[   0, -inf, -inf],
            #  [   0,    0, -inf],
            #  [   0,    0,    0]]

            # 关于这里为什么是 scores[:, :, :, -seq_len:]，而不是 scores[:, :, :, :]？
            # ⭐️⭐️⭐️ 这其实需要解释一下 x.shape 在不同阶段下的变换。
            # 1. 在训练阶段，我们是并行处理一整串 Token 的。x 的形状变化非常死板且固定，就是 (B, L, D)。
            # 2. 在推理的 Prefilling，这时的形状和训练阶段几乎一模一样，因为它也是一次性处理 L 个词，因此也是 (B, L, D)。
            # 3. 在推理的 Decoding，这是我们最需要关注的，此时 x 变得不一样，他变成了 [B, 1, D]，注意：中间永远是 1，因为一次只投喂一个新词。

            # 此外，在推理的 Decoding 阶段，由于 KV Cache 的存在，虽然 Q 的 seq_len 是 1，但是 KV 的 seq_len 是 L。
            # 注意，这里的 Q 是当前 token 的 Q，预测的是下一个 token。

            # 所以，所以在推理的 decoding 阶段，这里的 scores 的 size 是什么？
            # 是 (bs, n_heads, 1, L)。
            # 这里的 seq_len == 1，这在推理阶段的 Decoding 时，几乎是没有意义的，因为这等效于 scores[:, :, :, -1:] += 0。
            # 从物理意义上讲，这完全行得通 =>
            # 1. Query 只有 1 个：就是当前词。
            # 2. Keys 有 T 个：全都是过去发生的（或者是当前词自己）。
            # 3. 因果律：当前词看过去，没有任何问题。它根本不需要“遮盖”任何东西，因为它后面还没有词呢！
            # 所以，Mask 变成全 0（即不遮挡任何东西），逻辑是自洽的。

            # 但这段代码可以同时兼容 Prefilling（预填充） 和 Decoding（解码）。在 Prefilling 时，seq_len 是 N（比如 512）。此时 mask 是一个 512 \times 512 的矩阵，triu 会切掉右上角，非常关键。通过写成一行通用的代码，作者不需要写繁琐的 if-else。
            # 在一些高级推理技术中（如投机采样），模型可能会一次性尝试预测并验证 5 个词。此时 seq_len = 5。这时，这 5 个新词之间就产生了“因果关系”（第 1 个不能看第 5 个）。此时 -seq_len: 配合 triu 就能立刻发挥作用，确保这小块 5 \times 5 的区域符合因果律。

            # 综上就解释了为什么是 scores[:, :, :, -seq_len:]，而不是 scores[:, :, :, :]，这里的 seq_len 就是 x 的 seq_len，也可以是做要并行预测的 token 数。
            if attention_mask is not None:
                # Attention mask 不为 None，这代表着存在 Padding Mask。
                # attention_mask 形状通常是 (bs, total_len)。里面只有 1（有效词）和 0（Padding 占位符）。
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 进行 .unsqueeze(1).unsqueeze(2) 之后，mask 形状变成了 (bs, 1, 1, total_len)。
                # 这是为了利用 广播机制 (Broadcasting)。scores 的形状是 (bs, n_heads, seq_len, total_len)。通过增加这两个维度，这个 Mask 就能像一张“滤网”一样，自动复制并覆盖到每一个 head 和每一个 query 上。
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                # 经过这一步变换，有效词部分：0 * -1e9 = 0；Padding 部分：1 * -1e9 = -1,000,000,000（一个极大的负数）。
                scores = scores + extended_attention_mask
                # scores 是 (bs, n_heads, seq_len, seq_len)，第二个 seq_len 其实就是带了 kvcache 的 total_len。
                # extended_attention_mask 是 (bs, 1, 1, total_len)。
                # 将处理好的掩码直接加到原始得分 scores 上。
                # 结果就是有效词的位置：scores + 0 = scores（得分保持不变）。Padding 的位置：scores + (-1e9) \approx - \infty（得分变得极小）。

            # 现在的 scores 就是做了因果掩码和 Padding 掩码的 score 了。
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # .float(): 这是一个非常关键的细节。在深度学习中，Softmax 涉及指数运算（$e^x$），如果使用半精度（FP16/BF16），很容易出现数值溢出或不精准。所以先转成 FP32 计算，保证稳定。
            # dim=-1: 在最后一维（即 total_len 维度）做归一化，表明了对于某个 Q 来说，所有的 K 对于他的重要性。
            # .type_as(xq) 算完后再转回原先的低精度（如 BF16），为了和后面的张量对齐并节省显存。
            
            # 这里为什么需要转 FP32？
            # 这是一个非常经典且关键的数值稳定性（Numerical Stability）问题。简单来说，Softmax 对精度极其敏感，而低精度（FP16）容易让模型“变瞎”或“崩溃”。
            # 对于 Softmax 操作转回 FP32 的操作，这是非常常见的，我们一般称之为混合精度训练。
            # 一般来说，大部分的运算走 bf16 精度就够用了，但是对于像 Softmax、LayerNorm 和位置编码等操作，我们都需要 .float() 将其由 BF16 转 FP32。
            # 但是我们之前是使用 AMP 进行混合精度管理的，只需要使用 with torch.cuda.amp.autocast()，大部分混合精度训练都不需要手动。
            # 一般来讲，写业务代码/微调模型时，我们不需要管混合精度训练，autocast 一开万事大吉。
            # 写模型底层架构（如 Attention）时，我们必须像个老司机一样，在 Softmax、Norm 等关键弯道，手动踩下 FP32 的刹车。
            
            # with torch.cuda.amp.autocast() 底层的实现原理是什么呢？
            # 当我们调用一个函数（比如 F.softmax）时，PyTorch 的 Dispatcher（调度器） 会先检查当前是否在 autocast 环境下。如果是，它会查找这个函数在不在黑名单里。如果在，它会偷偷帮我们做一个 .float() 转换。
            
            # Pytorch 原生算子。通常只要你调用的是 torch.xxx、torch.nn.functional.xxx 或张量自带的方法，它们绝大多数都是原生算子。
            scores = self.attn_dropout(scores)
            # Dropout。
            # 在训练阶段，这是为了防止模型只盯着某一个词看，强迫它去挖掘其他词的潜在联系。
            # dropout 在推理时是自动关闭的（通常由 model.eval() 触发）。此时这行代码相当于 scores = scores，啥也不干。
            output = scores @ xv
            # 这是整个 Attention 模块最核心的加权求和操作。
            # scores：形状是 (bs, n_heads, seq_len, total_len)。
            # 它的物理含义是：“权重”。每一行都代表当前 Query 对所有历史位置的关注程度。
            # xv：形状是 (bs, n_heads, total_len, head_dim)。
            # 它的物理含义是：“内容”。每一行都是历史 Token 真正携带的信息特征。
            # @ 计算过程：(seq_len × total_len) @ (total_len × head_dim)。
            # 结果：(bs, n_heads, seq_len, head_dim)。

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 第一步的 .transpose(1,2) 是在进行维度变换，变成 (bs, seq_len, n_heads, head_dim)。
        # 然后进行 reshape，将维度压缩为 (bs, seq_len, embedding_dim)。
        output = self.resid_dropout(self.o_proj(output))
        # o_proj 是用来进行信息糅合的。
        # 这是一个线性层（Linear Layer），也叫 Output Projection。
        # 刚才的 reshape 只是简单的“物理拼接”，32 个头之间的信息依然是各说各的。o_proj 的矩阵乘法让这 D 个维度之间进行全连接计算。它相当于一个总编，把 32 个专家的意见进行加权整合，提取出最终的语义表示。
        # resid_dropout 也是 Dropout。
        return output, past_kv

# 简单前馈网络
class FeedForward(nn.Module):
    # 定义一个类，继承自 PyTorch 的基础模型类 nn.Module。
    # 这个 FeedForward 其实就是一个 SwiGLU 结构，它是由两个组件组合而成：
    # - GLU (Gated Linear Unit)：指代那种“两路并行再相乘”的结构（$A \otimes B$）。
    # - Swish (SiLU)：指代 gate_proj 之后用的那个特定的激活函数。

    # SwiGLU 的信息流向如下：
    # 1. x 流向 Up，同时 x 流向 Gate。
    # 2. Gate 的流出进入 silu，然后 silu 的结果和 up 的流出进行点乘。
    # 3. 最后的结果流出 Down。
    # 总体为 FFN_{SwiGLU}(x) = (Swish(xW_{gate}) \otimes xW_{up})W_{down}

    def __init__(self, config: MiniMindConfig):
        # 传入一个配置对象，里面包含了模型的所有超参数（如维度、激活函数类型等）。
        super().__init__()
        if config.intermediate_size is None:
            # 如果没有手动指定中间层维度（intermediate_size），则自动计算。
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 大模型（如 Llama）常用 hidden_size \times \frac{8}{3} 作为中间层宽度。
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            # 最后的计算是为了确保维度是 64 的倍数。这有助于 GPU 利用内存对齐（Memory Alignment）来加速计算。
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # “门控”层。将输入向量放大，准备通过激活函数产生“开关”信号。
        # 他的本质作用就是去实现条件过滤，up_proj 负责产生候选特征（大量的原始信息）。gate_proj 经过激活函数（SiLU）后，产生一个掩码（Mask）。
        # 当两者相乘时，如果 gate_proj 在某个位置输出接近 0，那么 up_proj 在该位置提取的特征就会被“杀掉”；如果接近大于 0，则允许通过。
        # 通俗点说，up_proj 是“可能有用的信息”，而 gate_proj 是根据当前上下文判断“哪些信息真的有用”。
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # “下降”层。将处理后的复杂特征映射回原始维度，保持特征维度一致。
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # “上升”层。同样放大向量，存储主要的特征信息。
        # up_proj 可以从原来的 hidden_size 中，经过参数的组合变化，提取出更多的特征。
        # 所有的 Linear 都不使用 Bias，这是目前大语言模型的通用做法，可以减少参数并微弱提升训练稳定性。
        self.dropout = nn.Dropout(config.dropout)
        # 随机丢弃一部分神经元防止过拟合。
        self.act_fn = ACT2FN[config.hidden_act] # silu。
        # 激活函数（通常是 SiLU，也叫 Swish）。ACT2FN 是一个字典，根据字符串映射到具体的函数。

    def forward(self, x):
        # # 1. gate_proj(x) -> 线性变换
        # 2. act_fn(...) -> 经过 SiLU 激活，产生 0 到 1 之间的权重（门控信号）
        # 3. up_proj(x) -> 另一个线性变换，获取特征
        # 4. * -> 两者逐元素相乘。门控信号决定了特征中哪些部分该“通过”
        # 5. self.down_proj(...) -> 将维度压回 hidden_size
        # 6. self.dropout(...) -> 应用随机失活
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

# 混合专家的路由门控
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        # 每个 Token 会被路由到前 k 个专家。
        self.n_routed_experts = config.n_routed_experts
        # 总共有多少个可选的专家。
        self.scoring_func = config.scoring_func
        # 多个专家的评分设计。
        # 通常是 softmax，决定如何将门控的原始输出转化为概率分布。
        self.alpha = config.aux_loss_alpha
        # 辅助损失系数。用于控制负载均衡损失（Aux Loss）在总损失中的权重，防止某些专家被“累死”，某些被“闲死”。
        # 在配置项中这个值被设置为 0.01

        self.seq_aux = config.seq_aux
        # 一个布尔开关，决定是否启用 序列级辅助损失（Sequence-level auxiliary loss），这是一种更细粒度的负载均衡策略。
        # 在配置项中这个值被设置为 True。

        self.norm_topk_prob = config.norm_topk_prob
        # 一个布尔开关，决定选出 Top-K 个专家分数后，是否要对这 $k$ 个分数进行 重新归一化（使它们相加等于 1）。
        # 在配置项中这个值被设置为 True。

        self.gating_dim = config.hidden_size
        # 门控输入维度。即模型隐藏层的维度（例如 512 或 1024），门控器需要在这个维度上进行线性投影。

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 核心可学习参数。创建门控权重矩阵。
        # 它的形状是 (专家数, 隐藏层维度)。
        # 他的原理相当于为每个专家学习一个“原型向量”。
        # 输入向量与这个权重矩阵相乘，本质上是在计算输入与各个专家的相似度。
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # 这里的 init.kaiming_uniform_ 只对门控权重网络 self.weight 使用 Kaiming Uniform (He初始化) 方案。
        # MoE 的门控层本质上是一个线性层。这种初始化方法能保证在深度网络中，信号在每一层传递时方差保持稳定，避免训练初期的梯度消失或爆炸。参数 a=math.sqrt(5) 是 PyTorch 默认线性层（nn.Linear）的标准初始化配置。

        # PyTorch 内置的“层”（Modules）绝大多数都自带了相对科学的默认初始化，只有 nn.Parameter 除外。
        # 当我们使用 PyTorch 提供的标准层时，它们在构造函数 __init__ 里都会自动调用一个类似 reset_parameters() 的方法。

    def forward(self, hidden_states):
        # 这段 forward 函数是 MoE 门控机制的核心逻辑。它负责两件事：
        # 第一，决定每个 Token 去哪（路由）；
        # 第二，计算负载均衡损失（防止专家闲置）。
        bsz, seq_len, h = hidden_states.shape
        # 维度获取——获取批次大小 (bsz)、序列长度 (seq_len) 和隐藏层维度 (h)。
        hidden_states = hidden_states.view(-1, h)
        # 打平 (Flatten)，将三维张量压扁成二维 (bsz * seq_len, h)，因为门控打分是针对每个 Token 独立进行的，不需要考虑序列结构。
        logits = F.linear(hidden_states, self.weight, None)
        # 用输入的 Token 特征与专家权重矩阵 self.weight 做矩阵乘法。计算结果 logits 的形状是 (总Token数, 专家总数)。
        # 那么，为什么这里不能用 logits = self.weight(hidden_states) 呢？
        # 核心原因是 self.weight 是一个张量（Tensor）。它只是一个存储数字的矩阵，它没有实现 __call__ 方法。如果我们写 self.weight(hidden_states)，Python 会报错：TypeError: 'Parameter' object is not callable。
        # 如果我们定义的是 self.gate = nn.Linear(...)，那么 self.gate 是一个包含权重、偏置和 forward 逻辑的对象。这种情况下我们才能像函数一样调用它：logits = self.gate(hidden_states)。

        # F.linear 的参数是 input, weight, bias，正好对应 y = input * weight + bias。
        # 这里的 weight 其实在计算时会被转置，因此 hidden_states 是 (T, D)，而 weight.T 则是（D，E），所以最终的 logtis 是 (总Token数, 专家总数)。
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
            # 将原始分数 logits 转化为概率分布。每个 Token 对所有专家的概率和为 1.0。
            # 如果 dim=-1：Softmax 会横着算。它取出一个 Token 对所有专家的 N 个分值，把它们变成概率。
            # 如果 dim=0（错误做法）：Softmax 会竖着算。它会去比较“第 1 个 Token 对专家 A 的分数”和“第 2 个 Token 对专家 A 的分数”，这在逻辑上是毫无意义的。
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # 在前一步，我们算出了每个 Token 对所有专家（比如 8 个）的概率得分。但在 MoE 设计中，我们不想让所有专家都处理每一个 Token（那样太慢了），我们只想选出最适合的那 $k$ 个。
        # scores 形状是 (Total_Tokens, n_experts)。里面是每个 Token 对应所有专家的概率。
        # k=self.top_k: 这是一个超参数（通常是 1 或 2）。它告诉 PyTorch：“我只要概率最大的前 $k$ 个”。
        # dim=-1: 同样是在“专家”这个维度进行选拔。
        # sorted=False: 这是一个性能优化小技巧。如果为 True，返回的前 $k$ 个结果会按得分从高到低排好序。如果为 False，它只保证抓出最大的 $k$ 个，但不保证这 $k$ 个内部的顺序。在 MoE 中，我们通常不在乎这 $k$ 个谁先谁后，只要选出来就行，所以设为 False 能跑得更快一点。

        # topk_weight (选中的权重):存的是这 $k$ 个专家的原始概率分。例如：某个 Token 对 8 个专家的分数里，最高的两个是 0.85 和 0.10，那么这里就存 [0.85, 0.10]。
        # topk_idx (选中的索引):存的是这 $k$ 个专家到底是谁（编号）。例如：如果得分最高的是第 3 号和第 7 号专家，这里就存 [3, 7]。
        # topk_weight 和 topk_idx 的 size 都是（总Token数， topK 专家数）。

        if self.top_k > 1 and self.norm_topk_prob:
            # 当我们从所有专家中选出前 $k$ 个后，这几个人的分数加起来往往不等于 1。这段代码通过除以它们的总和，强行让这 $k$ 个人的权重加起来重新等于 100%。
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # topk_weight.sum(dim=-1, keepdim=True) 表示对选出来的 （总Token数， topK 专家数）按照最后一个维度进行求和。
            # 由于 tensor.sum() 一般会减少一个维度，因此 keepdim=True 保持维度，方便后面做除法（形状保持为 (Total_Tokens, 1)。
            # + 1e-20: 这是一个极小的常数（Epsilon），防止万一权重和是 0（虽然 Softmax 后几乎不可能），避免产生“除以零”导致的 NaN 错误。
            topk_weight = topk_weight / denominator
            # 这里的除法利用了 PyTorch 的 广播机制 (Broadcasting)。
            # 将 (Total_Tokens, k) 的权重矩阵除以 (Total_Tokens, 1) 的总和矩阵，实现每个 Token 内部权重的重分配。

            # 广播机制会比 for 循环快出了几个数量级。
            # 在深度学习中，我们有一个原则：能用矩阵运算（向量化）就绝对不用 for 循环。
            # 广播机制是在底层（C++ 或 CUDA 驱动级）实现的。当我们执行 topk_weight / denominator 时，PyTorch 并不会真的在内存里把 denominator 复制一份，它只是在计算时逻辑上把维度对齐，然后直接调用高度优化的 CPU (AVX/SIMD) 或 GPU (Parallel Threads) 指令集。
            # 而 for 循环是串行的。必须算完第 1 个 Token，才能算第 2 个。如果我们有 8192 个 Token，就要排队 8192 次。
            # 广播 + 矩阵运算是高度并行的。在 GPU 上，成千上万个核心会同时处理这 8192 个 Token 的除法。

        if self.training and self.alpha > 0.0:
            # 只在训练模式下计算 Loss。推理（Inference）时不需要平衡专家。
            # 只有设置了损失系数（罚款力度）才计算。
            scores_for_aux = scores
            # scores_for_aux: 缓存所有专家的 Softmax 得分，形状 (Total_Tokens, n_experts)。
            # 在这里补充一个 python 的基本知识。
            # Python 没有“块级作用域”（Block Scope）。
            # 在 Python 中，if/else、for、while、with 等语句块不会开启新的作用域。只有 模块（Module）、类（Class） 和 函数（Function） 才会开启新的作用域。

            # 只要代码运行路径经过了变量定义的赋值语句，该变量在整个函数内部（即 forward 函数内）都是可见的。
            # Python 查找变量的顺序是：
            # L (Local)：函数内部。
            # E (Enclosing)：外部嵌套函数（闭包）。
            # G (Global)：模块级全局变量。
            # B (Built-in)：内建函数（如 len, range）。
            # 由于 if 块属于 Local 作用域的一部分，所以在 if 里面定义的变量，在 if 外面（只要还在同一个函数里）依然是同一个 Local 变量。

            # 虽然 Python 没有块作用域，但它有 “执行路径” 的限制。
            # def test(condition):
            # if condition:
            #     x = 10
            # print(x)  # 如果 condition 为 False，这里会报 NameError: name 'x' is not defined
            # 如果进入了 if：x 被赋值，print 没问题。
            # 如果没进 if：x 压根没被创建，print 就会崩溃。

            aux_topk = self.top_k
            # 计算 aux loss，其中的 aux_topk 用来保存选出的专家数量。
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            # 将选中的专家索引从 (Total_Tokens, k) 重新折叠回 (Batch_Size, Seq_Len * k)。这是为了方便按“句子”或“序列”来统计。

            if self.seq_aux:
                # 序列级辅助损失，这种模式要求每一条样本（句子）内部都要实现专家均衡。
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 将原始得分恢复成三维：(Batch, Sequence, Experts)。
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # 创建一个全 0 的计数器，准备记录每个 Batch 里每个专家被选中的次数。
                # 这里的 n_routed_experts 就是总共的专家个数。
                # 因此 ce 的 size 为 (Batch, 专家个数)，用来记录每个 batch 触发了哪些专家。
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # scatter_add_ 的三个参数分别是 dim、index、src。
                # ce 初始形状是 (bsz, n_experts)，全是 0。
                # index=topk_idx_for_aux_loss 形状是 (bsz, seq_len * k)，存储了每个 Token 选中的专家 ID。
                # src=torch.ones(...) 产生一堆“1”。
                # target[i][index[i][j]] += src[i][j]

                # scatter_add_ 这种向量化操作要比 for 循环快上几百倍，甚至上千倍。
                # 后面的 .div_(seq_len * aux_topk / self.n_routed_experts) 的目的是计算偏离度。
                # seq_len * aux_topk 代表一整条序列总共发出了多少次“专家邀请”，/ self.n_routed_experts: 这是理想情况。如果雨露均沾，每个专家应该分到这么多任务。
                # .div_(...): 用实际次数除以理想次数。
                # 如果某个专家分到的任务正好是平均值，ce 对应位置就是 1.0。
                # 如果某个专家被冷落（次数为 0），ce 就是 0.0。
                # 如果某个专家被过度使用，ce 就会 大于 1.0。

                # 在 Pytorch 中 为什么有的方法后缀有一个 _？
                # 这是一个非常关键的 PyTorch 编程细节。在 PyTorch 中，方法后缀带有一个下划线（如 .div_()、.add_()、.scatter_add_()）表示这是一个 原地操作 (In-place Operation)。
                # 简单来说，它会直接修改调用它的那个张量（Tensor）本身，而不是创建一个新的张量并返回。
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
                # ce 的 size 是 (bsz, n_routed_experts)，每行代表一个句子中，各个专家的任务达成率。
                # 之前的 scores_for_seq_aux size 是 (batch, seq_len, n_routed_experts)。
                # 执行完 mean(dim=1) 之后，PyTorch 会把每个句子中所有 Token 对专家的概率分加起来，然后除以序列长度。这个操作会消掉 seq_len 这一维。结果 Shape 是 (batch, n_routed_experts)，也就是对于一个 batch 而言，各个 expert 占据的比例，这是根据 batch 内所有 token 的均值算出来的。
                # 然后再过一个 sum(dim=1)，这里数据的 size 就是 (batch)，每个数据就是专家维度的和，然后经过 mean() 就是每个 batch 的专家平均数。最后的 self.alpha 就是惩罚系数。
                
                # 在这里说一下 * 和 @ 区别，对于矩阵而言，* 代表逐个元素进行相乘，而 @ 则是标准矩阵乘法。
                # 如何从物理意义上理解矩阵乘法 @ ？
                # 对于 y = x * W，假设 x 是 (1, N), W 是 (N, M)，那么 y 就是 (1, M)。
                # 物理意义可以这样看：可以把 x 看作是输入特征（比如一张图片的像素或一个单词的向量），而把 W 的每一行看作是一个过滤器（Filter）或模板。
                # 矩阵乘法的本质是点积。当 $W$ 的某一行与 $\mathbf{x}$ 进行点积时，结果越大，说明 $\mathbf{x}$ 与这一行所代表的特征越相似。
                # 权重（Weight）的大小决定了输入信号的重要性。大的正权重表示“增强”，负权重表示“抑制”，接近 0 则表示“无视”。
                # 如果 x 是 X，也就是说他的 size 是 (K,N)，无非输入也只是多一些，真正理解时还是只需要理解 x * W，W 就是对 x 特征的过滤。

                # ce * scores_for_seq_aux.mean(dim=1) 这里就是逐元素相乘，但这不是向量的点积。
                # ce 的数据源来自于 topk_idx，其 size 为 (bsz, n_routed_experts)。
                
                # 那么这里就会引入一个新的问题，ce * scores_for_seq_aux.mean(dim=1) 相乘的物理含义是什么？
                # 这里的 ce 表示专家分配的实际行动，而 scores 表示专家分配的心理预期。
                # 将二者的乘积作为 loss，隐式地约束了需要让实际行动和分配心理都要保持均衡，也就是说它只在“大家都挤在一起”的时候，才会产生巨大的惩罚值。如果我们想降低 Loss，我们必须让“大概率”和“大频率”错开。
                # 结果就是模型为了让 Loss 变小，会被迫学习：即使我主观上很喜欢 1 号专家（概率高），我也要把任务分给 2 号专家一些（降低 1 号的频率）。
                
                # 引入的新问题是，为什么不只考虑 ce 或者只考虑 scores？

                # 我们可能会问：“直接惩罚那些干活多的专家不就行了？干嘛还要乘以概率？”，这是因为scores（概率）是可导的，而 ce（计数）是通过索引拿到的，通常不可导。通过乘以 scores，Loss 就可以通过反向传播（Backpropagation）来修改门控器的权重，从而改变未来的分配倾向。如果没有这个乘法，梯度就断了，模型就不知道该怎么调整自己。

                # 那为什么不只考虑 scores 呢？反正只有 scores 也足够反向传播了。在这里就要说明一种情况，也就是思想上公平，但是行动不公平，即 scores 比较均衡，但是 ce 不均衡。出现这种情况的本质原因是 topk << 专家个数。
                # 首先，scores (思想) 是连续的。在 $0.0$ 到 $1.0$ 之间有无限的可能。topk (行动) 是离散的。它是一个 0/1 决策：要么我们被选中（1），要么我们被抛弃（0）。当我们把一个连续的分布强行塞进一个“只选前 k 个”的黑盒里时，我们实际上是把 N-k 个专家的生存权利直接抹杀了。即使某个专家在思想上拿到了 12% 的好感度（仅次于前两名的 13%），在行动上它拿到的也是 0。
                # 这就是 k << N 带来的必然结果，大量“还不错”的专家因为微弱的排名劣势，在宏观统计上被彻底边缘化了。

                # 思想公平只是模型的一种“态度”，行动不公平则是由于 k << N 这种稀疏筛选机制造成的“残酷现实”。如果不引入 ce 这种外部约束，模型就会像一个只看名次不看分数的面试官，永远只录用那两名“面霸”，而让其他同样优秀的专家（MLP 层）在冷宫里因得不到训练而逐渐“废掉”。
            else:
                # 全局辅助损失，它不关心单个句子内部是否均衡，而是要求**整个 Batch（所有 Token 放在一起）**对专家的调用是均匀的。
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # F.one_hot 的作用是生成一个 one-hot 矩阵，他的 row size 就是 topk_idx_for_aux_loss.view(-1) 的 size，也就是 total_token * k，one-hot 的 col size 就是 num_classes，这里也就是专家个数。
                # 具体来说，对应某一行，比如有 4 个专家，Token 选了 #2，就变成 [0, 0, 1, 0]。
                ce = mask_ce.float().mean(0)
                # 首先将 mask_ce 由 int 转化为 float 数据，这是由于在 PyTorch 中，整型是不能做平均值（mean）计算的。
                # 然后进行 mean(dim=0) 的运算，这会再 row 维度进行求均值，size 由 (total_token, experts_num) 变成 experts_num。
                # 这行代码把**“谁选了谁”这种琐碎的个体信息，提炼成了“每个专家干了百分之几的活”**这种全局统计信息。
                Pi = scores_for_aux.mean(0)
                # scores 形状是 (Total_Tokens, n_experts)。里面是每个 Token 对应所有专家的概率。
                # 做了 mean(dim=0) 之后求出来的就是每个专家的概率。
                # 这代表了门控器对这批数据主观上分配概率的平均值。
                fi = ce * self.n_routed_experts
                # 在理想的绝对平均情况下，每个专家应该拿到 1/N 的流量。
                # 乘以 N 之后，理想情况下的 fi 每个元素都应该是 1.0。如果某个专家拿了 2 倍的流量，fi 就是 2.0；如果没拿到，就是 0.0。这步操作将“占比”转化为了“偏离倍数”。
                aux_loss = (Pi * fi).sum() * self.alpha
                # Pi * fi: 还是那个逻辑——“我想给谁”乘以“我真给了谁”。
                # .sum(): 将所有专家的这个乘积加起来。

                # 无论是哪个序列级还是全局级，它们最终都在算同一个公式：
                # Loss = \alpha \cdot N \cdot \sum_{i=1}^{N} P_i \cdot f_i
                # 其中 $P_i$ 是预测概率平均值，$f_i$ 是实际频率。当 $P$ 和 $f$ 都是均匀分布时，这个值达到最小。

                # 什么时候用序列级，什么时候用全局级？
                # 序列级 Loss (Sequence-level)核心逻辑是强制**每一个句子（样本）**内部都要公平地分配专家。适用情况为：
                # - 处理长文本（Long Context）。在长文本中，一个序列（Sequence）可能包含数千个 Token。如果这一个序列就塞满了某一个专家的缓冲区（Capacity），会导致该序列的处理速度极慢。序列级 Loss 强制长文本在内部就把流量摊开。
                # - 训练初期（Warm-up 阶段）。在模型刚开始学习时，专家非常容易“塌陷”（所有 Token 都涌向同一个随机初始化较好的专家）。序列级 Loss 约束更严，能强迫模型在每一个样本处理时都去尝试不同的专家，加速专家多样性的开发。
                # - 推理延迟敏感场景。如果我们的线上服务 Batch Size 很小（甚至为 1），我们必须保证单个请求（Sequence）能够均匀触发多个专家并行工作，而不是串行等待一个专家。
                # 缺点是违背自然语义，有些短句子可能真的只涉及单一主题（比如全是金融词汇），强制它分给 8 个专家（包括文学专家、代码专家）会降低精度，属于“强行摊派”。

                # 全局级 Loss (Global-level)的核心逻辑是不限制单个句子，只要整个 Batch（成百上千个 Token）看过去是平的就行。适用情况为：
                # - 大规模预训练（Pre-training）。在极大的 Batch Size 下（例如 400 万个 Token 一个 Step），只要宏观上专家是平衡的，吞吐量（Throughput）就是最高的。它允许模型根据语义灵活选择：句子 A 全部给专家 #1，句子 B 全部给专家 #2。
                # - 多任务/多语言训练。不同任务的 Token 分布天然不同。全局 Loss 允许专家“术业有专攻”。比如代码专家的活全来自于 Batch 里的代码样本，而不用强行去分担翻译样本的活。
                # - 分布式专家并行（Expert Parallelism）。在模型跨机器分布时，我们更在乎的是总流量是否塞满了 8 张显卡的显存，而不是某一行数据在显卡间的跳跃。
                # 缺点是极端不均风险，如果 Batch 里的句子同质化严重（比如全是同一种风格的垃圾文本），全局 Loss 可能在宏观上觉得平衡，但微观上依然导致某些计算节点过载。
        else:
            aux_loss = scores.new_zeros(1).squeeze()
            # 这是一种防御性编程。
            # 它确保了无论在什么条件下（训练还是推理、单卡还是多卡），aux_loss 永远是一个格式正确、位置正确、数值为 0 的有效对象。
            # 1. 首先他保证了变量作用域的正确性。
            # 2. 其次是为了保证计算图的一致性，在 PyTorch 中，损失函数通常需要参与 total_loss = loss + alpha * aux_loss 的加法运算。
            # * 如果 aux_loss 是 None：加法会报错。
            # * 如果 aux_loss 是普通数字 0：虽然能加，但在某些分布式训练框架中，它可能因为不是 Tensor 而导致同步错误。
            # * 用 new_zeros：它返回的是一个带梯度的 Tensor 占位符。即使它是 0，它也是一个**“合法公民”**，可以顺畅地参与所有矩阵运算。
            # 3. 需要保证 aux_loss 是位于同一个 GPU 上，并且数据类型和 scores 一致，float 才能用于反向传播。squeeze() 是为了保证最后的 loss 是一个 tensor 张量，和之前的 loss 保持一致。
        return topk_idx, topk_weight, aux_loss
        # topk_idx 存的是这 $k$ 个专家到底是谁（编号）。
        # topk_weight (选中的权重):存的是这 $k$ 个专家的原始概率分。例如：某个 Token 对 8 个专家的分数里，最高的两个是 0.85 和 0.10，那么这里就存 [0.85, 0.10]。
        # aux_loss 就是专家负载均衡损失。

# 混合专家前馈
class MOEFeedForward(nn.Module):
    # 这段代码实现了一个经典的混合专家模型（Mixture of Experts, MoE） 前馈网络层。这种结构的核心思想是不让一个巨大的神经网络处理所有信息，而是将其拆分为多个“专家”，由“门控系统”决定哪些专家该出来干活。
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # nn.ModuleList 是一个特殊的列表，用于存储 PyTorch 子模块。如果我们用普通的 Python list，PyTorch 就找不到里面的参数，导致无法训练。
        # experts 就是一个个的 FeedForward。每一个“专家”本质上就是一个标准的前馈神经网络（通常是两层全连接加一个激活函数）。
        # config.n_routed_experts 通过循环创建了指定数量的专家（例如 8 个或 16 个）。这些专家被称为“被路由的”，因为输入数据会根据权重选择性地进入其中的某几个。
        self.gate = MoEGate(config)
        # MoEGate 是 MoE 的“大脑”，它的作用是接收输入向量，计算出一组权重，决定当前的输入应该送给 self.experts 中的哪几个专家处理。
        # 通常它会输出一个稀疏向量，例如对于某个 token，专家 A 的权重是 0.9，专家 B 是 0.1，其余全是 0。
        if config.n_shared_experts > 0:
            # if config.n_shared_experts > 0 是一个条件判断。有些 MoE 设计（如 DeepSeek）会包含“共享专家”。
            # 共享专家和路由专家。
            # 路由专家只有被选中的才干活（节省计算量）。
            # 共享专家无论输入是什么，它们总是参与计算。
            # 这样做通常是为了捕捉所有数据通用的基础知识，防止模型在切换专家时丢失核心语义。
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        # 这段代码展示了 MoE（混合专家模型）最核心的 forward（前向传播） 逻辑，它决定了输入数据如何被分发给不同的专家，以及如何将结果合并。
        identity = x
        # identity 会备份原始输入。如果后面有“共享专家”，它们会直接作用于这个原始输入。
        orig_shape = x.shape
        # 备份 orig_shape。
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 门控网络计算。
        # 1. topk_idx: 每个 token 选中的前 $k$ 个专家的索引。
        # 2. topk_weight: 对应专家的权重（通常是经过 Softmax 的）。这里的 topk_weight 的形状为 (x, 1-x)，这代表选择两个专家，第一个专家概率为 x，第二个专家概率为 1 - x。
        # 3. aux_loss: 辅助损失（用于平衡专家，防止某些专家“太忙”而另一些“太闲”）。
        x = x.view(-1, x.shape[-1])
        # x.view(-1, ...) 将输入展平为 (总 token 数, 隐藏维度)，方便统一处理。
        flat_topk_idx = topk_idx.view(-1)
        # 假设我的输入数据如下。Batch Size (bsz) = 2 （有两个句子）、Sequence Length (seq_len) = 3 （每句话有 3 个词/token）、Top-k = 2 （每个 token 选 2 个专家）。
        # 此时，topk_idx 的原始形状是 (2, 3, 2)。它看起来像这样：
        # [
        # [[专家1, 专家3], [专家2, 专家1], [专家4, 专家2]], # 句子1的三个词选的专家
        # [[专家3, 专家4], [专家1, 专家2], [专家3, 专家2]]  # 句子2的三个词选的专家
        # ]
        # 执行 flat_topk_idx = topk_idx.view(-1) 后，它变成了一个长度为 $2 \times 3 \times 2 = 12$ 的一维长条。
        # [专家1, 专家3, 专家2, 专家1, 专家4, 专家2, 专家3, 专家4, 专家1, 专家2, 专家3, 专家2]
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # torch.repeat_interleave 的作用是对张量中的元素进行“就地连续重复”
            # 假设我们有一排人 [A, B, C]。普通的 repeat(2)（整体复制）结果是 [A, B, C, A, B, C]（排两次队）。repeat_interleave(2)（就地重复）结果是 [A, A, B, B, C, C] （每个人原地变出个双胞胎）。
            # 这里为什么要执行 repeat_interleave？
            # 任务分配，如果每个 Token 要交给 2 个专家（top-2），那么每个 Token 必须有 2 份实体。
            # 索引对齐，x 经过重复后变成了：[T1, T1, T2, T2, T3, T3, ...]，而我们的 flat_topk_idx（拉平后的专家索引）也是 [专A, 专B, 专C, 专D, ...] 这样第 0 个的 T1 对应的就是 专A，第 1 个的 T1 对应的就是 专B。
            # 如果没有这一步，我们的数据量（Token 数）就对不上专家索引的数量（Token 数 $\times$ k），程序就会因为维度不匹配直接报错。
            y = torch.empty_like(x, dtype=x.dtype)
            # 创建一个与输入张量 x 形状、设备（CPU/GPU）完全一致的“空容器”，用来存放专家们计算出来的结果。
            for i, expert in enumerate(self.experts):
                # 遍历每一个专家。i 是专家的编号（比如 0 到 7），expert 是对应的那个前馈网络（FeedForward）层。
                expert_out = expert(x[flat_topk_idx == i])
                # 这是一个布尔索引（Boolean Indexing）操作。
                # 假设 flat_topk_idx 是 [0, 2, 0, 1, 2, 0]（表示 6 个任务对应的专家编号）。当循环运行到 i = 0 时 flat_topk_idx == 0 会得到 [True, False, True, False, False, True]。
                # 这个布尔列表就像一张名单，告诉程序——第 0、2、5 个位置的任务归专家 0 管。
                # x 为 [T1, T1, T2, T2, T3, T3, ...]。
                # flat_topk_idx 为 [专A, 专B, 专C, 专D, ...]。
                # x[flat_topk_idx == i] 会根据上面的名单，从巨大的 x（包含所有 Token 副本）中，只把那 3 个 True 对应的数据行抽出来。
                # 因此就是 T1、T2、T3 去 expert 0。专家 $i$ 只需要处理属于它的那部分数据。
                # 现在，专家 $i$ 拿到的不再是全量数据，而是一个变小了的张量。那么 expert(x[...]) 这一步，专家 0 的全连接层就只对这 125 条数据进行矩阵乘法。这就是 MoE 快的秘密——每个专家只负担总工作量的一小部分（约为 $1/N$）。
                
                # 现在有一个问题是，这似乎并没有让 MoE 比 Dense 更快，因为每个 token 还是要过一个 FFN。
                # 但其实我们要知道，一般情况同等规模的参数下，Dense 里面的 FFN 的大小和 MoE 里面的 FFN 的大小是不一样大的。
                # 如果我们希望 MoE 的总参数量（所有专家加起来）等于一个 Dense 模型，那么每个专家必须比 Dense FFN 小得多。
                # 假设 dense 模型的参数权重是 (4096, 16384)，那么 8 专家的 MoE 的 FFN 参数权重则是 (4096, 2048)。

                # 因此，从这个角度来看，MoE 模型的推理速度是要显著高于 Dense 模型的。对于同一个 Token 来说，在 FFN 层，MoE 前向传播激活的参数只有 Dense 前向传播的 1/n。
                # 虽然我们看到的这段代码用了 for 循环（这在单卡上确实不快），但在大规模分布式训练中，不同的专家会被分发到不同的 GPU 上（例如 GroupedGemm）。
                # 比如 GPU 0 只算专家 1，GPU 1 只算专家 2。
                # 通过 x[flat_topk_idx == i] 过滤后的数据被发送到对应的 GPU。由于每个专家拿到的数据变少了，单卡的计算延迟降低了，大家同时前向传播，整体吞吐量（Throughput）大幅提升。
                
                # 当然，需要明确的是，目前大模型的主流做法，每个专家的大小其实和 Dense FFN 是一样大的。
                # Dense 模型 (如 Mistral 7B)：FFN 维度是 14336。
                # MoE 模型 (如 Mixtral 8x7B)：它有 8 个专家，每个专家的 FFN 维度依然是 14336。
                # 从计算成本角度看，虽然总参数量达到了 47B，但每个 Token 只选 2 个专家计算，所以它的计算量只相当于一个 12B 左右的 Dense 模型。
                # 结果是我们用 12B 的推理速度，获得了一个接近 50B 规模模型的智商。
                if expert_out.shape[0] > 0: 
                    # expert_out 是动态的，因为每个专家被分到的 token 数会变化，假设分到的 token 数是 M 个，那么 expert_out 的形状就是 (M, 4096)，也就是普通 ffn 的输出结果。
                    # expert_out.shape[0] > 0 就代表 M > 0，也就是分到了一个 token。
                    # 在这种情况下，需要把专家计算出的结果 expert_out 填回到大容器 y 的对应位置。
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                    # 这是布尔索引，同样的，y[flat_topk_idx == 0] 就像是一把精准的镊子，只夹住了 y 中的第 0、3、4 行。
                    # 这行代码成立的前提是等号左边的“空位”数量，必须等于等号右边的“数据”数量。
                    # 左边 y[mask]：从全局容器中选出了 $M$ 个位置（即分配给专家 $i$ 的工位）。右边 expert_out：专家 $i$ 刚刚算好的 $M$ 条结果。
                    # 这是一个完美的 1:1 填充。 虽然在 expert_out 内部，这些 Token 是紧挨着排列的（第 0, 1, 2 行），但赋值操作会自动按照 Mask，把它们散开填回到 y 的第 0, 3, 4 行。
                else: 
                    # 如果该专家没有被分到任何的 token。
                    # 在 MoE 模型中，门控系统（Gate）是动态的。在某一次训练迭代中，可能会出现“某个专家一个 Token 也没分到”的情况（即 expert_out 的行数为 0）。
                    # 在分布式训练（Distributed Data Parallel, DDP）下，这会引发致命错误。DDP 的工作原理是它要求所有 GPU 在每一轮结束时，同步所有参数的梯度。
                    # 如果专家 i 没分到 Token，它的参数就没有参与计算，梯度就是 None。当 DDP 试图同步这个专家的梯度时，发现有些 GPU 有梯度，有些没有（None），程序就会直接崩溃或挂起。
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters()) # p.sum() 返回的是一个 tensor。
                    # 这一行代码通过一个无效但合法的运算，给 PyTorch 的 Autograd（自动求导引擎）开了一张“假条”。
                    # sum(p.sum() for p in expert.parameters()) 强行遍历了专家 i 的每一个参数（权重和偏置），并把它们加在一起算出一个总和。这意味着，在计算图里，专家 i 的参数被“激活”了，它们和当前的输出变量 y 建立起了逻辑连接。
                    # 然后我们将这个总和乘以 0。无论参数是多少，结果永远是 0。对 y 的影响 => expert_out + 0 依然等于 expert_out。数据本身没有受到任何污染。
                    # 通过这个操作，PyTorch 会认为专家 i 参与了计算。在反向传播时，会为专家 i 的参数计算梯度。虽然算出来的梯度值是 0（因为乘了 0），但它是一个实实在在的 “0”，而不是 “None”。
                    # 最后，DDP 看到梯度是 0，愉快地完成了同步，训练继续进行，不会报错。

                    # 为什么这里的 expert 的梯度是 None 呢？和计算图有关吗？
                    # 是的，这就是由于 MoE 稀疏选择特性导致的计算图断裂。
                    # 那么，在什么情况下我们可以说某一个参数被纳入到了计算图之中？
                    # 简单来说，当一个参数（Leaf Tensor）参与的任何数学运算，其结果最终被用来计算 Loss（标量）时，我们就说这个参数被纳入了计算图。我们一般可以通过下面的代码来判断某个参数是否还在计算图中，如下所示：
                    # output = expert(input)
                    # print(output.grad_fn)
                    # 如果输出类似 <AddBackward0...>, 说明它在图中。如果输出 None, 说明它是一个孤立的叶子节点，断路了。
                    # 以下几种常见情况，即使我们写了代码，参数也没有进入计算图：
                    # 1. 脱离张量操作。y = x + expert.weight.item()，.item()提取了 Python 数值，丢失了所有梯度跟踪信息。
                    # 2. 原地/非跟踪赋值。with torch.no_grad(): y = expert(x)，明确告诉框架不要记录这一段的操作。
                    # 3. 索引为空。y[False] = expert(x)，赋值操作未发生，专家参数与 y 之间没有建立逻辑指针。
                    # 4. 未参与 Loss 计算。y = expert(x); Loss = other_y.sum()，虽然算了专家，但结果没进 Loss，这条支路在反向传播时会被忽略。

                    # 所以，只要某个参数通过任意一种计算对最后的 loss 有贡献，那么我们就说这个参数被纳入到了计算图之中。
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 在整个 LLM 的 Decoder-only 架构中，y 的 size 和 x 的 size 是始终保持一致的。
            # 给出超参数假设来读懂这一段代码，假设 Token 数量 1 个（为了简化，假设 Batch=1, Seq_len=1），Top-K 2（每个 Token 选 2 个专家），隐藏层维度 (Hidden Size)是 3 维。y 如下：
            # [
            #   [1.0, 1.0, 1.0],  # 专家 A 的输出
            #   [2.0, 2.0, 2.0]   # 专家 B 的输出
            # ]
            # 那么 topk_weight 就是 [[0.7, 0.3]]（因为只有一个 token 选择两个专家）。
            # 这里的 *topk_weight.shape 是在参数解包，将原来的 (1, 2) 解包成两个数据，1 和 2。
            # y 原来是 (2, 3) 现在变成 (1, 2 ,3)
            # topk_weight.unsqueeze(-1) 把权重从 (1, 2) 变成 (1, 2, 1)，这是为了让权重能跟 3 维的专家输出做乘法（广播机制）。
            # 之后的 * 操作就是专家输出 * 对应的权重，如下所示：
            # [
            # [ 
            #     [1.0*0.7, 1.0*0.7, 1.0*0.7], # 专家 A 乘权重
            #     [2.0*0.3, 2.0*0.3, 2.0*0.3]  # 专家 B 乘权重
            # ]
            # ]
            # # 得到：
            # [[ [0.7, 0.7, 0.7], [0.6, 0.6, 0.6] ]]
            # 最后的 .sum(dim=1) 其实就是在“专家”这个维度（dim=1）上求和。
            # 最终 y = [[ 1.3, 1.3, 1.3 ]] (形状回到 (1, 3))
            # 我们要把同一个 Token 的 2 个专家结果合二为一。
            # 如果我们对 dim=0 求和：那是把所有 Token 加在一起（错误）。
            # 如果我们对 dim=2 求和：那是把 128 维特征加在一起（错误）。
            # 我们要对 dim=1 求和：也就是把第 1 维里的那 2 个专家加起来。
            y = y.view(*orig_shape)
            # 把 y 的输出还原为三维的 (Batch_Size, Seq_Len, Hidden_Size)。
        else:
            # 如果是推理状态，会有专门的 moe 实现。
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            # 共享专家融合，MoE 中的“路由专家”是特长生，只处理特定任务；而“共享专家”是全才，每个 Token 都要经过它们。
            for expert in self.shared_experts:
                y = y + expert(identity)
                # 共享专家负责捕获公共的、基础的知识，减少路由专家的负担，同时能提高模型的训练稳定性。
        self.aux_loss = aux_loss
        # 这里的 aux_loss 由 Gate 计算。
        return y

    @torch.no_grad()
    # 对整个函数开启无梯度模式，用于节省显存和加速计算，实际上这里节省的显存就是激活值。
    # 这里这个函数是 MoE 推理阶段的优化，核心逻辑是按专家分组处理。
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 首先需要介绍一下这里的参数：
        # flat_expert_indices 就是一个长度为 $2 \times 3 \times 2 = 12$ 的一维长条。
        # [专家1, 专家3, 专家2, 专家1, 专家4, 专家2, 专家3, 专家4, 专家1, 专家2, 专家3, 专家2]
        # flat_expert_weights 就是 topk_weight，只不过做了 shape 变形，如下所示：
        # [
        #  [0.7], # 对应 Token 0, 专家 1
        #  [0.3], # 对应 Token 0, 专家 2
        #  [0.8], # 对应 Token 1, 专家 0
        #  [0.2], # 对应 Token 1, 专家 1
        #  [0.6], # 对应 Token 2, 专家 2
        #  [0.4]  # 对应 Token 2, 专家 0
        # ]
        expert_cache = torch.zeros_like(x)
        # 创建一个和输入 x 形状一样的全零张量，用来累计所有专家的输出结果。
        # 在 LLM 中输出 y 和输入 x 的形状是完全一样的。
        idxs = flat_expert_indices.argsort()
        # argsort() 会把原本乱序分配给专家的任务，按照专家编号排好队。
        # 假设 flat_expert_indices 如下：
        # 索引位置：      0  1  2  3  4  5
        # 选中的专家号： [1, 2, 0, 1, 2, 0]
        # 那么重排后就为 idxs = [2, 5, 0, 3, 1, 4]。
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # bincount 用来统计每个专家出现的次数，上面的例子就是 [2,2,2]。
        # .cpu().numpy() 把结果从 GPU 搬到 CPU 并转成 NumPy 数组，因为在 Python 的 for 循环里遍历 NumPy 数组比遍历 GPU Tensor 效率更高，也更方便。
        # .cumsum(0) 的含义是计算前缀和，它会把前面的次数累加起来，这里就是 [2,4,6]。
        # 所以这里的 tokens_per_expert 就是 2 4 6，这会和 idxs 联动起来，idxs: [2, 5,  0, 3,  1, 4] (前两个是专家 0 的，中间两个是专家 1 的，最后两个是专家 2 的)。
        token_idxs = idxs // self.config.num_experts_per_tok
        # 本来 flat_expert_indices 是按照 token 顺序标注好了专家下标。
        # idxs 则打乱了 token 顺序，直接按照专家下标分了组。
        # 但其实我们知道原始下标和 idxs 下标的映射关系，因此可以直接做映射：
        # idxs = [2, 5, 0, 3, 1, 4] 的意思是原来的 2 5 index token 对应的是专家 0。
        # 这里的 token_idxs = idxs // self.config.num_experts_per_tok 本质上就是从 idxs = [2, 5, 0, 3, 1, 4] 还原出原始 token 下标。
        # token_idxs = [2//2, 5//2, 0//2, 3//2, 1//2, 4//2]
        # 结果如下 token_idxs = [1, 2, 0, 1, 0, 2]
        # 这意味着，第 1 个和第 2 个 token 对应的是 0 号专家。


        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            # len(tokens_per_expert) 就是专家的个数，i 则是专家编号。
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # start_idx 就是找到专家[i]负责的 idxs 的起始。
            # end_idx 就是找到专家[i]负责的 idxs 的终止。
            if start_idx == end_idx:
                # 如果这个专家一个 Token 也没选它（start == end），直接跳过。
                continue
            expert = self.experts[i]
            # 找到专家。
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 拿到原始 Token 的索引（比如 [1, 2]）
            expert_tokens = x[exp_token_idx]
            # 根据索引从 x 中抠出特征向量。
            # 这里的 expert_tokens 就是要交给 expert[i] 进行推理的 tokens。
            # 注意这里的 token 是没有经过复制的 token。
            expert_out = expert(expert_tokens).to(expert_cache.dtype) # 把打包好的 Token 丢给专家，得到输出。
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            # idxs：Gate 产生的逐 token 结果，按照专家编号从小到大排列的 list，idxs[i] 就是 double token 的 index。
            # 如果 idxs[start_idx:end_idx] 是 [2,5]，那么 flat_expert_weights[[2, 5]] 就能从 flat_expert_weights 中把属于专家 0 的那两个权重抠出来。
            # * expert_out 形状是 (当前批次Token数, Hidden_Size)。
            # * weights 形状是 (当前批次Token数, 1)。
            # * mul_：这是 multiply 的原地版本（In-place）。
            # 权重切片的第 2 维是 1，而 expert_out 的第 2 维是 Hidden_Size（比如 128）。PyTorch 会自动把这个 1 扩展成 128。
            # 因此上述代码的本质就是将对应专家处理的输出 * Gate 输出的专家权重。
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
            # expert_cache 就是最终存放输出的地方。
            # target.scatter_add_(dim, index, src) 的用法如下：
            # 1. target：最终要汇总的大表（收纳盒）。
            # 2. dim：沿着哪个维度分发（通常 0 代表行，1 代表列）。
            # 3. index：地址清单（标签），形状必须和 src 一致。
            # 4. src：我们要分发的数据源（零件）。
            # 因此本质上就等效于 target[index, :] += srcp[index, :]。

            # exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]) 就是 index，他的用法是：
            # token_idxs = [1, 2, 0, 1, 0, 2]
            # exp_token_idx = token_idxs[start_idx:end_idx]
            # 因此 exp_token_idx 就是 [1, 2]，这里的 exp_token_idx 是原始的输入 token 的下标。
            # exp_token_idx.view(-1, 1) 就是 [[1], [2]]
            # 然后来一个 .repeat(1, x.shape[-1])，x.shape[-1] 就是 hidden_size。
            # 那么 index 就是 [[1, 1, 1], [3, 3, 3]]。
            
            # 注意，在 scatter_add_ 中，index 和 src 的 size 必须一致。
            # 他的索引流程如下：
            # 对于 src 中的每一个元素 src[i][j]：
            # - 它在 index 中对应的位置也是 [i][j]。
            # - 它在 index[i][j] 处读到一个数值，假设这个值是 k。
            # 那么，src[i][j] 就会被加到 target[k][j] 上。
            # 数学表达：
            # 如果 dim=0：target[index[i][j]][j] += src[i][j]
            # 如果 dim=1：target[i][index[i][j]] += src[i][j]

        return expert_cache
        # 在这里总结下训练时 MoE 逻辑和推理时 MoE 逻辑的区别：
        # 1. 训练时追求吞吐。GPU 喜欢**“整块、连续、死板”**的任务。训练代码通过 repeat_interleave 把复杂的分发逻辑变成了简单的矩阵切片。虽然多花了一倍显存，但换取了极高的算力利用率（Tensor Cores 跑得满），且方便自动微分（Autograd）追踪梯度。
        # * 稠密化（Dense）：通过 repeat_interleave 将输入强行扩充为规则的长方形矩阵。
        # * 全量并行：不管专家被分配了多少 Token，都以统一的形状送入循环。
        # * 空间换效率：显存占用翻倍（$Batch \times TopK$），但内存访问是连续的。
        # 2. 推理时追求延迟和显存。推理通常是逐 Token 或小 Batch 进行的，此时显存带宽和容量是瓶颈。moe_infer 通过 scatter_add_ 实现了**“原地更新”**，避免了数据翻倍。虽然逻辑复杂（有排序、有索引），但在不计算梯度的情况下，这种灵活性能让显存占用降到最低。
        # * 稀疏化（Sparse）：保持输入大小不变，通过索引（Index）在不同专家间“跳跃”处理。
        # * 分组聚合：先用 argsort 给 Token 按专家编号排队，然后一拨一拨地处理。
        # * 时间换空间：显存占用极低，但非连续的内存访问（Scatter/Gather）开销较大。
    

# 单个 Transformer Decoder Block
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        # 从配置中读取多头注意力的头数
        self.hidden_size = config.hidden_size
        # 隐藏层维度（例如 512 或 1024）
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 计算每个注意力头的维度（总维度 / 头数）
        self.self_attn = Attention(config)
        # 实例化注意力层（模型用来理解上下文的核心组件）

        self.layer_id = layer_id
        # 保存这一层在整个模型中的序号（比如第 5 层）。

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 实例化第一个归一化层（在进入注意力机制前使用），RMSNorm 是一种比 LayerNorm 更高效的变体
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 实例化第二个归一化层（在进入全连接层前使用）
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        # 实例化全连接层（MLP）。如果配置里启用了 MoE（混合专家模型），就用 MOEFeedForward，否则用普通 FeedForward。
        # 如果采用了 FeedForward，那么就是 Dense 模型，例如 GPT-3、Llama-2。
        # 如果模型中引入了 MOEFeedForward，它就变成了 MoE（混合专家）模型。

        # 选 Dense：如果我们追求部署极其简单、显存资源有限，或者模型规模在 7B 以下。在小参数量级下，MoE 的路由开销和训练复杂度可能并不划算。而且 Dense 模型的训练稳定性比较高，且由于参数总量通常比 MoE 小，对显存总量的要求更低。

        # 选 MoE：如果我们想做 SOTA（业界领先） 效果，且有足够的存储资源。MoE 允许我们在保持 10B 级别推理速度的同时，拥有 50B 甚至 100B 级别的知识储备（Knowledge Capacity）。此外，不同的专家可以专注于不同的领域（如逻辑推理、创意写作、代码）。这种分工理论上能比同等计算量的 Dense 模型达到更高的上限。

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        # 将输入的 hidden_states（通常称为“隐藏状态”或“特征向量”）备份一份。
        # 为了实现残差连接（Residual Connection）。在经过复杂的注意力计算后，我们会把原始输入加回来，这有助于缓解深层网络中的梯度消失问题，让模型更容易训练。
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # 先对输入进行归一化（Layer Norm）。因为是 Pre-Norm 结构，归一化发生在计算注意力之前，这能让模型训练更稳定。
        # self.self_attn(...)：执行自注意力机制（Self-Attention）。
        # - position_embeddings：注入位置信息（如 RoPE 旋转位置编码）。
        # - past_key_value & use_cache：用于推理加速的 KV Cache 机制。
        # - attention_mask：确保模型不会“偷看”未来的 Token。
        # self.self_attn() 的返回值是处理后的特征 hidden_states 以及更新后的 present_key_value（用于下一轮推理的缓存）。
        hidden_states += residual
        # 添加残差。
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        # self.mlp(self.post_attention_layernorm(...)) 先对上一步的结果做第二次归一化（post_attention_layernorm），和 Attention 一样都是 Pre-Norm。
        # 送入 MLP（多层感知机）。MLP 负责对每个位置的特征进行非线性变换，提取更深层次的信息。
        # hidden_states + ... 代表再次应用残差连接，将 MLP 的输出与进入 MLP 之前的 hidden_states 相加。
        return hidden_states, present_key_value
        # hidden_states 是本层处理完的特征，将作为下一层 Transformer Block 的输入。
        # present_key_value 是当前层的 KV 缓存，返回给模型顶层以便在生成下一个 Token 时重复使用。

# 堆叠所有块，生成隐层输出
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 定义 Dropout 层。在训练过程中随机丢弃一部分神经元，防止模型过拟合。
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 使用 ModuleList 构建层级堆叠。这里会循环创建 num_hidden_layers 个 MiniMindBlock（通常包含自注意力机制和全连接层）。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 这里的 RMSNorm 也是一个 nn.Module。
        # 定义最终归一化层。这里使用的是 RMSNorm（均方根归一化），这在现代 Transformer 模型（如 Llama）中非常常见，用于稳定最后一层的输出。

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        # 调用函数预先计算 RoPE（旋转位置嵌入） 所需的正弦（sin）和余弦（cos）频率矩阵。
        # dim: 每个注意力头的维度。
        # end: 最大序列长度（max_position_embeddings）。
        # rope_theta: 控制旋转频率的底数（通常是一个很大的数，如 10000）。

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # 将计算好的频率矩阵注册为模型的 Buffer。为什么要做这样做？
        # 非参数化: Buffer 里的数据不属于模型参数，不会计算梯度，也就不会被更新。
        # 设备同步: 它们会随模型一起移动到 GPU 或 CPU，无需手动搬运。
        # persistent=False: 表示这些缓存不需要保存在模型的 state_dict（权重文件）中，因为它们每次启动时都可以重新计算出来。

        # freqs_cos 和 freqs_sin 的维度是 (max_seq_len, head_dim)。
        # max_seq_len 是模型支持的最大序列长度（例如 2048 或 32768）。
        # head_dim 是每个注意力头的维度（注意：不是整个模型的 hidden_size，而是 hidden_size / num_attention_heads）。

        # 假设 Head_Dim = 128：
        # freqs_cos[10, 0]：代表第 10 个词在第 0 个维度上的旋转余弦值（变化最快）。
        # freqs_cos[10, 63]：代表第 10 个词在第 63 个维度（前半截的最后一位）上的旋转余弦值（变化最慢）。
        # freqs_cos[10, 64]：数值等于 freqs_cos[10, 0]（因为是 cat 出来的副本）。

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                # input_ids: 词向量索引，形状通常为 [batch_size, seq_length]。
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                # past_key_values: 用于推理加速的 KV Cache（缓存的键值对）。
                # use_cache: 是否返回当前的 KV 缓存以供下一次推理使用。
                **kwargs):
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, 'layers'): 
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 确保 past_key_values 是一个可迭代的列表。如果传入的是空或旧格式，则初始化为每层对应的 None 列表
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # 计算当前输入的起始位置。如果是增量推理（生成模式），start_pos 就是之前生成的 Token 数量；如果是首次计算，则为 0。
        # 这里的 past_key_values 如下所示：
        # past_key_values = [
        # --- 第一层 (Layer 0) ---
        #     (
        #         torch.randn(1, 4, 3, 8), # Key 张量: [Batch, Heads, Seq, Dim]
        #         torch.randn(1, 4, 3, 8)  # Value 张量: [Batch, Heads, Seq, Dim]
        #     ),
        #     # --- 第二层 (Layer 1) ---
        #     (
        #         torch.randn(1, 4, 3, 8), # Key 张量
        #         torch.randn(1, 4, 3, 8)  # Value 张量
        #     )
        # ]

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 将整数形式的 input_ids 映射为稠密的向量（Embedding），并应用 Dropout（如果配置了的话）来防止过拟合。
        # 哪些层会调用 self.dropout？
        # 在一个完整的 Transformer 架构中，Dropout 通常会出现在以下三个关键位置：
        # 1. Embedding Dropout。在词向量叠加位置编码后执行（即我们代码中的位置），防止模型过度依赖某些特定的词维度。
        # 2. Attention Dropout。在计算 Attention Score（$Softmax$ 之后）执行，随机让某些注意力权重失效，防止模型只关注局部特征。
        # 3. Residual Dropout。在每个子层（Attention 或 MLP）的输出加回残差连接之前执行，这是最常用的正则化手段。
        # 现代大模型（如 Llama 3）为了训练稳定性，往往会减少甚至移除这些 Dropout，转而依赖更大的数据量和更强的正则化手段（如权重衰减）。

        # 两者都试图解决同一个问题：防止模型对训练数据中的特定路径产生过度依赖。
        # Weight Decay 说：“我不准任何一个权重 $w$ 变得太大。如果某个 $w$ 太大，说明模型在强行拟合某个特征，这很危险。”
        # Dropout 说：“我不准模型依赖任何一组特定的神经元。我随机关掉一些，强迫剩下的神经元也能把活干好。”
        # 带 Dropout 的神经网络，在数学上等价于一种特殊的贝叶斯近似（高斯过程），而这种近似包含了一个隐性的正则化项（类似于 L2 正则化）。

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        # 这两行代码是在为模型准备 RoPE（Rotary Positional Embedding，旋转位置编码） 所需的旋转矩阵参数。
        # self.freqs_cos / self.freqs_sin 是模型初始化时预先计算好的两张大表。它们存储了不同位置（从 0 到最大序列长度）对应的正弦（Sin）和余弦（Cos）频率值。
        # start_pos : start_pos + seq_length 是切片操作。
        # 如果是预填充阶段（Prefill）：一次性输入 10 个词，start_pos=0，seq_length=10，它会取出前 10 组旋转参数。
        # 如果是生成阶段（Decoding）：每次只输入 1 个新词。如果之前已经生成了 100 个词，那么 start_pos=100，seq_length=1，它只取出第 101 个位置对应的旋转参数。

        # 什么是预填充阶段？
        # LLM 是基于 Attention 机制的。要生成第 $N+1$ 个字，模型必须知道前 $N$ 个字的所有信息。计算用户输入的 Prompt 中所有 Token 的 Internal States，生成 KV Cache。模型会将 Prompt 中每个词的 Key 和 Value 存起来，这样后续生成新词时，就不需要重新计算旧词了。

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
            # layers 就是 MiniMindBlock。
            # past_key_values 是每一层的 KV Cache。
            # 为什么每一层的 KV Cache 会不一样呢？
            # KV Cache 存储的确实是乘积，但它是「输入 $X$」与「$W_K, W_V$ 权重」的乘积，而不是「输出 $X$」。
            # 本质上 KV Cache 不一样是因为每一层都有独立的 QKV 权重矩阵。
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                # 在标准的模型训练阶段，确实不需要传入 past_key_values。
                # 这里是为推理阶段做的兼容。
                use_cache=use_cache,
                # 训练阶段，use_cache 是 False。
                attention_mask=attention_mask
            )
            presents.append(present)
            # 在推理（Inference）阶段，present 就是在 past_key_value 的基础上，“追加”了当前这个时间步（Current Token）新产生的 $K$ 和 $V$。
            # 在训练阶段，此处的 KVCache 是没有意义的。
            # 这里就是在存储 KV Cache。

            # 下面这里简单去介绍一下 GQA。
            # 我们已经理解了 KV Cache 是存储每一层的 $K$ 和 $V$ 矩阵，那么理解 GQA (Grouped-Query Attention) 就非常简单了：GQA 的本质就是为了给 KV Cache “瘦身”。
            # 首先我们回顾一下 past_key_value 长什么样子，如下所示：
            # past_key_values = [
            # --- 第一层 (Layer 0) ---
            #     (
            #         torch.randn(1, 4, 3, 8), # Key 张量: [Batch, Heads, Seq, Dim]
            #         torch.randn(1, 4, 3, 8)  # Value 张量: [Batch, Heads, Seq, Dim]
            #     ),
            #     # --- 第二层 (Layer 1) ---
            #     (
            #         torch.randn(1, 4, 3, 8), # Key 张量
            #         torch.randn(1, 4, 3, 8)  # Value 张量
            #     )
            # ]
            # 在 LLM 推理中，显存瓶颈通常不在模型权重，而在不断增长的 KV Cache。GQA 正是目前 Llama 3、Mistral、DeepSeek 等主流模型采用的最优解。
            # 传统的 MHA (Multi-Head Attention) 中每个 Query Head ($Q$) 都有自己专属的一个 Key Head ($K$) 和 Value Head ($V$)。
            # KV Cache 体积 == $2 \times \text{层数} \times \text{头数} \times \text{序列长度} \times \text{每个头的维度}$。
            # 问题是随着生成长度增加，KV Cache 会迅速吃光显存，导致 Batch Size 提不上去，推理变慢。
            # 从底层实现来看，GQA 本质就是在减少 W K 和 W V 这两个权重矩阵的参数量。之前是 (d_model, n * d_head)，GQA 变成了 (d_model, g * d_head)，MQA 变成了 (d_model, 1 * d_head)。权重矩阵的形状改变是 GQA/MQA 的物质基础。
            # 采用了 GQA 之后，上面的 KV Cache 示例变成了如下所示：
            # 假设 GQA 分组配置：num_query_heads=4, num_kv_heads=2 (即 2 个 Q 共享 1 个 KV)
            # past_key_values = [
            #     # --- 第一层 (Layer 0) ---
            #     (
            #         # Key 张量: [Batch, KV_Heads, Seq, Dim] 
            #         # 注意：Heads 从 4 变成了 2
            #         torch.randn(1, 2, 3, 8), 
            #         # Value 张量: [Batch, KV_Heads, Seq, Dim]
            #         torch.randn(1, 2, 3, 8)  
            #     ),
            #     # --- 第二层 (Layer 1) ---
            #     (
            #         # 同理，KV 的头数被压缩了
            #         torch.randn(1, 2, 3, 8), 
            #         torch.randn(1, 2, 3, 8)
            #     )
            # ]

        hidden_states = self.norm(hidden_states)
        # 在标准的 Transformer 架构（尤其是 Pre-norm 架构，如 Llama、GPT）中，每一层的输入都会先经过 Norm，但最后一层的输出在送入最后的线性层（LM Head）预测单词之前，必须再经过一次总的归一化。
        # 将经过了数十层叠加、数值可能已经变得很大或分布偏移的 hidden_states 重新拉回到一个稳定的均值和方差。这个 norm 通常是 RMSNorm。完成这一步后，hidden_states 就可以直接用来计算下一个词的概率了。

        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        # 这一行是针对 MoE（Mixture of Experts，混合专家模型） 的特殊逻辑，非常关键。
        # 在 MoE 架构（如 DeepSeek, Mixtral）中，每一层都有多个“专家”（MLP）。模型会自动学习将 Token 交给最合适的专家处理。
        # 问题是如果不加干预，模型往往会产生“胜者全拿”效应——即只训练极少数几个表现好的专家，而其他专家处于“失业”状态。这会导致模型参数浪费，且容易过拟合。
        # 为此我们引入了 Auxiliary Loss（辅助损失），强制模型在训练时尽可能均匀地分配任务给所有专家。
        # 下面来详细拆解下代码实现：
        # [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)]
        # 它遍历模型的所有层，只对那些 MLP 层确实是 MOEFeedForward（即 MoE 结构）的层进行操作，从每一层取出该层计算出的 aux_loss（这个值通常在层的前向传播中根据 Token 的分发策略自动算出）。
        # sum(..., hidden_states.new_zeros(1).squeeze())
        # 将所有层的辅助损失相加，得到整个模型的总辅助损失。
        # hidden_states.new_zeros(1).squeeze()：这是一个编程技巧。它创建了一个初始值为 0 的标量张量，其设备（CPU/GPU）和数据类型（FP16/BF16）自动与 hidden_states 保持一致，确保加法运算不会报错。
        # hidden_states.new_zeros(1) 含义是请参照 hidden_states 这个张量，帮我创建一个形状为 (1,) 的全 0 张量。
        # new_zeros(1) 产生的是一个一维向量，形状是 [1]。.squeeze() 会把所有长度为 1 的维度删掉，把它变成一个标量（Scalar）
        return hidden_states, presents, aux_loss
        # 最后，函数返回了三个核心产物：
        # hidden_states：最终的特征表示（用于预测单词）。
        # presents：更新后的全量 KV Cache（用于下一轮推理）。
        # aux_loss：MoE 系统的“公平性”指标。


# 这里是核心部分，因果语言模型头部，用于生成。
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    # 继承自 PreTrainedModel（处理模型加载、保存、权重初始化）和 GenerationMixin（赋予模型 .generate() 方法进行文本生成的能力）。

    # PreTrainedModel 可以让我们不去写大量的样本代码，包括了：
    # 序列化与加载 (save_pretrained / from_pretrained)。继承后，我们可以直接通过 model.save_pretrained("./path") 保存模型，或者用 MiniMindForCausalLM.from_pretrained("repo_id") 从云端下载。它会自动处理权重转换、分片加载和版本校验。
    # 权重初始化。它定义了 init_weights() 的标准流程。当我们新建模型时，它能确保所有的 Linear 层、Embedding 层按照科学的高斯分布或 Xavier 分布初始化，而不是随机乱序。
    # 设备管理。让我们能轻松使用 .to("cuda")、.half()（半精度）或配合 accelerate 库进行分布式训练。

    # GenerationMixin。forward 函数只负责计算一次前向传播，并不负责推理生成。
    # 自动推理循环。文本生成是一个自回归过程（预测词 A -> 把 A 加入输入 -> 预测词 B）。继承 GenerationMixin 后，我们就拥有了强大的 .generate() 方法。
    # 内置高级搜索算法。我们不需要自己写代码，就能直接使用Beam Search（束搜索）、Top-K / Top-P Sampling（核采样）、Temperature（温度调节）、阻止重复（Repetition Penalty）、KV Cache 管理。
    # 它会自动处理 past_key_values 的传递，让生成速度提升数倍。
    config_class = MiniMindConfig
    # 在 Hugging Face 的框架下，它是必须的。
    # config_class：指定该模型使用的配置类，包含隐藏层维度、层数等超参数。
    # 当我们调用 from_pretrained 时，程序会先读取 config.json。通过这个声明，程序才知道该用 MiniMindConfig 这个类去解析这个 JSON 文件。

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # 实例化模型的核心体。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 线性层，将隐藏状态映射回词表大小（Vocab Size），用于预测下一个词。
        self.model.embed_tokens.weight = self.lm_head.weight
        # 权重共享，最后一行将 Embedding 层的权重与输出层的权重设为同一个。这是一种常用的正则化手段，可以大幅减少模型参数量并提升效果。

    # 这里的 forward 的函数即是训练代码又是推理代码，这里的 forward 参数在调用 model.generate() 是会被传入的。
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                # past_key_values = [
                # --- 第一层 (Layer 0) ---
                #     (
                #         torch.randn(1, 4, 3, 8), # Key 张量: [Batch, Heads, Seq, Dim]
                #         torch.randn(1, 4, 3, 8)  # Value 张量: [Batch, Heads, Seq, Dim]
                #     ),
                #     # --- 第二层 (Layer 1) ---
                #     (
                #         torch.randn(1, 4, 3, 8), # Key 张量
                #         torch.randn(1, 4, 3, 8)  # Value 张量
                #     )
                # ]
                # 当我们生成第 4 个词（比如“他”）时，我们会把这个结构传给模型。模型计算完后，会返回一个新的 past_key_values。新的数据结构依然是 List 包含 Tuple，但张量的形状会发生变化：
                # 输入时：形状是 [1, 4, 3, 8]
                # 输出时：形状变成了 [1, 4, 4, 8] （第 3 维从 3 变成了 4，因为多记了一个词）。
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        # input_ids，输入的 Token 序列。
        # attention_mask (注意力掩码)，用来告诉模型哪些 Token 是有效的，哪些是填充（Padding）出来的。
        # labels (标签/目标)，torch.LongTensor，形状通常与 input_ids 相同。训练时专用。它包含了模型应该预测出的正确 Token。
        # past_key_values (键值缓存 / KV Cache)，保存了之前推理步中每一层的 Key (K) 和 Value (V) 向量。
        # use_cache (是否使用缓存)，一个开关，告诉模型是否需要返回 past_key_values。训练时为 True，推理时为 False。
        # logits_to_keep (保留 Logits 的数量)，因为有一些 logits 是没有意义的。

        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # hidden_states 是语义向量。
        # past_key_values 是更新后的 KV Cache。
        # aux_loss (辅助损失)，这通常出现在 MoE (Mixture of Experts，混合专家模型) 中。MoE 模型里有多个“专家”，如果大家都去挤同一个专家，模型效率会变低。aux_loss 就是用来平衡各专家负载的“罚款”。在 forward 的最后，这个 aux_loss 会加到总的 loss 里一起参与反向传播。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 根据 logits_to_keep 的值，确定我们要从模型提取出的深层特征（hidden_states）中截取哪一部分。
        # 参数含义：slice(start, stop)。-logits_to_keep：从序列的末尾倒着数。None：表示直到序列的最后。
        # 如果传入的是 tensor，那就是对不同的样本采取不同的抽取方式，这经常在训练模式中出现。
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # 为什么要去裁剪呢？简单来说，裁剪（Slicing）是为了“只算有用的，省下没用的”。
        # 从推理的角度来看，由于使用了 KV Cache（past_key_values），模型每一轮 forward 实际上只需要关注当前最新输入的那一个词。
        # 如果不裁剪 lm_head 会计算这 2000 个词每一个词对应的 Logits（预测下一个词的分数）。但实际上，前 1999 个词对应的预测结果我们根本不需要，因为它们已经是过去式了。
        # 如果裁剪 (logits_to_keep=1)，通过 slice(-1, None)，模型只取最后一个特征向量喂给 lm_head。计算量骤降，速度提升，且节省了巨大的中间显存。

        # ~~从训练的角度来看，在训练时，虽然我们要处理整个序列，但并不是所有位置的输出都对“学习”有贡献。~~
        # ~~指令微调（SFT），通常我们会输入一个 Prompt（问题）和一个 Answer（回答）。在 SFT 中，我们通常只计算 Answer 部分的 Loss。因为问题是给定的，模型不需要学习如何生成问题。~~
        # ~~假设序列总长 4000，其中问题 3500 词，回答 500 词。如果不裁剪，模型会生成一个 [Batch, 4000, 128000] 的巨型 Logits 张量。这在 FP32 精度下大约占用 2GB/样本。如果 Batch Size 是 8，光这一个变量就要 16GB 显存，直接导致显卡崩溃（OOM）。如果裁剪 (logits_to_keep=500)，通过 slice(-500, None)，模型只计算最后 500 个回答词的 Logits。~~

        # ~~上述的“训练中的logits_to_keep”理解是错误的，一般来讲，在训练时一般不传入这个参数，只有推理会传入。训练（特别是像你进行的 SFT）是一个“全员考核”的过程。模型输入的每一个 Token（除去 Prompt 部分）都需要和 labels 进行对比。为了算 Loss，我们需要序列中每一个位置的预测结果（Logits）。~~

        # ⭐️⭐️⭐️⭐️⭐️ 在 SFT 的阶段，在计算 loss 时，只考虑了 Response，不考虑 User Prompt，相应实现可以参考 Im_dataset.py 的 119 行，最后生成的 labels 是 [-100, ..., -100] [-100, ..., -100] [我是人工智能] [Assistant结束符] [-100, ...]。
        # ⭐️⭐️⭐️⭐️⭐️ 即使在 Base 模型上进行指令遵循微调，也不应该关注 user prompt 带来的 loss。这是因为 SFT 的核心是关心生成结果的质量，而不是去学习如何生成问题。模型学习的是 P(resp|prompt)，而不是P(prompt+resp|[BOS])。
        # ⭐️⭐️⭐️⭐️⭐️ 这里有一个误解是，不计算 loss != 不学习。虽然主流训练代码都会 mask prompt。
        # ⭐️⭐️⭐️⭐️⭐️ 即使我们不参与对 user prompt 的 loss 计算，但是对 Response loss 的计算也会逼着模型学会理解 <im_start> 和 <im_end> 这样的对话格式，“强迫答案正确，你也就学会了理解题目”。
        # ⭐️⭐️⭐️⭐️⭐️ 此外，因为参数 W 是共享的，处理 prompt 时用的 W 和处理 response 时用的 W 是同一个 W。 梯度更新了 W，就等于同时影响了模型处理 prompt 和 response 的方式。
        # ⭐️⭐️⭐️⭐️⭐️ 那"不计算 Loss" 到底少了什么？
        # 对 Prompt 计算 Loss:
        # 模型学习："看到 [请] 之后，应该预测 [解释]"     ← 学习生成 prompt
        # 模型学习："理解 prompt 以便生成好的 response"   ← 学习理解 prompt
        
        # 不对 Prompt 计算 Loss:
        # 模型学习："理解 prompt 以便生成好的 response"   ← 只学习理解 prompt ✅
        # 模型不学："看到 [请] 之后，应该预测 [解释]"     ← 不学生成 prompt ✅

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 进行错位。
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            # ⭐️⭐️⭐️⭐️⭐️ 这里的交叉熵 loss 的计算和 dpo、grpo 中的 logits_to_log_probs 的计算方式是完全一样的，唯一的不同就是 label，grpo 的 label 取自自己的采样。
            # step 1，log_probs = F.log_softmax(logits, dim=-1)
            # step 2，target_logp = log_probs.gather(dim=-1, index=labels)
            # step 3，loss = -target_logp.mean()

            # 计算交叉熵损失。
            # 这是一个非常关键的约定。在数据预处理时，我们会把不需要计算 Loss 的部分（比如 Padding 或者 Prompt 部分）的标签设为 -100。
            # Label 是不是 -100，就是决定模型“学不学习这个词”的唯一信号灯。
            # 当 label 等于 -100 时，不管模型预测了什么（哪怕预测得离谱到天边），Loss 直接记为 0。模型不会因为这个词的预测好坏而产生任何权重更新。
            # 给 label 赋值为 -100 的时机，主要集中在**“数据喂给模型前的最后一刻”**。在实际的大模型开发中，这通常发生在 Dataset（数据集类） 或 Data Collator（数据整理器）中。

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        # CausalLMOutputWithPast 是 Hugging Face transformers 库定义的一个标准数据类（Dataclass）。
        # 如果返回 (loss, logits, ...)，我们必须死记硬背顺序（第 0 位是 Loss，第 1 位是 Logits）。而使用了这个对象后，调用者可以通过属性名直接访问，代码可读性极高。
        # 除了装入 loss、logits、past_key_values 和 hidden_states 以外，还把 MoE 的 aux_loss 也装进去了。
        output.aux_loss = aux_loss
        return output
