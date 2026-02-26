from transformers import PretrainedConfig
# 在深度学习中，一个模型由两部分组成：
# 权重（Weights）： 模型的“记忆”，即那些巨大的参数矩阵。
# 配置（Configuration）： 模型的“结构”，比如它有几层、每层有多宽、词汇表有多大等。

# 它的核心作用：
# - 定义结构： 它规定了模型的超参数（比如 hidden_size 隐藏层维度、num_attention_heads 注意力头数等）。
# - 加载与保存： 它可以从本地文件夹或 Hugging Face Hub 上读取 config.json 文件，也可以把当前的配置保存成 JSON。
# - 模型初始化的桥梁： 当你创建一个模型时，系统会先看这个“说明书”，知道该盖多少层楼、挖多深的基座，然后再把“权重”填进去。

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
            # 输入维度是 $d_{model}$（你的 512）。
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
            # 假设你的模型是在 2048 长度下训练的，现在你硬要它处理 32768 长度，直接跑肯定会“变傻”，因为它没见过那么大的位置编号。
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
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
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
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

# GQA 的键值复制
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

# 多头注意力（支持 FlashAttention）
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

# 简单前馈网络
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

# 混合专家的路由门控
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss

# 混合专家前馈
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

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

        # 选 Dense：如果你追求部署极其简单、显存资源有限，或者模型规模在 7B 以下。在小参数量级下，MoE 的路由开销和训练复杂度可能并不划算。而且 Dense 模型的训练稳定性比较高，且由于参数总量通常比 MoE 小，对显存总量的要求更低。

        # 选 MoE：如果你想做 SOTA（业界领先） 效果，且有足够的存储资源。MoE 允许你在保持 10B 级别推理速度的同时，拥有 50B 甚至 100B 级别的知识储备（Knowledge Capacity）。此外，不同的专家可以专注于不同的领域（如逻辑推理、创意写作、代码）。这种分工理论上能比同等计算量的 Dense 模型达到更高的上限。

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
        # 1. Embedding Dropout。在词向量叠加位置编码后执行（即你代码中的位置），防止模型过度依赖某些特定的词维度。
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
    # 序列化与加载 (save_pretrained / from_pretrained)。继承后，你可以直接通过 model.save_pretrained("./path") 保存模型，或者用 MiniMindForCausalLM.from_pretrained("repo_id") 从云端下载。它会自动处理权重转换、分片加载和版本校验。
    # 权重初始化。它定义了 init_weights() 的标准流程。当你新建模型时，它能确保所有的 Linear 层、Embedding 层按照科学的高斯分布或 Xavier 分布初始化，而不是随机乱序。
    # 设备管理。让你能轻松使用 .to("cuda")、.half()（半精度）或配合 accelerate 库进行分布式训练。

    # GenerationMixin。forward 函数只负责计算一次前向传播，并不负责推理生成。
    # 自动推理循环。文本生成是一个自回归过程（预测词 A -> 把 A 加入输入 -> 预测词 B）。继承 GenerationMixin 后，你就拥有了强大的 .generate() 方法。
    # 内置高级搜索算法。你不需要自己写代码，就能直接使用Beam Search（束搜索）、Top-K / Top-P Sampling（核采样）、Temperature（温度调节）、阻止重复（Repetition Penalty）、KV Cache 管理。
    # 它会自动处理 past_key_values 的传递，让生成速度提升数倍。
    config_class = MiniMindConfig
    # 在 Hugging Face 的框架下，它是必须的。
    # config_class：指定该模型使用的配置类，包含隐藏层维度、层数等超参数。
    # 当你调用 from_pretrained 时，程序会先读取 config.json。通过这个声明，程序才知道该用 MiniMindConfig 这个类去解析这个 JSON 文件。

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # 实例化模型的核心体。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 线性层，将隐藏状态映射回词表大小（Vocab Size），用于预测下一个词。
        self.model.embed_tokens.weight = self.lm_head.weight
        # 权重共享，最后一行将 Embedding 层的权重与输出层的权重设为同一个。这是一种常用的正则化手段，可以大幅减少模型参数量并提升效果。

    # 这里的 forward 的函数即是训练代码又是推理代码。
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
                # 当你生成第 4 个词（比如“他”）时，你会把这个结构传给模型。模型计算完后，会返回一个新的 past_key_values。新的数据结构依然是 List 包含 Tuple，但张量的形状会发生变化：
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
        # 如果不裁剪 lm_head 会计算这 2000 个词每一个词对应的 Logits（预测下一个词的分数）。但实际上，前 1999 个词对应的预测结果你根本不需要，因为它们已经是过去式了。
        # 如果裁剪 (logits_to_keep=1)，通过 slice(-1, None)，模型只取最后一个特征向量喂给 lm_head。计算量骤降，速度提升，且节省了巨大的中间显存。

        # 从训练的角度来看，在训练时，虽然我们要处理整个序列，但并不是所有位置的输出都对“学习”有贡献。
        # 指令微调（SFT），通常我们会输入一个 Prompt（问题）和一个 Answer（回答）。在 SFT 中，我们通常只计算 Answer 部分的 Loss。因为问题是给定的，模型不需要学习如何生成问题。
        # 假设序列总长 4000，其中问题 3500 词，回答 500 词。如果不裁剪，模型会生成一个 [Batch, 4000, 128000] 的巨型 Logits 张量。这在 FP32 精度下大约占用 2GB/样本。如果 Batch Size 是 8，光这一个变量就要 16GB 显存，直接导致显卡崩溃（OOM）。如果裁剪 (logits_to_keep=500)，通过 slice(-500, None)，模型只计算最后 500 个回答词的 Logits。

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 进行错位。
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            # 计算交叉熵损失。
            # 这是一个非常关键的约定。在数据预处理时，我们会把不需要计算 Loss 的部分（比如 Padding 或者 Prompt 部分）的标签设为 -100。
            # Label 是不是 -100，就是决定模型“学不学习这个词”的唯一信号灯。
            # 当 label 等于 -100 时，不管模型预测了什么（哪怕预测得离谱到天边），Loss 直接记为 0。模型不会因为这个词的预测好坏而产生任何权重更新。
            # 给 label 赋值为 -100 的时机，主要集中在**“数据喂给模型前的最后一刻”**。在实际的大模型开发中，这通常发生在 Dataset（数据集类） 或 Data Collator（数据整理器）中。

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        # CausalLMOutputWithPast 是 Hugging Face transformers 库定义的一个标准数据类（Dataclass）。
        # 如果返回 (loss, logits, ...)，你必须死记硬背顺序（第 0 位是 Loss，第 1 位是 Logits）。而使用了这个对象后，调用者可以通过属性名直接访问，代码可读性极高。
        # 除了装入 loss、logits、past_key_values 和 hidden_states 以外，还把 MoE 的 aux_loss 也装进去了。
        output.aux_loss = aux_loss
        return output
