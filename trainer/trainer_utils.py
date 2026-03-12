"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    # 计算模型所有参数的总和，并转换成 M（百万） 为单位。这包括了所有的骨干网络（Attention 等）和所有的专家层。
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    # 总共训练了多少个路由专家（可选专家总数）。
    n_active = getattr(config, 'num_experts_per_tok', 0)
    # 每个 Token 实际上会激活（路由到）多少个专家。
    n_shared = getattr(config, 'n_shared_experts', 0)
    # 共享专家的数量（始终保持激活，不参与路由）。

    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    # 通过字符串匹配 named_parameters()，精准找到第一个路由专家（experts.0）和第一个共享专家（shared_experts.0）的大小。

    base = total - (expert * n_routed) - (shared_expert * n_shared)
    # 计算**非专家层（Base）**的参数量。

    active = base + (expert * n_active) + (shared_expert * n_shared)
    # 计算激活参数量（Active Params）。

    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    # 如果是 MoE 模型（active < total），它会打印两个数字。例如：Model Params: 1000.00M-A250.00M。这意味着模型虽然有 10 亿参数，但实际跑起来只相当于 2.5 亿参数的计算量。
    else: Logger(f'Model Params: {total:.2f}M')
    # 如果是普通 Dense 模型，只打印总参数量。


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    # cosine 学习率计算。
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    # 影响 Python 内置 random 库，比如你代码里用 random.shuffle(list) 打乱数据集，或者 random.random() 生成随机数，都会受此控制。
    np.random.seed(seed)
    # 影响 NumPy 库，LLM 预处理中经常用 NumPy 处理矩阵（比如旋转位置编码的预计算），或者某些数据增强操作。
    torch.manual_seed(seed)
    # 影响所有在 CPU 上进行的 PyTorch 运算，模型权重的初始分布、在 CPU 上进行的 Dropout、以及张量的随机采样。
    torch.cuda.manual_seed(seed)
    # 影响当前显卡（Current GPU）的随机生成。
    torch.cuda.manual_seed_all(seed)
    # 影响所有显卡，在多卡训练（DDP）中极其关键。
    # 这决定了显卡上生成的随机噪声、Dropout 掩码位置、以及 MoE 架构中专家路由的随机抖动。
    torch.backends.cudnn.deterministic = True
    # 默认情况下，为了追求极致速度，卷积等算子会根据硬件尝试多种算法，选最快的那个。但不同的算法可能因为浮点数舍入顺序不同，导致微小的计算差异。
    # 引入该配置后，强制使用确定性算法。虽然会牺牲一点点性能，但保证了输入相同，输出绝对相同。
    torch.backends.cudnn.benchmark = False
    # 防止 cuDNN 在每一步根据输入形状动态调整算法，从而消除因算法切换带来的数值漂移。

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # 最终发布版的权重路径（仅包含模型参数，用于推理）。
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'
    # 续训存档的路径（包含模型、优化器、Epoch 等完整状态）。

    if model is not None:
        # 如果调用时传入了 model 对象，说明是要执行“保存”操作。
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # 如果模型被 DDP 包裹了，取其内部的 .module，否则存的是带 module. 前缀的键，推理时会报错。
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        # 如果用了 torch.compile，需要剥离编译器包装，拿到最原始的 PyTorch 模型。
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        # 非常关键的优化。将权重转为 half (float16) 并移到 CPU。这样保存的文件更小，且不会在保存瞬间因占用显存导致 OOM（显存溢出）。
        ckp_tmp = ckp_path + '.tmp'
        # 先存成临时文件再改名。这能防止保存到一半断电导致原本的存档文件损坏。
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        # wandb_id 记录当前实验在云端的 ID。这样下次续训时，曲线能接在同一个实验下，而不是新开一个。
        if wandb:
            if hasattr(wandb, 'get_run'):
                # 检查 wandb 对象里有没有一个叫 get_run 的函数。
                # 这是一个版本兼容性设计。
                run = wandb.get_run()
                # 执行函数，拿到代表当前这次实验的对象 run。
                wandb_id = getattr(run, 'id', None) if run else None
                # 尝试从 run 对象里读出 id 属性。如果读不到，就返回 None。
            else:
                wandb_id = getattr(wandb, 'id', None)
                # 直接尝试从 wandb 对象本身读取 id 属性。

        resume_data = {
            'model': state_dict, # 权重。
            'optimizer': optimizer.state_dict(),
            # 优化器的动量、步数等状态（续训必须有这个，否则学习率和方向会乱）。
            # 这里面存储的信息包括了：
            # 1. 一阶动量。
            # 2. 二阶动量。
            # 3. step，也就是该参数更新了多少次。
            # 其它的参数包括了：
            # lr (Learning Rate)：当前的学习率。
            # betas：Adam 的平滑系数（默认通常是 (0.9, 0.999)）。
            # eps：数值稳定性的小分母（防止除以 0）。
            # weight_decay：权重衰减系数（L2 正则化）。
            # amsgrad：是否使用该变体。
            # params：一个 ID 列表，对应它所管理的模型参数索引。

            # 整体结构为：
            # {
            #     'state': { ... },  # 这里存的是动量、平方项等（针对每个参数的数据）
            #     'param_groups': [   # 这里存的是超参数（控制训练逻辑的数据）
            #         {
            #             'lr': 0.0001,           # <--- 学习率就在这里！
            #             'weight_decay': 0.01,
            #             'betas': (0.9, 0.999),
            #             'eps': 1e-08,
            #             'amsgrad': False,
            #             'params': [0, 1, 2, ...] # 这一组参数对应的 ID
            #         }
            #     ]
            # }

            # 虽然 optimizer 存了当前的 'lr'，但在代码逻辑中，通常还会配合一个 LRScheduler（比如 CosineAnnealing，用来实现学习率随时间衰减）。
            # 如果你只恢复了 optimizer 的状态，而没有保存和恢复 scheduler 的状态。那么当你重启训练时，scheduler 可能会从第 0 步重新开始计算，导致它强行把 optimizer 里的学习率覆盖成初始的大数值。
            # 但通过手动记录 epoch 和 step，并在加载后重新计算或设置步数，确保了即使不专门存 scheduler 的 state_dict，也能通过步数对齐让学习率回到正确的位置。

            'epoch': epoch,
            'step': step,
            # 当前练到哪了。
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            # 显卡数
            'wandb_id': wandb_id
        }

        # 这段代码让除了模型和优化器这两个“主角”外，允许你把任何其他重要的训练组件
        # （比如梯度缩放器 scaler、学习率调度器 scheduler 等）也顺便打包存进存档里。
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    # 剥离 DDP 前缀。
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    # 剥离 compile 前缀。
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        # state_dict 重复保存了两次，原因何在？
        # 一次是为了给别人看（推理），一次是为了留给自己练（续训）。

        del state_dict, resume_data
        # 手动删除这两个巨大的 Python 变量。
        torch.cuda.empty_cache()
        # 强制 PyTorch 释放显存管理器中缓存的空闲显存，还给操作系统。
        # 在执行 torch.save 时，PyTorch 可能会产生临时的显存分配。保存结束后，如果不手动清理，这些被占用的显存可能不会立即释放，导致紧接着开始的下一轮训练（Forward 过程）因为显存不足而报错。
    else:  # 加载模式

        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            # 看一眼硬盘里有没有那个带 _resume.pth 后缀的“全家桶”存档。
            # 关键点 map_location='cpu'：先把数据全部加载到 内存（RAM） 而不是显存（VRAM）。
            # 为什么：存档可能非常大（包含优化器状态）。如果直接载入 GPU，而你此时正好在初始化其他东西，极易导致显存瞬间炸掉（OOM）。
            saved_ws = ckp_data.get('world_size', 1)
            # 从存档里读出上次训练时一共用了几张显卡（World Size）。
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            # 看看你现在这台机器上一共有几张显卡在跑。
            if saved_ws != current_ws:
                # 跨设备进度的“等效转换”
                # 在大模型训练中，一个 step（步数）代表一次参数更新。
                # 总训练量 = step × 每一步消耗的数据量。
                # 而“每一步消耗的数据量”通常正比于 GPU 的数量

                # 上次（8张卡）：练了 100 步。总共看过了 100 * 8 = 800 份数据。
                # 这次（2张卡）：如果你直接从第 100 步开始，由于每步只看 2 份数据，你接下来的训练节奏就乱了。公式转换：100 * 8 / 2 = 400。
                # 结果：代码会自动把 step 改成 400。这样虽然步数变多了，但保证了模型已经“吃”过的数据量在逻辑上是连续的。
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # 当 tokenizer_path 是 ../model 时，这一行代码做了如下内容 =>
    # 1. 寻找 tokenizer_config.json 和 tokenizer.json。前者记录了分词器的配置信息（比如是否剔除空格、截断长度是多少）。后者是核心词库，里面记录了成千上万个词汇/字符与数字 ID 的对应关系。
    # 2. 由于你使用的是 AutoTokenizer，它会读取配置文件中的 tokenizer_class。如果你的模型是 Llama，它会自动实例化 LlamaTokenizer。如果你的模型是 BERT，它会自动实例化 BertTokenizer。在这里使用了 PreTrainedTokenizerFast。这是一个基于 Rust 核心实现的“快速分词器”。底层的算法实现还是 BPE，Rust 实现会比 Python 实现快很多。
    
    model = MiniMindForCausalLM(lm_config)
    # 基于传入的 lm_config（如层数、维度、头数等配置）创建一个 MiniMind 架构的因果语言模型（Causal LM）。此时模型参数是随机初始化的。
    if from_weight!= 'none':
        # 如果 from_weight 不是 'none'，说明需要加载已有的权重文件，而不是从零（Scratch）开始训练。
        moe_suffix = '_moe' if lm_config.use_moe else ''
        # 检查配置是否使用了 MoE（混合专家模型）。如果是，就在文件名加个 _moe 后缀，以便精准定位对应的权重文件。
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        # 使用 PyTorch 的 load 函数读取权重字典。map_location=device 确保权重直接加载到你指定的显卡（或 CPU）上，避免内存溢出。
        model.load_state_dict(weights, strict=False)
        # 将加载的字典映射到模型参数中。strict=False 允许模型结构和权重字典有微小差异（比如你在微调时修改了部分层），不会直接报错中断。
        # strict=False 就像是一个“消音开关”。它让程序不再因为参数不匹配而报错崩溃，但代价是它可能悄悄掩盖了一些足以毁掉训练的致命问题。
        # 因为 strict=False 使得 key 对不上也能 load。
        
        # 什么时候需要 strict=False？
        # 结构微调 (如 MoE 改版)	strict=False	旧权重里没有专家层的路由信息，需要手动跳过。
        # 新增功能 (如加了 LoRA)	strict=False	只加载 Backbone，忽略/新增特定的 Adapter 权重。

    get_model_params(model, lm_config)
    # 打印一些模型的参数。
     
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    # 计算参数量，model.parameters() 返回的是一个个 Tensor。
    # 这里的 p.numel() 实际上就是计算每个 tensor 中有多少可训练参数。
    return model.to(device), tokenizer
    # 将模型彻底搬运到指定的 device（如显卡）上，并将模型和分词器一并返回。 


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)