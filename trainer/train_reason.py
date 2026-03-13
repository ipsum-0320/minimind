import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, tokenizer, lm_config, start_step=0, wandb=None):
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids
    # 将特定的特殊标记（Special Tokens）从文本格式转换为模型能理解的数字 ID（Input IDs）。

    # 这些标签通常出现在类似 DeepSeek-R1 这种具有“思考过程”的模型中。模型通过这些标签来区分“思考逻辑”和“最终答案”：
    # 1. <think> / </think>: 思考过程的开始和结束。
    # 2. <answer> / </answer>: 最终回答内容的开始和结束。

    # 这几行代码的作用如下：
    # 1. 在计算 Loss 时，你可能只想让模型学习 <answer> 部分，或者对 <think> 部分应用不同的权重。通过获取 start_of_answer_ids，你可以找到答案在序列中的起始位置，从而在 Mask 中标记出来。
    # 2. 在推理时，你可以检查模型生成的 ID 序列中是否包含这些特定的 ID，以判断模型是否按照要求的格式输出。如果模型没输出 end_of_think_ids 就直接输出了回答，你可以判断其逻辑链可能断裂了。
    # 3. 在强化学习（RL）中，如果模型在 </think> 之前写了太长的内容，或者在 <think> 标签内进行了有效的逻辑推导，程序会利用这些 ID 坐标来定位内容，并给予相应的奖励或惩罚。

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    # nn.CrossEntropyLoss 定义了交叉熵损失。
    # 交叉熵损失如下：
    # L = -\log \left( \frac{e^{z_y}}{\sum_{j=1}^{C} e^{z_j}} \right)
    # 其中：
    # z_y：模型预测“正确类别 $y$”的原始得分。
    # \sum e^{z_j}：所有可能类别的得分指数和（用于将得分转为概率）。
    # C：词表的大小（Vocabulary Size）。

    # 如果正确 Token 的概率接近 1，\log(1) = 0，则损失趋近于 0。如果正确 Token 的概率接近 0，-\log(\text{很小的数}) 会变得趋于正无穷，从而产生巨大的惩罚。

    # 在 LLM 中，z_y 是进入 Softmax 之前的值（Logits），而不是之后的值（Probability）。

    # 这里的 reduction='none' 需要特别注意一下，默认情况下 reduction='mean'。
    # 假设你正在训练一个模型，输入数据如下为 Batch Size: 2（一次训练 2 句话）Sequence Length: 4（每句话有 4 个词）。如果为 mean，模型计算出这 2 * 4 = 8 个位置的损失，然后把它们全部加起来除以 8。输出是一个零维张量（标量），例如 tensor(2.3501)。后果：你失去了对单个 Token 损失的控制权。

    # 开启后 (reduction='none')，模型会给出每个位置独立的损失值。
    # 输出是一个形状为 [2, 4] 的张量：
    # tensor([[1.5, 0.2, 3.1, 0.8],  # 第一句话的 4 个 Token 的 Loss
    #         [0.9, 2.5, 0.4, 1.2]]) # 第二句话的 4 个 Token 的 Loss

    # 为什么需要 reduction='none'？
    # 1. 实现 Masking（掩码计算）。在 LLM 训练中，我们通常不希望对 Prompt（提示词） 部分计算 Loss，只希望对 Response（回答） 部分计算 Loss。
    # 2. 区分推理（Thinking）和回答（Answer）。让模型多学习 <answer> 里的内容（给这部分的 Loss 乘个 1.2）；让模型少纠结 <think> 里的琐碎细节（给这部分的 Loss 乘个 0.5）。

    start_time = time.time()
    
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

            # {"conversations": [{"role": "user", "content": "你好，我是第一次使用你们的平台，我该怎么使用你来帮助我解决问题呢？"}, {"role": "assistant", "content": "<think>\n好，用户第一次用这个平台，不知道怎么解决什么问题。我需要的帮助是指导他们如何使用。首先，我应该问他们遇到的问题是什么，或者他们希望了解哪些方面。这能帮助我更好地帮助他们。接下来，我可以鼓励他们继续提问，说明自己如果需要更多帮助时的步骤。最后，邀请他们在后续遇到问题时再来问，这样他们就不会感到害怕，也能得到更详细的解答。\n</think>\n<answer>\n您好！如果您有遇到问题，或者想了解帮助内容，请随时告诉我，我会尽力为您解答。您可以描述问题、分享经验和具体请求，我会根据您的需求帮助您解决问题。如果您需要更多帮助，请随时告诉我！\n</answer>"}]}

        with autocast_ctx:
            res = model(input_ids)
            shift_logits = res.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 错位对齐，在自回归任务中，我们需要用当前位置的词去预测下一个位置的词。

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
            # 这里调用了你之前定义的 reduction='none' 的交叉熵函数。
            # view(-1, ...): 将多维张量展平为二维，以符合 PyTorch 损失函数的要求。
            # .view(shift_labels.size()): 计算完后将 loss 重新塑造成 [Batch, Seq_Len] 的形状，方便后面按位置打掩码。

            loss_mask = (shift_labels != -100).float()
            # 创建一个掩码。正如我们之前讨论的，Prompt 部分通常被设为 -100。这行代码让 Prompt 部分权重为 0，Response 部分权重为 1。
            # 这部分的逻辑可以去看 Im_dataset.py。

            sp_ids = torch.isin(
                # shift_labels 原本的形状是 [Batch_Size, Seq_Len]。为了能快速比对，使用 .view(-1) 将它拉成一根长条（一维向量）。
                shift_labels.view(-1),
                # 将之前提取的四个特殊标记的 ID 列表合并成一个长列表。
                # 假设 <think> 的 ID 是 100，</think> 是 101，<answer> 是 102，</answer> 是 103。这一块会产生一个类似 [100, 101, 102, 103] 的 Tensor，并移动到 GPU（args.device）上。
                torch.tensor(start_of_think_ids + end_of_think_ids
                + start_of_answer_ids + end_of_answer_ids).to(args.device))
            # 寻找“特殊标签”（<think>, </think>, <answer>, </answer>）在当前序列中的位置。torch.isin 会返回一个布尔索引。
            # sp_ids = torch.isin(...)。torch.isin(A, B) 会遍历 A 中的每一个元素，问：“你在 B 里面吗？”
            # 返回值是一个与 shift_labels.view(-1) 长度完全相同的布尔张量（Boolean Tensor）。
            # 如果该位置的 Token ID 是 100/101/102/103 其中的一个，对应位置就是 True。否则，就是 False。

            loss_mask_flat = loss_mask.view(-1)
            # 将之前形状为 [Batch, Seq_Len] 的 loss_mask 拉平为一维向量。
            # 目的是为了配合刚才计算出来的 sp_ids（也是一维的布尔索引），方便进行批量索引赋值。
            loss_mask_sum = loss_mask_flat.sum()
            # 统计当前 Batch 中所有非 -100 的 Token 数量（即所有回答部分的 Token 总数）。
            # 这个数值将作为分母。注意，这里在修改权重之前先求和，意味着最后计算平均 Loss 时，分母依然是原始的 Token 数量，这样加权后的 Loss 会整体变大。
            loss_mask_flat[sp_ids] = 10
            # 利用布尔索引 sp_ids，将所有属于 <think>、</think>、<answer>、</answer> 位置的权重从 1.0 提升到 10.0。
            # 这是在给模型“划重点”。如果模型在这些关键的格式标记上出错，产生的梯度将是普通文字的 10 倍。这能强迫模型完美遵循推理格式。
            # 下面讲一下布尔索引的原理：
            # 1. 当你使用 loss_mask_flat[sp_ids] 时，PyTorch 要求这两个张量的形状必须完全一致。
            # * loss_mask_flat（数据）：[1.0, 1.0, 1.0, 1.0, 1.0]
            # * sp_ids（滤镜）：[False, True, False, False, True]
            # 2. 处理逻辑：PyTorch 会同时遍历这两个张量。只有当 sp_ids 中某个位置的值为 True 时，它才会选出（或修改）loss_mask_flat 中对应位置的那个元素。
            # 3. 布尔索引极其高效，因为它避免了循环计算，是在底层 C++/CUDA 层面并行完成的。

            loss_mask = loss_mask_flat.view(shift_labels.size())
            # 将刚才在一维状态下修改好的 loss_mask_flat（包含 0、1、10 权重的那根“长条”）重新折叠回二维形状。
            # 形状变化从 [Batch * Seq_Len] 回到 [Batch, Seq_Len]。
            logits_loss = (loss * loss_mask).sum() / loss_mask_sum
            # loss * loss_mask 是核心的差异化对待。
            # - Prompt 部分：$Loss \times 0 = 0$（被彻底抹除，不产生梯度）。
            # - 普通回答部分：$Loss \times 1 = Loss$（正常计算）。
            # - 特殊标签部分：$Loss \times 10 = 10 \times Loss$（权重放大 10 倍）。
            # .sum() 是将整个 Batch 中所有位置的加权损失加在一起，得到一个总分。
            # / loss_mask_sum：除以原始有效 Token 数（即 mask 为 1 的数量）。注意：这里没除以加权后的总和，而是除以了数量。这意味着由于特殊标签被加权到了 10，整体的 logits_loss 会比平时略高，这会引导模型更强力地收敛到这些格式标签上。

            # 为什么这会有效呢？
            # 1. 改变了 loss 惩罚力度。对于特殊标签（如 <think>）：由于 Loss 被乘以了 $10$，根据微积分的线性法则，该位置产生的梯度也会直接放大 10 倍。
            # 2. 解决了样本稀疏的问题。在长文本推理中，<think>、</think> 这些标签在成千上万个 Token 中可能只占不到 0.1%。加权 10 倍后，这些稀疏标签的存在感瞬间提升。这能保证模型在处理长序列时，依然对这些“微小但关键”的结构点保持高度警觉。
            
            # Loss 的大小直接决定了梯度（信号）的强弱。某个 token 的 loss 越高，更新时对他的重视就越强。
            # 但是这里的 loss 实际上是对所有的参数权重都生效的，所以即使 Prompt 不产生梯度贡献，但是对 Prompt 产生计算的权重仍然会被更新，只不过他的更新不顾及 Prompt。因此随着模型的更新，对 user Prompt 的理解会越来越强大，但是不去考虑他的生成。
            # 由于 Prompt 位置的 Loss 是 0，梯度信号里完全没有“为了让 Prompt 预测更准”而产生的修正要求。所有的修正要求（梯度向量）都只指向一个方向：“怎么让 Response 预测更准”。

            loss = logits_loss + res.aux_loss
            # 将文本预测的损失与 MoE 负载均衡损失相加。
            loss = loss / args.accumulation_steps
            # 将最终 Loss 除以累积步数。

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = logits_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Reasoning Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='reason', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=720, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl", help="推理蒸馏数据路径")
    parser.add_argument('--from_weight', default='dpo', type=str, help="基于哪个权重训练，默认dpo")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Reasoning", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, tokenizer, lm_config, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), tokenizer, lm_config, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()