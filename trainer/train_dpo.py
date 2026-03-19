import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    # 在 vocab_size 的维度去做 softmax。
    # log_probs 的形状也是 (batch_size, seq_len, vocab_size)。
    # log_probs 表示在句子的每一个位置，模型对词表中“所有可能出现的词”所预测的对数概率。因为 P \in [0, 1]，因此 logP \in (-\infty, 0]。越接近 0，代表概率越大（模型越确定是这个词）。越小的负数（如 -10），代表概率越小。
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    # labels 在执行了 unsqueeze(2) 之后，其 shape 变化如下：

    # # 原始 labels
    # # 每一行代表一个句子，每个数字代表该位置正确单词的 ID
    # tensor([[ 101,  500,  200],
    #         [ 300,  101,  888]])

    # # unsqueeze(2) 之后
    # # 每个数字都被包裹在一个单独的方括号里
    # tensor([[[101],
    #         [500],
    #         [200]],
    #         [[300],
    #         [101],
    #         [888]]])
    
    # 这里的 dim=2 代表着，请保持 Batch 和 Sequence 的位置不动，只在 Vocab（词表） 这一横排数据里，根据 labels 提供的编号去抓取那个数值。

    # 下面是一个例子说明 torch.gather 在做什么 =>
    # 在 torch.gather 中，labels 只是“索引清单”，而数据永远是从 log_probs（那个大矩阵）里取的。

    # 假设我们有一个非常小的矩阵 input（模拟 log_probs）：
    # input = [
    #     [10, 20, 30],  # 位置0的三个词得分
    #     [40, 50, 60]   # 位置1的三个词得分
    # ]
    # 我们有一个索引清单 index（模拟 labels.unsqueeze(2)）：
    # index = [
    #     [0],  # 位置0我要取索引为0的数（即10）
    #     [2]   # 位置1我要取索引为2的数（即60）
    # ]

    # 执行 torch.gather(input, dim=1, index=index)：
    # 第一行：根据 index 的 0，去 [10, 20, 30] 里拿走第 0 个数 -> 10
    # 第二行：根据 index 的 2，去 [40, 50, 60] 里拿走第 2 个数 -> 60
    # ⭐️ 结果：
    # [ [10], [60] ]

    # torch.gather 在这里完成了从“全词表概率”到“指定标签概率”的过滤。
    # log_probs_per_token 的 shape 是 (batch_size, seq_len)。
    # 这个形状非常直观，它就是一个二维矩阵：
    # 行：代表 Batch 中的每一个样本（每一句话）。
    # 列：代表句子中的每一个 Token 位置。
    # 数值：每个位置上的数值就是模型预测那个正确 token 的对数概率。

    return log_probs_per_token
    # ⭐️⭐️⭐️⭐️⭐️ 这个值表示在当前模型看来，在已知前面所有词的情况下，写下『这个特定词』的可能性（对数概率）有多大
    # 这里的已知前面所有词的情况下非常重要，因为 logits 就是这么来的，他是 NTP 获取的。


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    # https://github.com/jingyaogong/minimind/issues/298
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  
    # 防止零长度mask导致除零NaN
    # 统计每个样本中 assistant 回答部分的真实长度（即 Mask 中 1 的个数）。clamp_min(1e-8) 是一个安全保险。如果某条数据全是 Mask 0（异常数据），长度会变成一个极小的正数，防止后面除法时出现 NaN（除零错误）。
    # clamp_min(保底值)，如果数值小于这个保底值，就强制让它等于这个保底值；如果数值本来就比保底值大，那就保持原样。

    # 关于整个序列的概率表示应该是 sum 还是 mean，可以参考 github 中原作者的回答：
    # 严格来说对于整个序列的概率应该是各个token概率的乘积（对数概率相加），所以使用 .sum(dim=1)，但是需要平均对数概率：
    # 1. 避免 loss 忽视长序列（因为长序列的 logits 总和会天然更小）。
    # 2. 不同长度 text 对 loss 影响更均衡。
    # 如果希望长序列和短序列的贡献不同，用sum；如果希望不同长度的样本贡献均衡，用mean。

    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    # 这行代码将“逐个 Token 的概率”转化为“整条回答的得分”，其可被描述为如下三步：
    # 1. (ref_log_probs * mask) —— 遮罩过滤。
    # * ref_log_probs 形状是 (batch_size, seq_len)，包含了整条序列（包含 User 问题、Prompt 模板和 Padding）的所有概率。
    # * 这行代码将概率矩阵与 mask（只有 Assistant 回答部分为 1，其余为 0）按元素相乘。
    # * 结果就是非回答部分的概率全部归零，这样我们就只剩下了模型生成“那段话”本身的概率。

    # 2. .sum(dim=1) —— 累加求和。
    # * 在对数环境下，logP + logQ = log(P * Q)。
    # * 这行将每一 token 的对数概率加在一起。
    # * 结果就得到了整段回答的总对数概率。这个数值反映了模型生成这一整段话的“总信心”。

    # 3. / seq_lengths.squeeze() —— 长度归一化（取平均）。
    # * 用总分除以回答部分的实际 Token 数量（seq_lengths），其实就是 Assistant 的回答 token 数。
    # * 为什么要这样做？
    # - 消除长度偏见：如果不除以长度，长句子因为 Token 多，累加出来的 logP 绝对值会非常小。
    # - 公平竞争：归一化后，得到的是平均每个 Token 的概率。这样短回答和长回答就能在同一个尺度下进行对比。
    # - 训练稳定性：平均值的波动比总和小，有助于梯度下降更平稳。

    # 这里为什么要 seq_lengths.squeeze()？
    # 假设 batch size 是 4，那么 ref_log_probs.sum(dim=1) 的结果形状是 [4]。在 PyTorch 中，当我们在某个维度求和且没设置 keepdim=True 时，该维度会消失。
    # 对于分母 (seq_lengths)。因为我们之前的定义是 seq_lengths = mask.sum(dim=1, keepdim=True)。设置了 keepdim=True，它的形状是 [4, 1]（一个二维矩阵，有一列），不做 squeeze 的话就会出现 Pytorch 广播。
    # seq_lengths.squeeze() 之后就是 [4,1] => [4]，因此分子是 [4]；分母是 [4]，结果是 [4] / [4] = [4]，这才是我们要的 4 个样本对应的平均概率。

    # 这里再复习一下 Pytorch 的广播机制。
    # 并不是任意两个张量都能广播，必须满足以下条件：
    # 右对齐：从最后一个维度（最右边）开始往左比对。
    # 维度相等 OR 其中一个是 1：
    # - 如果某个维度一个是 12，另一个是 1，可以广播（1 会变成 12）。
    # - 如果一个是 12，另一个是 5，报错！（无法克隆出这种对应关系）。
    # 维度缺失：如果一个张量比另一个维度少，它会被自动在左边补 1。

    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    # 在当前正在优化的模型（Policy Model）视角下，整条“好/坏回答”的平均可信度。
    # policy_log_probs 的 shape 就是 [batch]。

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    # 获取 batch size，这里的 batch = chosen batch + reject batch。

    # ref 的平均每个 token 拿了多少分。
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    # 这个就是 \pi_{ref}(y_w|x)
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    # 这个就是 \pi_{ref}(y_l|x)

    # policy 的平均每个 token 拿了多少分。
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    # 这个就是 \pi_{\theta}(y_w|x)
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]
    # 这个就是 \pi_{\theta}(y_l|x)

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 当前新模型觉得“好回答”比“坏回答”好多少，或者说好回答比坏回答生成的概率要大多少。
    # 由于是 log 空间，减法等于除法。这代表了模型认为好坏答案概率的倍数关系。

    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # 原始老模型觉得“好回答”比“坏回答”好多少。这提供了一个基准线。
    
    logits = pi_logratios - ref_logratios
    # 这是 DPO 最天才的地方。它计算的是 (新模型的好坏差) - (老模型的好坏差)。
    # 如果 logits > 0：说明新模型比老模型更懂得区分好坏了（进步了）。
    # 如果 logits < 0：说明新模型变傻了，甚至可能觉得坏回答更好。

    loss = -F.logsigmoid(beta * logits)
    # 一个超参数（通常在 0.1 到 0.5）。它决定了我们对“偏离老模型”的容忍度。beta 越大，惩罚越严厉。
    # logsigmoid：这是一个平滑的二分类逻辑函数，我们希望 logits 越大越好。
    # 当 logits 很大时，sigmoid(logits) 接近 1，log(1) 是 0，Loss 就很小。
    # 当 logits 很小时，Loss 会迅速增加。
    # - (负号)：因为梯度下降是往小的走，所以要把“最大化进步”转为“最小化损失”。

    return loss.mean()
    # 将整个 Batch 的损失取平均，交给优化器去更新参数。


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    # ref_model。在 DPO 算法中，我们要让当前模型产生的“好回答”概率变大，“坏回答”概率变小。但为了防止模型“练废了”（比如为了刷分而只会说重复的话），我们需要用一个原始的、没动过的模型作为基准。
    # beta。DPO 的温度系数（KL 散度约束系数）。beta 越大，对模型偏离参考模型（ref_model）的惩罚越重。模型会表现得更“保守”。beta 越小，模型就越不在乎参考模型，会更激进地向“好回答”的方向靠拢，但也更容易导致模式崩溃（Mode Collapse）。通常取值在 0.1 到 0.5 之间。

    # β 大 → 更保守的更新（强 KL）。
    # β 小 → 更激进的 preference learning（弱 KL），易模式崩溃。

    # β 很大：我非常信任 ref_model（老模型），你改动的时候必须极度克制。
    # β 很小：我不怎么在乎老模型，你可以为了区分好坏而大刀阔斧地改。

    # 在公式推导过程中，β 是放在 KL 的系数上的。
    start_time = time.time()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        # 遍历 DataLoader，每次循环会拿到一个 batch。
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        # 使用 torch.cat 函数在 第 0 维（Batch 维度） 将 chosen 和 rejected 两组数据垂直拼接起来。
        # 拼接后的形状：
        # - 如果原本 x_chosen 的形状是 [batch_size, seq_len]。
        # - 那么拼接后的 x 的形状就是 [2 * batch_size, seq_len]。

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                # 参考模型不更新参数，不记录梯度。
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
                # 这里的 logits 是最后一层输出的激活值，在进入 Softmax 之前。
            
            # 下面的逻辑是获取“旧模型”（参考模型）和“新模型”（当前正在练的模型）对同一组对话的“看法”。
            ref_log_probs = logits_to_log_probs(ref_logits, y)  
            # ref_log_probs 记录了没训练之前的模型，预测出这些词的概率是多少。这就像是一份“考前成绩单”，作为我们后续优化的基准线。
            outputs = model(x)
            # model 就是 policy model。
            logits = outputs.logits
            # logits 就是 policy model 对 x 里的每一位 token 预测出的全词表得分。
            policy_log_probs = logits_to_log_probs(logits, y)
            # 这是当前正在训练的模型，预测出同样这些词的概率是多少。
            # 重点是这里的 policy_log_probs 的 shape 是 (batch, seq_len)。
            # 这里的 batch 是 batch_chosen 和 batch_rejected 的组合。
            
            # 计算 dpo loss。
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # ⭐️⭐️⭐️⭐️⭐️ DPO 其实和 SFT 非常相似，在采样 policy model 的输出的时候，也是基于了 shift label 的方式去采样的。
            # 也就是说 DPO 在训练时，和 SFT 一样是在做高度并行的 NTP 的。我们可以把 Prompt + Answer 整个长序列一次性丢进 Transformer。利用 Transformer 的 Causal Mask（因果掩码），模型在计算第 n 个位置的 Logits 时，会自动“并行地”参考前 n-1 个 Token。
            # 这一步是并行的。无论序列多长，一次 Forward Pass（前向传播）就能算出整句话所有 Token 的概率。这也就是为什么 DPO 训练很快。

            # 此外，这也是为什么 DPO 的 Logits 和 Label 的长度确实是严格对齐的。但是这也就带来一个问题，DPO 是在强迫模型去精确模仿 chosen 样本的每一个 Token，包括它的长度、语气、甚至废话。

            # 因此，LLM RL 来了！

            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps
            # 做梯度积累。

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
            current_dpo_loss = dpo_loss_val.item() # dpo loss
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

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

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
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
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型（ref_model冻结）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    # 加载第二个模型实例，即 Reference Model。在 DPO 中，我们需要对比“新模型”和“旧模型”对同一输出的概率。这个 ref_model 通常加载与策略模型完全相同的初始权重。
    ref_model.eval()
    ref_model.requires_grad_(False)
    # 冻结参考模型。
    # DPO 的 Loss 含义为：
    # 1. 模型认为“好回答”比“原始状态”进步了多少。
    # 2. 模型认为“坏回答”比“原始状态”进步了多少。
    # 3. 优化方向应该使得好回答和坏回答的差距尽可能地大，也就是说，让“好回答”的进步程度远远超过“坏回答”。
    
    # 为什么需要在 DPO 中引入参考模型？
    # DPO 的目标函数中包含了一个隐式的 KL 散度（KL Divergence） 约束。参考模型就是这个约束的基准，确保模型在进化的同时，保留原有的语言理解能力。
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 构建 DPO Dataset。
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
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()