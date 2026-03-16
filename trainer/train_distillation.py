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
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    # 让学生模型不仅向真实标签（Ground Truth）学习，还向教师模型的“软概率分布”学习。
    # 这个函数计算的是学生分布与教师分布之间的 KL 散度（Kullback-Leibler Divergence）。
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
        # 强制不对教师模型进行梯度计算。在蒸馏中，教师是“老师”，他的知识是固定的，我们只更新“学生”的参数。
        # torch.no_grad() 表示前向传播不计算梯度，也就不会有激活值占据显存。
        # detach 的作用：
        # 1. 调用 detach() 后返回的新张量，其 requires_grad 属性会被强制设为 False。即便原张量的 requires_grad 是 True，剥离出来的那个分身也不会被优化器更新。
        # 2. 内存层面的“共享空间”。detach() 并不拷贝数据。它创建了一个新的张量头部（Tensor Header），但它指向的底层内存数据和原张量是同一块。如果我们修改了 y.detach() 里的值，原张量 y 里的值也会跟着变。它只是在逻辑上断开了梯度联系，而不是在内存里复制了一份。

        # 这里的 detach 的作用是告诉 Pytorch：我只需要教师模型吐出来的这些概率数值。请把它们当成一堆死的数据（像从磁盘读入的标签一样），不要试图去计算教师模型的梯度，更不要把我的学生模型的梯度传给教师。

        # 如果这里不写 with torch.no_grad()，就会出现梯度回传给教师，例子如下：
        
        # # 错误示范：没有 detach，也没有 no_grad
        # teacher_output = teacher_model(input) 
        # student_output = student_model(input)
        # # 计算 Loss，此时 loss 关联了两个模型的全部参数
        # loss = F.kl_div(F.log_softmax(student_output), F.softmax(teacher_output))
        # loss.backward() # 这一步会试图计算两个模型的梯度

        # 虽然由于你没把 teacher_model 给优化器，教师的权重不会被修改，但 loss.backward() 在执行过程中依然会消耗大量的计算资源和显存去追溯教师模型的梯度，这极大地降低了训练效率。
        # 教师模型通常参数量巨大。一旦开始为它计算梯度，显存中会立刻堆积大量的中间激活值（Activation）和梯度张量（Gradients），导致显存溢出报错。

        # with torch.no_grad() 是如何运行的？
        # 1. 在块内生成的 teacher_logits，其属性 requires_grad 会被自动设为 False。既然 requires_grad 是 False，当你在块外计算 kl_div(student_log_probs, teacher_logits) 时，Autograd 发现其中一个支路（Teacher）是不可导的，它就会自动停止向那个方向回溯。
        # 2. 前向传播的激活值不再保存。

        # detach 的根本作用是断开计算图。加入后续有下面的代码：
        # with torch.no_grad():
        #     t_out = teacher_model(x)
        # # 离开块后，由于某种业务需求，你写了：
        # t_out.requires_grad = True 
        # # 或者你把它和一个有梯度的张量进行了某种就地（inplace）操作

        # 就会出现梯度回传了，并且路径是 loss -> teacher_logits -> 教师模型参数。但是如果直接进行了 detach，那么梯度回传路径就会在 teacher_logits 这里断开。loss -> teacher_logits -> 戛然而止。基本上，加了 detach，就当做 GroundTruth 看待就行。

        # ⭐️⭐️⭐️⭐️⭐️ Pytorch 的 Autograd 需要传染性。在 PyTorch 中，一个操作的输出是否需要梯度，取决于它的输入，如果一个函数的所有输入都不需要梯度（requires_grad=False），那么输出也就没有梯度；但是，如果输入中哪怕只有一个张量的 requires_grad=True，那么输出就一定会是 True。

        # 所以在 Pytorch 中大部分中间输出的梯度都会被记录，但即使这些梯度没有用，因为我们不会更新中间输出的数值。但也不能说没用，我们确实不会直接更新这些数值（你更新的是 W 和 b），但根据微积分的链式法则，我们计算 W 的梯度需要这些中间输出的梯度。
        # PyTorch 是一个“忠实记录者”，它不知道哪些梯度是你“打算要”的，它只知道哪些梯度是“理论上可以算”的。

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    # 教师的输出是 teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)，teacher 进行了 detach，而 student_log_probs 没有做 detach，这是因为需要纳入梯度回传。

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 计算 kl 散度，reduction=reduction：
    # * 如果设为 batchmean（常用），则返回整个 Batch 的平均 Loss。
    # * 如果设为 none，则返回每个 Token 独立的 Loss。

    # 如何理解 kl 散度？
    # KL 散度就是“知识的搬运工”。它通过计算学生分布与教师分布的“不匹配度”，将教师模型对 Token 之间关系的深层理解，压缩并传递给学生。
    # 在模型训练里，如果你发现 KL 散度一直很高降不下来，从信息论的角度看，就是学生模型的“复杂度（容量）”不足以编码教师模型输出的高复杂度信息。

    return (temperature ** 2) * kl
    # 梯度缩放补偿。
    # 当你通过温度 T 平滑了概率分布后，梯度的强度会由于 1/T^2 的比例发生萎缩，必须通过乘以 T^2 来补回这部分强度。
    # 这里的补偿一般只存在于 KL 散度的蒸馏损失中。
    # KL 散度越小，代表两个分布越接近；KL 散度越大，代表两个分布差异越明显。
    # KL 越小。学生已经成功捕捉到了教师的“逻辑”。不仅是知道正确答案是什么，连“错误答案之间的相似性”也模仿得很像。
    # KL 越大。学生模型目前还理解不了教师给出的“软标签”。例如：老师觉得 A 和 B 很像，但学生觉得 A 和 C 很像。

# 关于蒸馏，一种是数据蒸馏（例如大模型蒸馏），一种是 logits 蒸馏，logits 蒸馏应该是在分类任务中应用，那么回归任务蒸馏是怎么做的呢？例如 Reranker 蒸馏。
# 回归蒸馏采用 MSE Loss 或者 Margin MSE，后者是目前 Reranker 蒸馏的主流。我们不要求分数值绝对相等，但要求相对差距一致。

# 什么是数据蒸馏？什么是 logits 蒸馏？
# * 数据蒸馏是离线的，老师和学生不需要同时运行。数据蒸馏的学习目标是老师生成的文本 (Text)。由于老师更有文采（30个词），学生通过模仿这些长难句，变得更有文采。
# * logits 蒸馏是在线的，老师和学生必须同时运行（或者你要提前存下巨大的 Logits 文件）。学习的是老师输出的 概率分布 (Logits)。由于老师的“直觉”更敏锐，学生学到了词与词之间深层的关联。

# 什么时候用数据蒸馏？什么时候用 logits 蒸馏？
# * 数据蒸馏：离线、成本低、跨架构、学“逻辑”。
# * logits 蒸馏：在线、学“直觉”、防过拟合、学“暗知识”。
# 在目前的工业界（如 DeepSeek、Llama-3 的开发），最强的做法是先做数据蒸馏，用大模型（Teacher）把几千亿、几万亿的原始语料进行清洗、重写、扩充，生成一套完美的“教材”。然后再做 Logits 蒸馏，在学生模型学习这套“教材”的过程中，实时计算老师的 Logits，用你的那行 distillation_loss 进行监督。


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    start_time = time.time()
    
    if teacher_model is not None:
        # redo，重复将 teacher model 设置为评估模式，同时冻结参数，禁止其更新。
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        loss_mask = (labels[..., 1:] != -100).float()
        # 损失掩码计算。
        # 1. labels[..., 1:]：切片操作，通常是为了对齐模型输出（Logits）和标签。在自回归语言模型中，预测的是“下一个词”，所以标签通常会向后偏移一位。
        # 2. != -100：在 PyTorch 的 CrossEntropyLoss 中，-100 是默认的忽略索引（Ignore Index）。比如 Padding（填充部分）或 Prompt 部分通常设为 -100。
        # 3. .float()：将布尔值（True/False）转换为浮点数（1.0/0.0）。
        # 目的是得到一个掩码矩阵。在后续计算损失时，只有 1.0 的位置（即真正的答案部分）会被计入总损失，0.0 的位置（填充或无关部分）会被过滤掉。
        
        # (labels[..., 1:] != -100).float() 的结果如下：
        # [0.0, 1.0, 1.0, 0.0]。
        # A. 1.0 (True)：表示该位置是一个实际需要学习的 Token（比如对话中的回答部分）。
        # B. 0.0 (False)：表示该位置是 Padding（填充占位符） 或者 Prompt（问题部分）。
  
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（学生模型）
        with autocast_ctx:
            res = model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()
            # 拿到学生的 logits。
            # 为什么要切掉最后一位？因为最后一个位置生成的 Logits 没有对应的 labels 可以去对比（Label 已经到头了）。为了让 N-1 个预测结果和 N-1 个标签一一对应，我们必须舍弃掉 Logits 的最后一步输出。

        # 教师模型前向传播（只在eval & no_grad）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 拿到 teacher 的 logits。
                vocab_size_student = student_logits.size(-1)
                # 这两行代码揭示了知识蒸馏（Knowledge Distillation）中一个非常经典且棘手的现实问题：教师模型和学生模型的词表（Vocabulary）大小不一致。
                # 获取学生模型输出层（Logits）的最后一个维度。含义为这个维度就是学生模型所认知的词表大小（例如 32000）。
                # 在知识蒸馏中，经常会出现这种情况：教师模型（大模型）可能使用了更广的词表（比如 64000 个词），包含了很多生僻词或特殊符号。学生模型（小模型）为了轻量化，通常会压缩词表（比如只保留最常用的 32000 个词）。
                teacher_logits = teacher_logits[..., :vocab_size_student]
                # 它对教师模型的 Logits 进行切片，只保留前 vocab_size_student 个词的预测概率，把超出学生词表范围的那部分预测直接“扔掉”。
                # 这种做法（直接切片 [:vocab_size_student]）默认了一个前提条件——学生模型的词表必须是教师模型词表的“前缀子集”。
                # 但是在这里的代码，词表大小是一样的，因为输入的 input_ids 都一样。
                

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss
        shift_labels = labels[..., 1:].contiguous()
        # 拿到移位的 label。
        loss_mask_flat = loss_mask.view(-1)
        # 将二维的掩码张量 [Batch, Seq_Len] 拉平为一维向量，内容是 0 1 相间的内容。
        # 并不是序列中所有的词都需要计算损失（例如 Padding 部分或 Prompt 部分）。将其拉平是为了后续与展平后的 Loss 进行逐元素相乘。
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            # 将学生模型的输出（Logits）和标签全部展平为一维长向量，方便批量计算。
            shift_labels.view(-1),
            ignore_index=-100,
            # 对于标签为 -100 的位置，其对应的 Loss 在 F.cross_entropy 计算结果中确实是 0。
            reduction='none'
            # 这是关键一步。它不立即对整个 Batch 取平均，而是返回每个 Token 对应的独立损失。这样我们后面才能手动应用 loss_mask。
        )
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # 这里的 * 就是 逐元素相乘。
        # 这里的 ce_loss_raw 是一个标量，用来反向传播。
        # ce_loss 已经通过 ignore_index=-100 设置为了 (..., 0, 0, ...) 的形式，这里再乘 loss_mask_flat 有点多此一举，但是在对话微调（SFT）中，Prompt（问题）部分的 Label 可能并不是 -100（为了方便调试或记录），但我们通过 loss_mask 规定，只计算 Answer（回答）部分的损失。
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        # 添加专家负载均衡损失。
        else: ce_loss = ce_loss_raw

        # 2) Distillation Loss
        if teacher_model is not None:
            # 兼容性编程。这让你的代码具有兼容性。你可以用同一套脚本进行普通 SFT（此时 teacher_model 为 None）或者进行蒸馏训练。
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                # 将多维的输出（通常是 [Batch, Seq_Len, Vocab]）展平为二维矩阵 [Total_Tokens, Vocab]。
                # loss_mask_flat == 1：这是一个布尔掩码（Mask）。在 SFT 或指令微调中，我们通常只计算“回答”部分的 Loss，而不计算“提示词（Prompt）”部分的 Loss。
                # [...]（切片索引）：利用掩码只提取出有效 Token 对应的预测值。
                # 结果就是只把学生模型对“答案部分”的预测送入 Loss 函数，剔除掉 Padding 和 Prompt 干扰。

                # 回答一个问题——在 logits 蒸馏的过程中，是怎么保证 teacher 的输出 token 数目和 student 输出 token 数目保持一致的？
                # 需要清楚辨析自回归框架下，训练和推理的不同，训练是 Teacher Forcing 的，给定 N 个 Token，模型就需要依据前 N 个 Token 预测下一个，所以最终的预测结果一定是 N，学生和教师模型都是 N。
                # 在个数都是 N 的基础上，预测 token 分布是否一致。实际过程有点类似于——摊开一本现成的书（数据集），老师和学生并排坐着，老师指着书上的每一个字，告诉学生他在这个位置是怎么想的。

                temperature=temperature
                # 温度越高，KL 散度包含的“暗知识”就越多。
                # 在 T=1 时（标准状态）：KL 散度几乎只关注“谁是第一名”。如果老师认为“苹果”是 0.9，“香蕉”是 0.09，“汽车”是 0.01。学生只要把“苹果”预测准了，KL 散度就很小。至于学生认为“香蕉”和“汽车”哪个更像水果，KL 散度根本不在乎（因为数值太小，没存在感）。
                # 在 T=5 时（高温度）：原来的分布可能变成了：“苹果” 0.4，“香蕉” 0.3，“汽车” 0.1。这时 KL 散度开始“干活”了：如果学生认为“汽车”比“香蕉”更像水果，KL 散度会产生明显的波动。结论：高温度下的 KL 散度强迫学生去模仿老师对非正确选项的排列组合。这就是在学“直觉”。

                # 因此，在 Logits 蒸馏中，T 应该设置为 2~3，如果你觉得教师模型非常强大（比如用 70B 蒸 1B），且你的训练数据非常干净，可以用更高的温度来榨取教师模型对低概率 Token 排序的细微直觉，例如 4~5。
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
        # 计算总损失。只要是在 loss.backward() 之前定义的 Loss 组成部分，都必须除以 accumulation_steps。
        # 梯度积累（Gradient Accumulation）的本质是模拟一个大 Batch。假设 accumulation_steps = 2，原本你每一步的 Batch Size 是 16，现在你希望模拟 Batch Size 为 32 (2 * 16) 的训练效果。
        # 在正常的梯度下降中，Loss 应该是整个大 Batch 的平均值。当你把 2 个小 Batch 的梯度直接加起来时，总梯度会变成原来的 2 倍。为了让梯度回退到“平均值”的量级，每一项 Loss（无论是 ce_loss 还是 distill_loss）都必须先除以 2。
        # ⭐️⭐️⭐️⭐️⭐️ 我们要看的是单位样本的更新强度，梯度积累本质就是在模拟大 Batch，因此也需要保证单位样本。

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
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=512, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=16, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    # 一般来说，教师模型（Teacher Model）更大更复杂，此外教师模型还是冻结的模型。
    # 只有架构更简单的学生模型会得到优化。
    
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
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义学生和教师模型 ==========
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    # 初始化学生模型及其对应的分词器（Tokenizer）。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    # 1. 加载教师模型。教师模型通常参数量更大、精度更高，作为学生学习的标杆。
    # 2. 将教师模型设为评估模式（关闭 Dropout 和 Batch Normalization 的动态更新）。
    # 3. 冻结参数。告知 PyTorch 不需要为教师模型计算梯度，从而节省内存和计算资源。
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
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
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()