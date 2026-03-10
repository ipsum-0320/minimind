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


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

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
            current_logits_loss = current_loss - current_aux_loss
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
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    # wandb 和 tensorboard 的区别是啥？
    # TensorBoard: 由 Google 推出，最早随 TensorFlow 发布（现在也完美支持 PyTorch）。它将日志写在本地磁盘，你需要手动启动一个本地服务器（执行 tensorboard --logdir=...）才能在浏览器查看。
    # WandB / SwanLab: 属于 MLOps 工具。它们通常将数据自动上传到云端（也可以私有化部署）。你只需要打开一个网页链接，就能在任何地方查看训练进度。SwanLab 是一个国产开源的实验训练看板，它的操作逻辑和 WandB 非常相似，可以看作是 WandB 的轻量化/国产化替代方案。
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    # 初始化分布式环境（DDP），它会确定当前进程在所有 GPU 中的编号（Rank）。
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 如果是多卡训练，将当前进程绑定到对应的 GPU 上。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    # 设置随机种子。注意：每张卡的种子微调了一下，防止多卡训练时读取完全相同的数据顺序。
    # 在这里引入随机数种子是为了引入多样性，这体现在 Dropout（随机丢弃神经元）或某些随机增强上。如果所有卡的 Dropout 位置完全一样，那多卡训练就变成了简单的“数值放大”；如果每张卡丢弃的位置略有不同，它们计算出的梯度就会有微小的差异，合并（All-reduce）后的梯度会更平滑，模型泛化能力更强。
    # 具体 seed 还影响了哪些因素，可以去看 setup_seed。

    # 后续还会有 set_epoch(epoch)，这里的 set_epoch 的作用是做好数据分配。
    # 1. 所有卡必须商量好，“在第 2 轮（Epoch 2），我们要把食材按 [C, A, D, B] 这个顺序排好。”
    # 2. 达成共识后，卡 0 拿走 C，卡 1 拿走 A。
    # 3. 如果没有这个共识，卡 0 以为顺序是 [A, B, C, D] 取走了 A；卡 1 以为顺序是 [C, A, D, B] 也取走了 A；结果大家都在练重复的数据，效率减半，甚至导致过拟合。

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 在硬盘上创建一个文件夹，用来存放训练过程中产生的模型权重（.pth 或 .bin 文件）。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 通过 MiniMindConfig 配置创建模型骨架，核心就是去定义一些 LLM 的超参数。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    # 这是一个逻辑判断。如果用户设置了 from_resume=1（即想要从上次断掉的地方接着练），程序就会去指定的目录寻找旧的“存档”。
    # 代码会根据 ckp_data 是否为空，来决定是“白手起家”还是“接力训练”。
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 确定计算将运行在什么硬件上。
    # 注意，在多卡运行环境下，如果要指定当前进程运行在某个卡上，需要指定 cuda:{local_rank}。
    # 由于目前默认 device_type = "cuda"，因此所有的进程都会挤到 cuda:0，也就是 0 号卡上。
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 决定使用哪种半精度浮点数格式。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    # 根据设备创建一个“自动缩放精度”的上下文管理器。
    # 节省显存并大幅提升算力受限操作的执行速度。
    # 因为默认的精度是 bf16，因此 autocast_ctx 会将 bf16 精度提升至 fp32 进行计算，例如 Softmax。
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 在分布式（多卡）训练中，**只有主进程（Rank 0）**负责记录日志。如果不加这个判断，每张显卡都会启动一个 SwanLab 网页，导致数据重复和乱码。
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 导入库，并尝试从 ckp_data（通常是加载的 checkpoint 权重文件）中读取之前保存的 wandb_id。wandb_id 是实验的唯一身份证。如果能读到 ID，说明你是从之前的进度断点重跑的。
        resume = 'must' if wandb_id else None
        # 逻辑判断。如果找到了旧的 wandb_id，就将模式设为 'must'（必须续接）；如果是新实验，则为 None（开启新纪录）。
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        # 模型名 (MiniMind)、任务 (Full-SFT)、总轮数、Batch Size 和学习率。这让你在后台一眼就能看出这组实验的超参数。
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
        # 连接服务器，开启本次实验记录。
        # project: 项目分组（如 "MiniMind-LLM"）。
        # name: 刚才生成的实验易读名称。
        # id: 传入旧 ID 保证图表曲线不会断开。
        # resume: 告诉服务器这是续传还是新建。
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 这里的 init_model 会根据配置项加载模型权重和分词器（Tokenizer）。
    if args.use_compile == 1:
        # 这是 PyTorch 2.0+ 的黑科技。它会将 Python 代码编译成更高效的内核（Kernels），
        # 从而在不改变模型结构的前提下，显著提升训练速度。
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 加载训练数据，并根据给定的 max_length 进行截断或填充。
    # SFTDataset 继承自 Dataset。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # DistributedSampler 分布式采样，如果你在多张显卡（Multi-GPU）上训练，
    # 这个采样器会确保每张显卡拿到的是不同的数据，避免大家都在算同样的内容，实现真正的并行训练。
    
    # 详细说说 DistributedSampler？
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 这里的 GradScaler 用于处理 float16 训练时的梯度缩放。它可以减少显存占用并加快计算速度，同时通过缩放防止梯度在大模型训练中因精度不足而“消失”（变成 0）。
    # GradScaler 的核心作用是防止溢出（Overflow），如果我们的 dtype 是 bf16（8 + 7，指数 + 尾数）就不需要了。
    # 但其实这行代码在 dtype 不是 'float16'（比如是 'bf16' 或 'fp32'）时，会将 enabled 设为 False。此时 scaler 实际上是一个“空壳” (Identity function)，它不会执行任何缩放操作。
    # bf16 需要较新的显卡支持（如 NVIDIA Ampere 架构及以上，即 RTX 30系列、A100、H100 等），老卡只支持 fp16。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 深度学习中最常用的优化器之一，它是 Adam 的改进版（更好地处理了权重衰减），负责根据计算出的梯度来更新模型的参数。
    
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
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()