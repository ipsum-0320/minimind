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
    # epoch 表明是第几个 epoch。
    # loader 表明是存储数据的地方。函数通过 for ... in loader 不断从中获取 (input_ids, labels) 数据包。它决定了模型在这一轮里能看到哪些数据，以及是以多快的速度（num_workers）看到的，其实 loader 出来的数据是 batch 数据，因为 loader 会被 sampler 按照 batch 去采样。
    # iters 表示每一轮预期的总迭代次数，每个 step 蕴含的是 batch 个数据。
    # start_step 是起始训练的 step。
    # wandb，如果传入了该对象，函数会将 Loss、学习率、耗时等数据上传到云端。
    start_time = time.time()
    # 后面用来计算训练速度（Tokens/sec）以及预估剩余完成时间（ETA）。
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # start=start_step + 1 是为了衔接断点。
        # 如果 start_step 是 500（即之前练了 500 步），那么这次循环拿到的第一个 step 变量就是 501。
        # 如果不加这个，断点恢复后的日志会从 1 开始显示，导致曲线断裂。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 移动到 GPU 上。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 调用 get_lr 函数（通常是余弦退火策略），根据当前的进度百分比算出这一步应该用的学习率（通常是先上升再平滑下降）。
        for param_group in optimizer.param_groups:
            # 将刚才算出的 lr 真正注入到优化器中。
            # optimizer.param_groups 是一个列表，里面存了模型各层参数及其对应的超参数（如学习率、权重衰减）。虽然大多数时候只有一个 group，但如果我们对模型的 Embeddings 层和 Transformer 层设置了不同的学习率，这里就会循环更新所有的组。
            param_group['lr'] = lr

            # 这里的 optimizer.param_groups 如下所示，其中 params 就是每个参数权重（m * n）：
            # [
            #     {
            #         'params': [Parameter(tensor(...)), Parameter(tensor(...)), ...], # 模型的所有权重张量
            #         'lr': 0.0001,           # 当前组的学习率
            #         'betas': (0.9, 0.999),  # Adam 专有的超参数
            #         'eps': 1e-08,
            #         'weight_decay': 0.01,   # 权重衰减
            #         'amsgrad': False,
            #         'maximize': False,
            #         # ... 其他优化器特定的超参数
            #     }
            # ]

        with autocast_ctx:
            # 它告诉 GPU：在接下来的计算中，自动将适合的操作（如矩阵乘法）从 float32 转为更快的 bf16 或 fp16。
            # model(...) 执行前向传播，计算预测值并与 labels 对比。
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            # 这是 MoE（混合专家） 模型的典型写法。除了主损失，还有 aux_loss（负载均衡损失），
            # 确保模型不会只盯着某一个专家练，而是“雨露均沾”。
            loss = loss / args.accumulation_steps
            # 梯度累加的关键步，如果我们想用大 Batch 但显存不够，可以分多次计算。比如累加 4 步，每一步的 Loss 就要除以 4，这样 4 步加起来的梯度才是正确的平均梯度。


        scaler.scale(loss).backward() 
        # 这里的 backward() 会不断累积梯度，这正是梯度累加能够实现的底层逻辑。同时，如果是多卡训练，那么在 backward() 还会进行梯度同步。
        
        # 对于 fp16 精度，其实他能表示的数字范围很窄，如果太大就会 inf 溢出，如果太小则会直接表示为 0。
        # 在大模型反向传播时，很多梯度的数值恰恰就是这么小。如果直接用 fp16 计算，这些微小的梯度会变成 0，模型就练不动了。

        # 一个理想的解决方案是把信号“放大”。
        # scaler.scale(loss) 做的事情非常直观——在算梯度之前，把 Loss 乘以一个很大的系数。因为导数具有线性关系，当我们把 Loss 放大 S 倍，算出来的梯度也会同步放大 S 倍。
        # 这样，原本那些极其微小、快要变成 0 的梯度，被强行拉回到了 fp16 能表示的数值范围内。信号被保住了。

        # 我们肯定会问，梯度都被放大了几万倍，直接更新参数不就乱套了吗？这就是为什么我们后面的代码里有一行：
        # scaler.unscale_(optimizer)，在正式更新参数之前，scaler 会负责把那些放大的梯度再除以 S，还原回真实的大小。

        # 那如果 Loss 本来就很大呢？如果 Loss 本来就很大，再乘以 scaler 的放大倍数（默认往往是 65536），确实会面临“数值溢出” (Overflow) 的风险。
        # 针对这种情况，GradScaler 有一套非常聪明的**“动态伸缩”**机制。如果在 scaler.scale(loss).backward() 之后，某些梯度因为 Loss 太大而溢出变成了 Inf 或 NaN。scaler.step(optimizer) 在执行时会检查所有的梯度。 一旦发现任何一个参数的梯度里含有 Inf，它会直接跳过这次参数更新（即不调用 optimizer.step()）。这样可以防止坏数据把已经练好的模型权重“改坏”。
        # 然后就是动态调低倍数，这是 scaler.update() 这行代码的功劳。如果这一次迭代因为溢出而被跳过了，scaler 会在后台自动调低放大倍数（比如把 $S$ 从 65536 降到 32768）。它会不断尝试，直到找到一个既能保住小梯度、又不会让大 Loss 溢出的“甜点”倍数。

        # 为什么 bf16（1 + 8 + 7）可以解决 fp16（1 + 5 + 9）的问题？
        # 核心就是 bf16 的指数更大，能够表示更大/更小的数字，但是 fp16 的指数比较小，能表示的数值范围很窄。

        # 指数的本质：
        # 指数决定了数字的量级（Scale）。它告诉计算机，这个数字应该在哪个“数量级”的区间内寻找。
        # 它的作用是决定量程（Range）。直观理解就是指数就像是一个变倍倍镜。指数位决定了我们的模型能否“够得着”那些极其微小的梯度。如果指数位不够（像 FP16 只有 5 位），它的倍镜倍率调不到纳米级，遇到 $10^{-9}$ 这种微小信号时，倍镜里就是一片漆黑（显示为 0）。

        # 尾数的本质：
        # FP16 的尾数有 10 位。二进制的 $2^{10} = 1024$。这意味着：
        # 在任何一个指数档位内，FP16 只能把这个区间等分成 1024 份。
        # 假设现在的指数档位在 $1.0$ 到 $2.0$ 之间（即指数 $E=0$）。
        # 尾数位全为 0：数字是 $1.0000$
        # 尾数位加 1：数字是 $1.0009765625$（即 $1 + 1/1024$）
        # 尾数位再加 1：数字是 $1.001953125$（即 $1 + 2/1024$）
        # 如果我们想表示 $1.0005$？对不起，表示不了。计算机只能在 $1.0000$ 和 $1.0009$ 之间选一个最接近的。这就是精度限制。

        if (step + 1) % args.accumulation_steps == 0:
            # 检查是否达到更新时机，核心是判断是否积累了足够步数的梯度。
            # 如果我们设置 accumulation_steps = 4，模型会连续算 4 个 batch 的梯度但不更新参数。只有到第 4 步（以及 8, 12...）时，才会进入这个 if 块执行真正的“参数修改”。
            
            scaler.unscale_(optimizer)
            # 将之前通过 scaler.scale(loss) 放大过的梯度除以缩放因子，还原回真实大小。
            # 为什么，因为接下来的“梯度裁剪”必须基于真实梯度的范数（Norm）来计算。如果梯度还是放大的状态，裁剪逻辑就会出错。

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 为什么要做梯度裁剪？
            # 在训练 MiniMind 这样的深度神经网络时，反向传播涉及大量的矩阵连乘。如果模型初始化不当、学习率过高或者数据中存在离群值，梯度可能会在层层回传中迅速膨胀。如果梯度变得极其巨大（例如 10^6），那么权重会瞬间被推飞到无法挽回的地步。
            # 因此我们需要去做梯度裁剪，梯度裁剪的核心使命就是解决“梯度爆炸（Gradient Explosion）”问题，尤其是在深度学习网络中。
            # 在深度神经网络中，反向传播遵循链式法则。梯度从输出层传回输入层的过程中，本质上是权重的连乘。如果不裁剪，如果层数很多（深度大），且权重 W 的值稍微大一点（比如初始化时略大于 1），这些数字连乘起来会呈指数级增长。

            # 梯度裁剪的原理是什么？
            # 代码中的 torch.nn.utils.clip_grad_norm_ 使用的是 范数裁剪（Norm Clipping）。它的逻辑不是简单地把超过 1 的梯度抹平，而是按比例缩放。
            # 1. 它把模型中所有的参数梯度想象成一根超长的一维向量，并计算这根向量的“长度”。
            # 2. 计算缩放因子 g (Scale Factor)。
            # 3. 假设你设置的裁剪阈值是 max_norm（代码里的 args.grad_clip，通常设为 1.0），如果 $||\mathbf{g}||$ 超过了 max_norm，系统会计算一个系数。
            # 4. 最后，将模型所有的梯度都乘以这个系数。
            scaler.step(optimizer)
            # 正式迈出更新的一步。它是 optimizer.step() 的混合精度版。它的逻辑非常聪明，它首先检查梯度中是否包含 Inf 或 NaN（由 scaler.scale 放大过头导致的）。如果梯度正常，它会调用 optimizer.step() 更新模型参数。如果梯度溢出（包含 Inf）：它会拒绝更新。这一步参数原地不动，直接跳过，防止坏数据污染模型。

            scaler.update()
            # 调整“放大镜”的倍数，这是 GradScaler 的动态自我进化过程。
            # 如果刚才 scaler.step 更新成功了（说明当前的放大倍数很安全），它会尝试稍微调大倍数（以便更好地捕捉微小梯度）。
            # 如果刚才因为溢出跳过了更新（说明倍数太大了，撞到了 FP16 的天花板），它会调小倍数（通常是减半）。

            optimizer.zero_grad(set_to_none=True)
            # 清空梯度，迎接下一次迭代。在 PyTorch 中，梯度是累加的。
            # 如果我们不手动清空，上一步的梯度会和这一步的梯度叠加在一起，导致训练逻辑彻底错误。

        if step % args.log_interval == 0 or step == iters - 1:
            # 每隔 log_interval 个 step 打印一次（防止屏幕被刷屏），或者在最后一个 step（iters - 1）强制打印，确保不遗漏最后的进度。
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            # 你在前面为了梯度累加，执行了 loss = loss / args.accumulation_steps。
            # 在记录日志时，必须乘回去。这样显示的才是当前这个 Batch 真实的、未被缩小的 Loss 平均值。
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            
            current_logits_loss = current_loss - current_aux_loss
            # 总 Loss = 文本预测 Loss (Logits) + 专家均衡 Loss (Aux)。
            # 通过相减，你可以观察到模型主要是哪部分在优化。如果 logits_loss 下降但 aux_loss 飙升，说明模型虽然学到了东西，但专家分配极度不均。

            current_lr = optimizer.param_groups[-1]['lr']
            # 获取当前的学习率。

            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            # 该 Epoch 还有大约多少分钟跑完。
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 终端输出。

            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
            # 将这些数据发送到云端。

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 每隔指定的步数（如每 1000 步）保存一次。
            # 在多卡训练（DDP）中，这段代码只允许“主卡”（Rank 0）执行。如果不加这个判断，所有 GPU 都会尝试同时写同一个文件，会导致文件损坏或系统崩溃。
            model.eval()

            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 根据模型配置（是否使用 MoE、隐藏层维度等）自动生成文件名。例如：model_512_moe.
            # pth。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 获取原始模型。
            
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # state_dict(): 获取所有参数的名称和数值。
            # v.half().cpu(): 这是工程优化的精华。
            # .half(): 将 fp32 转换为 fp16。这能让保存的模型文件大小直接减半，且基本不影响推理精度。
            # .cpu(): 将权重从显存（GPU）转到内存（CPU）再保存。这样可以避免保存过程占用宝贵的显存。

            
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            # 保存权重，同时保存优化器状态（optimizer.state）、**当前步数（step）**和 GradScaler 的倍数。
            # 这是为了“断点续传”。如果训练突然中断，你可以利用这个 checkpoint 完美复活，继续训练，而不会丢失进度。
            model.train() 
            # 把模型切换回训练模式（开启 Dropout 等）
            del state_dict
            # Python 有垃圾回收机制（GC），但它并不总是那么“勤快”。
            # 这段代码会马上释放掉 state_dict 所占据的内存。

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
    # TensorBoard: 由 Google 推出，最早随 TensorFlow 发布（现在也完美支持 PyTorch）。它将日志写在本地磁盘，我们需要手动启动一个本地服务器（执行 tensorboard --logdir=...）才能在浏览器查看。
    # WandB / SwanLab: 属于 MLOps 工具。它们通常将数据自动上传到云端（也可以私有化部署）。我们只需要打开一个网页链接，就能在任何地方查看训练进度。SwanLab 是一个国产开源的实验训练看板，它的操作逻辑和 WandB 非常相似，可以看作是 WandB 的轻量化/国产化替代方案。
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
        # 导入库，并尝试从 ckp_data（通常是加载的 checkpoint 权重文件）中读取之前保存的 wandb_id。wandb_id 是实验的唯一身份证。如果能读到 ID，说明我们是从之前的进度断点重跑的。
        resume = 'must' if wandb_id else None
        # 逻辑判断。如果找到了旧的 wandb_id，就将模式设为 'must'（必须续接）；如果是新实验，则为 None（开启新纪录）。
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        # 模型名 (MiniMind)、任务 (Full-SFT)、总轮数、Batch Size 和学习率。这让我们在后台一眼就能看出这组实验的超参数。
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
    # DistributedSampler 分布式采样，如果我们在多张显卡（Multi-GPU）上训练，
    # 这个采样器会确保每张显卡拿到的是不同的数据，避免大家都在算同样的内容，实现真正的并行训练。
    
    # 详细说说 DistributedSampler？
    # 1. 数据分片。在 DDP 环境中，代码是在所有 GPU 上并行复制运行的。如果没有 Sampler：所有 GPU 都会默认从 index 0 开始读，结果 4 张显卡都在练一模一样的数据，这不但浪费了算力，还会导致模型过拟合。有了 Sampler：它会根据当前进程的 rank（第几张卡）和 world_size（总共有几张卡），自动计算出当前卡该负责的索引区间。比如：卡 0 拿 [0, 4, 8...]，卡 1 拿 [1, 5, 9...]。

    # 2. 保证不重复不遗漏。它通过数学上的取模运算，确保在同一个 Epoch 内。不重复：同一条数据不会被发送给两张不同的显卡；不遗漏：数据集里的每一条数据都会被某一张显卡处理到；对齐处理：如果数据总量（比如 100 条）不能被显卡数（比如 3 张）整除，它通常会通过“微调”索引（补一点或删一点）来保证每张卡分到的 Batch 数量一致，防止训练时某张卡早早跑完而导致死锁（Barrier Wait）。

    # 3. 分布式环境下的“伪随机”混洗（Shuffle）。在单机训练时，我们直接 shuffle=True 即可。但在分布式环境下，所有卡必须使用相同的随机种子（Seed）来决定这一轮的顺序，但又要取出不同的子集。我们可以在每个 epoch 之前写入 train_sampler.set_epoch(epoch)。这一行会利用 epoch 数作为随机种子的一部分。这样可以保证虽然每张卡运行的是独立进程，但它们算出来的“打乱后的顺序”是基于同一份全局地图。此外，随着 Epoch 增加，数据顺序会变化，保证模型每一轮看的数据顺序都不同。

    # DistributedSampler 与 DataLoader 配合如下：
    # * Sampler 说：“这轮卡 0 应该拿索引 [5, 12, 18]”。
    # * DataLoader 拿到这几个数字，派出 Workers（子进程）。
    # * Workers 调用我们写的 __getitem__(5)、__getitem__(12)...
    # * 处理好的数据被打包发给 GPU。
    
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
        # 如果是断点续训。
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler']) 
        # 恢复混合精度缩放器
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 忽略特定参数。
        # 这行代码通常出现在使用了 RoPE（旋转位置编码） 的模型中（如 Llama, MiniMind 等）。
        # 这是什么？ freqs_cos 和 freqs_sin 是预先计算好的三角函数常量矩阵，用于辅助计算位置编码。它们存放在 buffer 中，而不是 parameter 中。
        # 为什么要忽略？ 
        # 1.  它们不需要同步：在分布式训练中，DDP 默认会检查所有显卡上的参数和 Buffer 是否一致，并在反向传播时尝试同步它们
        # 2.  避免冗余开销：这两个矩阵是固定的常量，每张显卡在初始化模型时生成的都一模一样。
        # 3.  防止报错：有些模型架构中，这些 Buffer 不参与梯度计算（甚至不参与前向传播的某些路径）。如果不手动忽略，DDP 可能会因为“发现某些 Buffer 没被使用”而抛出警告或错误，或者浪费带宽去同步这些永远不会改变的常量。
        model = DistributedDataParallel(model, device_ids=[local_rank])
        # DDP 还会确保 Buffer（如 Batch Norm 的均值）也能在多卡间正确同步。
        # local_rank 代表当前进程正在使用的那一张显卡的编号（比如 0 或 1）。
        
        # 为什么是 local_rank，而不是 cuda:{local_rank}？
        # 这是一个非常细微但关键的语义区别。其实，在代码的其他地方（比如 tensor.to()）确实可以写 cuda:{local_rank}，但在 DistributedDataParallel 的构造函数里，直接传入一个整数列表 device_ids=[local_rank] 是 PyTorch 预设的标准协议。

        # # 这种写法是完全正确的，也是常见的！
        # device = torch.device(f"cuda:{local_rank}")
        # model.to(device) 

        # # 或者更简洁地写：
        # torch.cuda.set_device(local_rank) # 设置当前进程的默认显卡
        # model.cuda() # 它会自动去到 set_device 指定的那张卡
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 从 start_epoch 开始循环。如果是从断点恢复，它会跳过之前已经完成的 Epoch。
        train_sampler and train_sampler.set_epoch(epoch)
        # 如果使用了分布式采样器，这行代码确保每一张显卡在每一轮拿到的数据乱序顺序是同步且变化的。
        # 原理就是它将 epoch 作为随机种子的一部分，使得同一 Epoch 内多卡数据不重合，不同 Epoch 间数据顺序不一致。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 在多卡（DDP）环境下，DistributedSampler 已经接管了数据分发和乱序；但在单卡模式下，由于 DistributedSampler 往往是 None，我们就需要手动接管这个“洗牌”过程。
        # 因此这里的代码仅仅是为单卡（或非分布式）训练设计的“洗牌”逻辑，属于 patch code。torch.randperm 会受到 torch.manual_seed(seed) 的影响。
        # indices 扮演了“单卡版采样器”的角色。在分布式训练中，DistributedSampler 会自动生成这种清单，但在单卡训练时，没有采样器帮我们洗牌，所以我们通过这行代码生成 0 到 数据总量-1 的所有数字。随机打乱它们的顺序。存入 indices 列表。交给 SkipBatchSampler。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 跳过多少个 step。只有当我们恢复训练的第一个 Epoch 需要跳过数据。
        # 场景逻辑：假设我们设定的训练是 10 个 Epoch，我们在 Epoch 3 的第 500 个 Step 挂了。
        # 当我们重启，epoch 等于 3（即 start_epoch）。此时 skip 被设为 500。
        # 等 Epoch 3 跑完，进入 Epoch 4，此时 epoch != start_epoch，skip 自动归零，恢复正常训练。

        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 普通的 Sampler 像是一个发牌员，从第一张牌发到最后一张。而 SkipBatchSampler 是一个**“快进发牌员”**。
        # 在数据读取的“启动速度”和“跳过效率”上，BatchSampler（尤其是我们代码中的 SkipBatchSampler）确实比普通的 Sampler 配合手动跳过要快得多。
        # 为什么 SkipBatchSampler 在断点续训时更快？如果我们不使用 BatchSampler 拦截索引，而是用普通的 Sampler 配合 for 循环跳过，逻辑会是这样的，首先是普通做法：
        # for i, data in enumerate(loader):
        #     if i < start_step:
        #         continue  # 这里的 data 已经被 Dataset 读出来并加载到内存了！
        #     train(data)
        # 然后是 batch_sampler 的快速做法：
        # # SkipBatchSampler 直接在索引层面切片
        # # 它只把剩下的索引交给 DataLoader
        # batch_sampler = SkipBatchSampler(sampler, batch_size, skip)
        # loader = DataLoader(..., batch_sampler=batch_sampler)

        # batch_sampler 做法如下 =>
        # 当 batch_sampler 的底层输入是 train_sampler（indices 同理）时，索引的产生流程是这样的：
        # 第一层：train_sampler (DistributedSampler)职责：负责“分片（Sharding）”。它根据当前的 local_rank（哪张卡）和 world_size（总卡数），从全局数据中挑出属于当前这张卡的那一部分索引。比如卡 0 拿到了全局索引中的 [0, 4, 8, 12, ...]。
        # 第二层：SkipBatchSampler职责：负责“打包”和“快进”。它从 train_sampler 给出的这一串属于当前卡的索引里，按 batch_size 捆成一组组，并根据 skip 参数扔掉前 N 组。卡 0 的索引变成了 [[0, 4], [8, 12]]（假设 batch=2），如果 skip=1，它就直接把 [0, 4] 扔了，只返回 [8, 12]。


        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 后面在定义 DataLoader 时，一定不能再设置 shuffle=True。如果指定了 sampler，DataLoader 的 shuffle 必须为 False（或者默认不设），因为数据的乱序逻辑已经由 DistributedSampler 接管了。如果我们强行再设 shuffle=True，程序会报错。
        # 1. pin_memory=True 锁页内存优化。它能让数据从内存（CPU）拷贝到显存（GPU）的速度更快，是训练加速的标配。
        # 2. batch_sampler=batch_sampler 是我们之前定义的“快进版打包员”。它已经决定了这轮训练要从哪个索引开始、每一批（Batch）拿哪几条数据。
        # 3. num_workers 开启多进程读数据。比如设为 8，就会开 8个子进程预先从硬盘读数据并处理，防止 GPU 因为等不到数据而“饿死”。
        if skip > 0: 
            # len(loader) + skip 是为了修正进度条的总步数。
            # 假设总共 1000 步，跳过了 300 步。此时 loader 里其实只剩 700 步了。
            # 如果不加 skip，进度条会显示 0/700。加上 skip 后，进度条会显示 300/1000，视觉体验上是连续的。
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
    # 正式销毁进程组，关闭进程间的通信渠道。