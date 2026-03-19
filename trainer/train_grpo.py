import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')

# ⭐️⭐️⭐️⭐️⭐️ 在 GRPO（Group Relative Policy Optimization）的训练阶段，不需要传统的“标准答案”（Ground Truth Label），它依靠的是“奖励信号”（Reward Signal）。因此，在 GRPO 阶段是没有 label 字段的，或者说，即使有 label，其也是评判的标准，而不是模仿的目标。
# ⭐️⭐️⭐️⭐️⭐️ GRPO 的 label。虽然 GRPO 训练不需要逐字对齐的答案，但奖励函数（Reward Function）需要知道什么是“对”的。对于数学/逻辑题，在 verl 的数据流中，label 通常存放的是最终答案（如 42 或 x=5）。Reward Manager 会提取模型生成的 <answer> 标签内容，并与这个 label 进行对比。
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    # prompts: 原始提示词列表。list[str], length B。
    # responses: 模型生成的回答列表。[B*num_gen]，内容是一个 string list。
    # reward_model: 预训练的奖励模型（RM），用于评估回答质量。
    # reward_tokenizer: 用于奖励模型的分词器。

    def reasoning_model_reward(rewards):
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        # 正则符号拆解：
        # ^ 和 $: 必须从字符串开头开始匹配，一直到结尾。这意味着除了这两个标签里的内容，模型不能在外面乱写任何东西。
        # <think>\n: 强制要求第一行必须是 <think> 标签并换行。
        # .*?: 非贪婪匹配，表示“思考过程”中的任何字符。
        # \n</think>\n: 思考结束必须有闭合标签并换行。
        # <answer>\n...: 接下来必须是紧跟的答案部分。
        # pattern2 的区别: 只是多了一个 \n，即兼容思考和答案之间多一个空行的情况。

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]
        # re.match: 尝试从字符串起始位置匹配。如果匹配成功返回一个 Match 对象，失败则返回 None。
        # re.S (DOTALL) 是最重要的参数。默认情况下，正则里的点 . 不匹配换行符。加上 re.S 后，. 可以匹配包括换行在内的所有字符。由于大模型的回答通常是多行的，不加这个参数正则会直接失效。

        format_rewards = [] # 用来存放这一批次 B * G 个样本的格式得分。
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                # match_pattern 是 Match 对象（Match Object）或 None。
                format_rewards.append(0.5)
                # 格式符合的给 0.5 分。
            else:
                format_rewards.append(0.0)
                # 格式不正确，给出 0 分，比如漏了 </think> 标签，或者标签写错了。
        rewards += torch.tensor(format_rewards, device=args.device)
        # 在 PyTorch 中，+= 符号（即 __iadd__）通常执行的是 原地操作（In-place Operation）。
        # 这行代码会将 Python 列表转为 GPU 上的张量，方便后续进行数学运算。
        # += 是一个累加操作。这意味着 rewards 变量之前可能已经存储了其他的奖励（比如答案正确性奖励）。
        # 此时 rewards 是一个形状为 [B * G] 的一维向量。

        # 这里的 reward 是引用传递。这和 Java 非常一样，我们可以完全用 Java 处理 对象（Object） 的逻辑来理解 Python 处理 Tensor 的逻辑 =>
        # 1. 名字是引用的拷贝。
        # 2. 方法调用（或原地操作符）改的是对象本身。
        # 3. 等号赋值改的是引用指向的位置。
        # 一般来讲，不可变对象（Integers, Strings, Tuples）是值传递，函数内的改动不会传递到函数外。
        # 可变对象（Lists, Dicts, Tensors）是引用的值传递，函数内的改动会传递到函数外。
        # Python 和 Java 一样，都是只有值传递的！
        
        # 在 Python 中，rewards 这个变量名就像一个标签，贴在了一块特定的 GPU 显存地址上。
        # 如果是 rewards = rewards + tensor。Python 会先算右边的加法，申请一块新显存存放结果，然后把 rewards 这个标签从旧地方撕下来，贴到新地方。旧的数据块会被回收。
        # 如果是 rewards += tensor。Python 会直接在 rewards 标签所指向的那块原始显存里，把数值加进去。

        # 在 PyTorch 中，+= 符号（即 __iadd__）通常执行的是原地操作（In-place Operation）。
        # rewards += torch.tensor(...) 相当于 rewards.add_(...)。它不会在内存中开辟一块新空间来存放相加后的结果，而是直接修改 rewards 原本指向的那块显存里的数据。这对性能非常友好，因为它避免了在大批量数据下频繁地分配和释放 GPU 显存。

        def mark_num(text):
            # 定义了一个名为“记号打分”的内部函数，它专门负责统计标签出现的次数。
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            # 只有当某个标签不多不少正好出现 1 次时，才给 0.25 分。
            # 为什么要这样设计？
            # - 防止漏写。如果模型忘了写 </think>，它会丢掉 0.25 分。
            # - 防止复读。如果模型为了刷分，疯狂输出 <think><think><think>，因为次数不等于 1，它拿不到这 0.25 分。
            # 总分 1.0，如果四个标签都规范地出现了一次，这个函数会返回满分 1.0。
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        # 使用列表推导式，对 B * G 条生成的回答文本逐一调用上面的打分函数。结果是一个包含 B * G 个浮点数的 Python 列表。
        rewards += torch.tensor(mark_rewards, device=args.device)
        # 将得分列表转为 GPU Tensor，并原地累加到之前的总分 rewards 中。
        # 此时的总分构成 rewards = 正确性分 + 格式正则分 (0.5) + 标签计数分 (最高1.0)。

        # 为什么要这样设计？
        # 这种打分方式被称为 “奖励塑造” (Reward Shaping)。
        # 如果只用之前的正则匹配（那个严苛的 r"^<think>..."），在训练初期，模型由于完全是随机乱写，可能 1000 次尝试里没有一次能完全对上正则，导致 Reward 永远是 0。这会导致模型因为收不到任何正面反馈而无法学习（梯度消失）。
        # 而 mark_num 提供了“安慰奖”：
        # 1. 即使模型写得一团糟，但只要它偶然写出了一个 <think>，就能拿到 0.25 分。
        # 2. 算法会捕捉到这 0.25 分的微小差异，引导模型往“多写标签”的方向靠拢。
        # 3. 随着训练进行，模型会慢慢拼凑出完整的格式，直到最终触发那个 0.5 分的“正则格式大奖”。

        # mark_num 只奖励了输出 <think> </think> <answer> </answer>，而 pattern 则奖励了这几个特殊符号按照正确的范式组织。
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)
    # len(responses) 获取生成的回答总数（即 B * G），torch.zeros 表示在 GPU 上创建一个全为 0.0 的张量。
    # 这是给每个样本建立一个“初始账户”。在没有任何表现之前，每个样本的奖励都是零。
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)
        # 如果训练的是普通对话模型，可能不需要这种严苛的 <think> 格式。
        # 只有当我们明确想训练一个像 DeepSeek-R1 那样的推理模型（Reasoning Model）时，才需要进行 format 奖励。
        # 虽然我们之前讨论过 Python 是对象引用传递，且函数内部有 += 原地操作，但显式地写成 rewards = func(rewards) 是一种更安全的工程实践。

    with torch.no_grad():
        # 这里使用的是 Reward Model (RM)，它是一个预训练好的“评委”。我们只需要它给出一个分数（推理），不需要更新它的参数。
        # 因此这里调用 torch.no_grad()，大幅减少显存占用，加快计算速度。
        reward_model_scores = [] # 用来存储这一批次所有生成的回答 (B * G) 被 RM 打完分后的结果。
        batch_size = len(prompts) # 获取原始问题的数量 B。
        scale = 3.0 # 定义一个缩放因子，神经奖励模型输出的原始值（Raw Score）通常是任意范围的浮点数。乘以 scale 或进行归一化，是为了让这个奖励的“力度”与之前的格式奖励（0.5、0.25）处于相近的量级，防止某一项奖励完全掩盖掉另一项。

        for i in range(batch_size):
            # 遍历每一个原始问题。
            for j in range(args.num_generations):
                # 遍历同一个问题的每一个不同回答（共有 G 个）。
                response_idx = i * args.num_generations + j
                # 定义 Response 回答索引，[i, j] 二维转一维。
                response = responses[response_idx] # 从刚才那个 batch_decode 出来的长列表里，取出当前正在处理的这条回答文本（String）。
                prompt = prompts[i] # 取出这条回答所对应的原始问题文本。
                # 奖励模型（RM）通常需要同时看 [Question + Answer] 才能准确判断回答得好不好。如果我们只给它看 Answer，它不知道模型有没有“跑题”。

                # 还原对话结构。
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                # <\|im_start\|>: 匹配对话起始标签。
                # (system|user|assistant): 第一个捕获组。识别这一段是谁说的（角色）。
                # \s+: 匹配标签后的空格或换行。
                # (.*?): 第二个捕获组。非贪婪匹配对话的实际内容。
                # <\|im_end\|>: 匹配对话结束标签。
                matches = re.findall(pattern, prompt, re.DOTALL) # 从 Im_dataset 可以看出，Prompt 是 apply_chat_template 的。
                # re.findall + re.DOTALL，找出 prompt 中所有的对话片段。re.DOTALL 确保 . 能匹配换行符（因为对话内容通常是多行的）。
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                # 将正则抓取到的结果（元组列表）转换为 OpenAI/常用框架标准的 List of Dicts 格式。
                # content.strip()：去掉内容前后的多余空格或换行，保证数据纯净。

                tmp_chat = messages + [{"role": "assistant", "content": response}]
                # 1. 这行代码实际上在做 list 的 cat 的。
                # 2. 此外，assistant 之前一定是有 user 的。
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                # get_score 是作者（或我们使用的框架）在 Hugging Face 基础模型之上手动封装的一个自定义方法。
                # 这个 get_score 可能是下面这样的：
                # def get_score(self, tokenizer, messages):
                #     # 1. 将 List[Dict] 转换成 ChatML 字符串
                #     prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                    
                #     # 2. 转化成 Tensor
                #     inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                #     # 3. 推理
                #     with torch.no_grad():
                #         logits = self.model(**inputs).logits # 假设是一个 Sequence Classification 模型
                    
                #     # 4. 返回浮点数
                #     return logits.squeeze().item()
                # 这个 score 代表了评委模型对这次回答的主观好感度。分数越高，表示模型认为回答越符合人类偏好（更准确、更有礼貌、逻辑更强）。
                score = max(min(score, scale), -scale)
                # 这是一个经典的夹逼函数逻辑（相当于 torch.clamp 的 Python 版），它将 score 强制限制在 [-scale, +scale] 之间。
                # min(score, scale): 确保分数不会超过上限。如果模型给出了 10.0 分，而 scale 是 3.0，则结果取 3.0。
                # max(..., -scale): 确保分数不会低于下限。如果模型给出了 -5.0 分，则结果取 -3.0。

                # 为什么要这样设计？
                # 防止“奖励作弊”（Reward Hacking）。强化学习中的模型非常“狡猾”。如果模型发现说某句话（甚至是一些无意义的乱码）能骗过 Reward Model 拿到极高分（比如 100 分），它就会疯狂复读这句话。截断上限可以降低这种“暴利”诱惑。
                # 稳定梯度（Gradient Stability）。在 GRPO 或 PPO 中，奖励分会直接影响梯度的大小。如果某个样本的分数特别极端，它产生的梯度会非常巨大，可能一次更新就把模型参数“推下悬崖”，导致训练崩溃。
                # 平衡“规则奖励”与“神经奖励”。我们的 格式奖励（正则匹配）通常只有 0.5 或 1.0 分。如果 神经网络奖励 随便给个 10 分，那么模型就会只顾着讨好 Reward Model，而完全不在乎格式对不对了。通过 scale，我们可以确保两类奖励处于同一个量级，让模型既要“听话（格式对）”，又要“活好（内容准）”。

                if args.reasoning == 1:
                    # 判断当前是否处于推理模型训练模式。只有在这种模式下，才会去拆解 <answer> 标签。
                    # 这段逻辑展示了针对推理模型（Reasoning Model）的一种更精细的评分策略 => 加权合并“全量回答”与“纯净答案”的分数。
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    # 使用正则表达式从模型生成的完整 response 中寻找 <answer>...</answer> 块。
                    # 1. re.search: 扫描整个字符串，寻找第一个匹配项。
                    # 2. (.*?): 捕获组，提取标签中间的实质内容。
                    # 3. re.DOTALL: 确保即使答案是多行的，. 也能匹配成功。
                    if answer_match:
                        # 只有当模型规范地写出了 <answer> 标签时，才进入更精细的打分逻辑。如果没写，则保持之前的 score（全量分数）不变。
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        # 这里做了一次“偷梁换柱”。
                        # 它把原本包含 思考 + 答案 的长回复，替换成了只有纯净答案的短回复。
                        # 这样做的目的是问 Reward Model：“抛开那些华丽的思考过程不谈，这个最终结论本身值多少分？”
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6
                        # 进行加权融合。
                        # 1. 全量回答（含思考过程）占 40% 的权重。这部分负责评估思考的逻辑性、格式等。
                        # 2. 纯净答案占 60% 的权重。这占了身位上的大头，意味着模型如果思考得天花乱坠但最后答案写错了，依然会拿到很低的分数。
                        # 防止模型产生“幻觉”——思考了一堆正确的逻辑，结果最后给了一个错误的答案。

                        # 为什么要这样设计？
                        # 在训练像 DeepSeek-R1 这种模型时，存在一种风险叫 “思考漂移”，模型发现只要在 <think> 里写满看起来很专业的数学公式（即使是乱写的），Reward Model 就会给高分。
                        # 通过这样的奖励设计，就能够实现：
                        # 1. 强迫结论正确：无论思考过程多努力，如果结论（<answer> 里的内容）不能独立地通过评委的审核，分数就上不去。
                        # 2. 抑制冗余：如果思考过程（score * 0.4 那部分）充满了复读或废话，总分会受到拖累。

                reward_model_scores.append(score)
                # 将计算好的综合分数（或是加权后的，或是原始的）存入列表，准备后续与格式奖励合并。

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        # 进行数据转换，在此之前，reward_model_scores 是一个 Python List，里面存着循环里一个个算出来的浮点数（比如 [1.2, -0.5, 2.8, ...]）。这一行把它打包成了一个 PyTorch Tensor。
        # 这个 Tensor 的形状现在是 [B * G]，与之前初始化的 rewards 形状完全一致。

        rewards += reward_model_scores
        # 汇总奖励，正确性奖励（如果模型算对了，可能已经有分了） + 格式奖励（正则匹配成功，+0.5） + 标签计数奖励（四个标签都在，+1.0）。

    return rewards


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, start_step=0, wandb=None):
    # iters：一个 Epoch 里有多少个 Batch。它通常配合 epoch 用来计算全局训练步数（Global Step）。
    # ref_model, reward_model, reward_tokenizer 是更加关键的参数。

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        # prompts 是一个字符串列表，长度为 B（Batch Size）。这些是模型需要回答的问题或指令。
        # 将文本字符串转换为数值张量。
        prompt_inputs = tokenizer(prompts, 
                                  return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)  
        # input_ids: [B, P], attention_mask: [B, P]
        
        
        # return_tensors="pt"：返回 PyTorch 张量格式。如果不加默认返回的是 Python 的原生列表（List），无法直接传给 GPU 或模型进行矩阵运算。

        # padding=True：将这一批 Prompt 补齐到相同长度。同一个 Batch 里的 Prompt 长度通常不一样（比如一个 10 词，一个 20 词）。为了组成一个整齐的矩形矩阵，tokenizer 会找出这一组中最长的一个，然后把短的那些用特定的补全符（[PAD]）填满。

        # return_token_type_ids=False：不返回 token_type_ids（也叫 Segment IDs）。这是 BERT 时代的产物，用于区分“第一句话”和“第二句话”。现在的生成式模型（如 Llama, Qwen, GPT）基本不需要这个，为了节省显存和简化输入，通常设为 False。

        # padding_side="left"：关键点！在生成任务中必须从左侧填充。因为模型是从左往右生成新 Token 的，如果右侧有 Padding（占位符），会干扰模型对序列末尾的判断。
        # 1. 如果是右填充：输入看起来像 [Prompt, PAD, PAD]。模型生成时会紧跟在 PAD 后面，这会破坏模型的位置感，导致生成的回答逻辑混乱。
        # 2. 如果是左填充：输入看起来像 [PAD, PAD, Prompt]。这样模型的输入末尾始终是 Prompt 的最后一个有效字符，模型能立刻顺着接下去生成。
        # 3.在所有 Generation（生成回答）任务中，左填充是死命令。
        
        # 在这里强调一下左填充。GRPO（以及 PPO）属于在线强化学习（Online RL）。DPO 是“翻看现成的答卷”：好回答和坏回答都在硬盘里躺着呢，直接读进来算分就行。GRPO 是“现场面试”：代码运行到这一行时，并没有回答。模型必须拿着 prompt_inputs，像个活人一样思考，一个字一个字地往外蹦（model.generate）。一旦涉及 generate() 动作，右填充就会让模型在“思考”前先看到一堆无意义的空字符，从而导致思维混乱。

        # 因此，在 GRPO 中，采样必须左填充，学习可以用右填充。否则，如果用右填充，模型在生成第一个词之前，会先“撞上”那些 PAD。例如 [Prompt, PAD, PAD, (模型该在这里写答案)]。模型必须跨过这些 PAD 才能开始写字。但这会导致它在推理时，注意力机制（Attention）去关注这些本该被忽略的 PAD，从而导致生成的逻辑断裂。

        # add_special_tokens=False：不自动添加 [BOS] 等特殊字符（通常由数据集预先处理好或根据需要手动添加）。在 Im_dataset 中已经调用了 apply_chat_template。

        # ⭐️⭐️⭐️⭐️⭐️ GRPO 的训练过程确实可以拆解为两个阶段，我们可以称之为：Rollout（采样阶段） 和 Optimization / Update（优化阶段）。Rollout 是串行的，Optimization 是并行的 NTP。
        
        if args.max_seq_len:
            # Prompt 长度截断。如果 Prompt 太长，只保留最右侧（最新）的 args.max_seq_len 个 Token。
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            # 这里的 attention_mask 就是 Padding Mask。
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        with torch.no_grad(): # 强制在此上下文块内不构建计算图，即不记录梯度信息。
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            # DDP 模型需要使用 .module 访问 generate 方法。
            outputs = model_for_gen.generate(
                # 这里直接调用了 generate，说明 model 继承了 HuggingFace 的基类。默认情况下 model.generate 在 PyTorch 环境下返回的是一个 LongTensor，里面全是被分词后的 Token ID（整数）。
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)  # [B*num_gen, P+R]

            # **prompt_inputs：展开包含 input_ids 和 attention_mask（左填充）的字典。
            # max_new_tokens=args.max_gen_len：设定生成的硬上限，防止模型变成“话痨”没完没了。
            # do_sample=True, temperature=0.8：do_sample=True：开启随机采样，而非贪婪搜索。temperature=0.8：控制多样性。数值越高，生成的回答越天马行空。这是 GRPO 的生命线，因为只有回答各不相同，组内才有“相对优劣”可以比较。
            # num_return_sequences=args.num_generations：GRPO 的标志性参数。对于 Batch 里的每一个 Prompt，模型会产生 G 个独立样本。如果 B=4, G=8，这一步会生成 32 条轨迹。这里的参数表示了针对每一个输入的 Prompt，要求模型返回多少个独立的生成序列。
            # pad_token_id=tokenizer.pad_token_id：告诉模型生成的过程中用什么来做补位。为什么这里需要 Padding？核心是因为模型在采样时，不同的回答长度是不一样的。
            # - 回答 1：写了 50 个词，遇到了 EOS（结束符），停下了。
            # - 回答 2：写了 200 个词才遇到 EOS。
            # 但是GPU 的张量（Tensor）必须是矩形的（每一行长度必须一致）。因此当回答 1 提前停下时，为了保持 Batch 矩阵的整齐，后续的位置就会被填上 pad_token_id。这样，所有 B * G 个回答都能塞进一个形状为 [B*G, Max_Length] 的大矩阵里。

            # 最后生成的 outputs 是一个 [B * num_gen, Prompt + Response] 的巨大张量，output 已经被 token 化了。
            # ⭐️⭐️⭐️⭐️⭐️ 需要注意的是，model_for_gen.generate() 调用后是会做 Padding 补齐的。
            # 在一个 Batch 中，假设 num_generations=4。
            # 回答 1 只有 50 个 Token。
            # 回答 2 有 200 个 Token。
            # 回答 3 触发了 max_new_tokens（比如 512），还没说完。
            # 为了让这 3 个回答能躺在一个形状为 [4, P + R] 的 Tensor 里，PyTorch 必须把所有回答都撑开到和最长的那一个（或者 P + max_new_tokens）一样长。那些多出来的空间，就会被填入我们代码里指定的 pad_token_id。

        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]
        # 从模型生成的完整序列中，精准切除掉开头的 Prompt 部分，只保留模型自己“写”出来的回答内容（Completion）。为什么要这样做？
        # 1. 为了给奖励模型（Reward Model）打分。奖励模型通常只需要评估模型给出的“答案”质量。如果不把 Prompt 切掉，奖励模型可能会被问题本身的特征（比如问题的长短、难易）干扰，无法纯粹地给答案打分。
        # 2. 为了计算 KL 散度（Constraint）。在 GRPO 更新阶段，我们需要对比 Policy 模型和 Reference 模型对“回答部分”的概率分布。Prompt 是固定的（概率为 1），只有 Completion 部分的概率是随模型参数变化的。如果不切除，计算 KL 散度时会包含大量无意义的零值。
        # 3. 为了计算 Advantage（优势值）。GRPO 是根据模型生成的不同 结果（Answer）来计算相对优劣的。通过只保留 completion_ids，我们可以更方便地统计每个答案的生成长度、计算其 log 概率。
        
        def get_per_token_logps(mdl, input_ids, n_keep):
            # 计算生成文本中每个 Token 的对数概率（Log-Probability）。
            # 在 GRPO 或 PPO 这种强化学习算法中，我们需要知道模型生成某个词的“确定程度”（即概率）。

            # mdl 当前的语言模型。
            # input_ids 完整的 Token ID 序列（包含 Prompt 和生成的 Answer）。
            # n_keep 这是一个长度值，表示我们要计算多少个 Token 的概率（通常等于生成的 Answer 的长度）。

            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            # 如果 input_ids 处于 PyTorch 的“推理张量”模式（Inference Mode），直接操作会报错，所以需要先 detach() 出来并 clone() 一份。
            # 如何理解 input_ids.is_inference()？
            # is_inference() 返回一个布尔值，判断这个张量是否是在 with torch.inference_mode(): 上下文内创建的。inference_mode 是比 no_grad 更激进的优化。在推理模式下的张量：
            # 1. 不记录梯度（类似 no_grad）。
            # 2. 不保留版本计数（Version Counter）：这使得它在内存操作上更快。
            # 3. 限制多：我们不能对推理模式下的张量直接进行某些“原地（In-place）”操作，也无法直接将其带入需要计算梯度的训练环节。
            # 因为接下来的 mdl(input_ids, ...) 是在训练环节（Optimization）调用的，代码需要确保 input_ids 已经“还俗”（脱离推理模式），变成一个普通的、可以参与后续计算的 Tensor。
            
            # input_ids 的 shape 是 [B * G, P + R]

            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :] # ⭐️⭐️⭐️⭐️⭐️ 这是一个 NTP 的过程。
            # 进行前向传播，这里的 logits 存储了每个位置的逐 token 词表数值。
            # * logits_to_keep=n_keep + 1：这是一个优化参数。我们不需要算整个长序列的 Logits，只需要计算最后那一段（Answer 部分）的。具体实现可以去看 model_minmind.py 部分，其实就是在 self.lm_head 之前对 hidden_state 做截断，只计算 Response 部分的 logits。
            # * [:, :-1, :]：这是之前讨论过的 Shift 操作。因为模型在位置 i 的输出 Logits 是用来预测位置 i + 1 的 Token 的。为了对齐，我们需要把 Logits 整体往左挪一位。

            # 最终出来的 logits 的 shape 是 [B*G, R, Vocab_Size]。

            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                # 遍历 Batch 中的每一行。ids_row（其实就是 input_ids）此时只取最后 n_keep 个词，也就是我们真正关心的生成内容。
                # 其中 logits_row 的 shape 是 [R, Vocab_Size]，ids_row 的 shape 是 [R]。
                # 最终 ids_row 有点类似于 label，在计算 Log-probs 时，我们把 input_ids（模型自己生成的词）当成了目标索引，去 logits 矩阵里查找。
                
                # 但在本质上并不同，在 SFT 中，Label 是“正确答案”，我们无条件地希望模型去模仿它（提高其概率）。在 RL 中，这些 input_ids 只是模型的一次尝试。我们并不急于提高它的概率，而是先看它拿到了多少 Reward。
                # 1. 如果 Reward 高：我们把它当成“好的 Label”，增加它的概率。
                # 2. 如果 Reward 低：我们反而要减小它的概率。

                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                # 防御性编程。

                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), dim=1, index=ids_row.unsqueeze(1)).squeeze(1))
                # torch.gather 的用法可见 dpo，具体来说就是 =>

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
                
                # 结果：
                # [ [10], [60] ]

                # 因此这里就是按照 ids_row 寻找索引，然后从 logits_row.log_softmax(dim=-1)（那个大矩阵）里取的正确 token 的输出概率。

            return torch.stack(per_token_logps) # shape 为 [B*G, R]。
            # torch.stack 的作用是“增加一个新维度，并把一组张量整齐地堆叠起来”。
            # ⭐️⭐️⭐️⭐️⭐️ 这个值表示在当前模型看来，在已知前面所有词的情况下，写下『这个特定词』的可能性（对数概率）有多大。这里的已知前面所有词的情况下非常重要，因为 logits 就是这么来的，他是 NTP 获取的。

            # 为什么不用 torch.cat 而是 torch.stack？
            # torch.cat（拼接）：是在已有维度上接龙。如果我们对三个 [10] 的张量用 cat，我们会得到一个长度为 30 的一维长向量。
            # torch.stack（堆叠）：是开辟一个新维度。我们会得到一个 3x10 的二维矩阵。
            # 在 GRPO 中必须用 stack，因为我们需要保持 [Batch_Size * G, Sequence_Length] 这种结构，这样后续才能方便地与 Advantage（优势值）进行按行相乘。

            # torch.stack 要求列表里的所有张量形状必须完全一致。这就是为什么之前讨论的 pad_token_id 和补位逻辑如此重要。
            # 如果生成的 3 个回答长度不一样（一个是 8 token，一个是 10 token），stack 就会报错。正是因为我们在生成时进行了 Padding，确保了所有生成的 ids_row 长度都是一样的（例如都是 max_gen_len），这里才能顺利地一键堆叠。

        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
            # 利用 get_per_token_logps，计算当前模型生成这些回答中每一个 Token 的 logP。
            # outputs 是 [Prompt + Answer]。
            # completion_ids.size(1) 就是 R。
            # per_token_logps 的 shape 就是 [B*num_gen, R]。
            # 这是典型的 并行 NTP 计算。模型通过一次前向传播（Forward Pass），拿到了对自己刚刚生成的 B * G 个回答的概率评分。
            res = model(outputs) if lm_config.use_moe else None
            # 如果是 MoE 模型（如 DeepSeek-V3/R1, Mixtral），需要执行一次完整的 forward 来获取辅助信息。在 MoE 架构中，每个 Token 会由 Router（路由器）分发给不同的 Expert（专家）。为了防止模型偷懒（只用其中一两个专家，导致其他专家没练出来），需要记录专家的使用情况。

            # 这里的 aux 其实可以在 get_per_token_logps 中的 model 前向传播时获取的。
            # 此外，两次前向传播，MoE 的 aux_loss 能保证一样吗？
            # 这是一个非常关键的问题。结论是不一定完全一样，但由于在 Optimization 阶段关闭了随机性，它们是极度趋同的。
            # - 随机性控制：在生成阶段（Rollout），模型会采样（Sampling）。但在现在这个优化阶段（Optimization），模型是在计算给定序列的概率，此时不会进行随机采样。
            # - Dropout：如果模型开启了 Dropout，两次前向传播确实会不同。但是，在 RLHF 训练中，通常会调用 model.train()。为了避免这种不一致，规范的做法是在这个 Block 里确保所有的随机因子（如路由器的微小噪声）在相同的 seed 下或通过特定机制保持稳定。
            # - 深度思考：即便有微小差异，对于 aux_loss 这种统计性质的全局 Loss 来说，这种波动在梯度更新中是可以被接受的（类似一种正则噪声）。
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        with torch.no_grad():
            # 强制关闭梯度计算。
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B * G, R]，与刚才 model 算出的 per_token_logps 形状完全一致。
            # 计算原始模型对刚才生成的这些回答的“看法”，即在已知前面所有词的情况下，写下『这个特定词』的可能性（对数概率）有多大。
            # per_token_logps 是“现在的我”对这段话的看法；ref_per_token_logps 是“过去的我”对这段话的看法。

            # ⭐️⭐️⭐️⭐️⭐️ 为什么说 get_per_token_logps 就是在计算 PPL？
            # 因为 get_per_token_logps 返回的是每一个位置的 \log P(x_i | x_{<i})。因此，它算出了 PPL 公式里 sum 符号后面的那个核心项。我们只需要对这个函数的结果取负号、求平均、再取指数，得到的就是标准 PPL。
            # 什么是困惑度的物理意义？
            # PPL 衡量的是：模型在看到上文后，对下文出现的这个词感到多么“意外”。
            # - 如果 log_p 很大（接近 0），概率就高，模型觉得“意料之中”，PPL 低。
            # - 如果 log_p 很小（负得厉害），概率就低，模型觉得“大吃一惊”，PPL 高。

            # 一个形象的比喻是：
            # 想象模型在玩“看开头猜词”的游戏：
            # - Prompt: "床前明月"
            # - 实际生成的词: "光"
            # get_per_token_logps 的工作就是翻开模型的“备选概率表”，找到“光”这个词，看看模型给它打了多少分。
            # - 如果模型给“光”打了 0.99，它就很淡定（PPL 低）。
            # - 如果模型给“光”打了 0.01，它就很困惑（PPL 高）。

            # 在 LLM 领域，当我们谈论“计算某个句子的 PPL”时，代码实现通常就是这个函数的逻辑。

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # 将模型生成的“数字序列”（Token IDs）翻译回人类能读懂的“文本字符串”（Text）。
        # 对 input_ids 去做批量解码，由 token id 转为 token string，skip_special_tokens=True 表示跳过特殊标识。在生成的过程中，模型可能会吐出一些控制字符，比如 <|endoftext|>、<pad> 或 <|im_end|>。设置这个参数可以把这些“技术性废话”过滤掉，只留下纯净的文本内容。

        # 需要注意的是，completion_ids 的 shape 是 [B*num_gen, R]，但是进行了 batch_decode 之后，shape 就变成了 [B*num_gen]。
        # 这个过程就相当于 R 个 id 组合成了一个字符串，如下所示，因此 R 这一维度消失了 =>
        # tensor([[ 70,  80,  90],    # 第一条回答的 ID
        #         [100, 110, 120]])   # 第二条回答的 ID
        # [
        #     "I like cats",          # 这一个字符串包含了原来的 3 个 ID 对应的意思
        #     "Java is good"          # 这一个字符串包含了原来的另外 3 个 ID
        # ]

        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]
        # prompts 是输入给模型的问题，completions 是模型输出的回答。
        # 这里的 rewards 是最终拿到的奖励，包括了正确性奖励 + 格式奖励 + 标签计数奖励。

        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        # 将一维的 [B*G] 奖励重塑为二维的 [B, G]，每一行代表一个问题，每一列代表针对该问题的不同生成结果。

        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 说明一下 repeat_interleave 作用，假设原始 tensor 为 torch.tensor([A, B])。
        # 那么 repeat_interleave 对应的结果是 [A, A, A, B, B, B]。
        # repeat 对应的结果是 [A, B, A, B, A, B]。
        # 所以这里的 mean_r 是 dim=1 维度 mean，然后每个 Prompt 都复制 args.num_generations 个 reward 均值，std_r 表示标准差。
    
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        # 实现组内优势计算。
        # (rewards - mean_r)：计算每个回答相对于同组平均分的偏差。如果结果 > 0：说明这个回答比同组其他回答好（优等生）；如果结果 < 0：说明这个回答比同组其他回答差（差生）。
        # / (std_r + 1e-4)：除以组内标准差。这是一种“缩放”，目的是让不同难度的题目具有可比性。简单题的话，大家分都很高，标准差小，除以它会放大微小的差异。难题的话，大家分都很乱，标准差大，除以它会缩小分差的影响。
        # torch.clamp(..., -10, 10) 是安全阀。防止某些组因为所有回答分都一样（标准差趋于 0）导致计算出无穷大的优势值。

        # advantages 的 shape 是 [batch * G]。

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]
        # 这里的核心目的是实现“跨题目”的公平性修正。
        # 问题 A (简单) => G 个回答分数为 [0.9, 0.8, 0.7]，均值 0.8。第一步后，0.7 变成了负分。
        # 问题 B (极难) => G 个回答分数为 [0.1, 0.05, 0.0]，均值 0.05。第一步后，0.0 变成了负分。
        # 如果没有第二行，由于这两组的内部差异（方差）可能完全不同，导致问题 A 的负反馈力度是 -2.0，而问题 B 的负反馈力度是 -0.5。
        # 这就会导致模型会因为简单题的分差大，就拼命去改简单题的参数，而忽略了难题。有了第二行（全局 Norm），它把全 Batch（所有问题）的优势值再次拉回到标准的 (0, 1) 分布。它确保了不管题目难易，只要是“差生”，在全局梯度更新中的“杀伤力”是一样大的。 这就是我之前提到的“让学习率在所有题目面前保持公平”。

        # 这里的除标准差要着重介绍一下。除以标准差的工程本质是剥夺奖励函数对梯度大小的控制权，将控制权重新交回到学习率（Learning Rate）手中。不然高奖励会带来高梯度，影响学习率对更新幅度的控制。
        # 总而言之，这行代码是给优化器（Optimizer）看的。优化器最喜欢的输入就是均值为 0、方差为 1 的梯度信号。这能让 Adam 等优化器的动量（Momentum）累积得更加平滑。

        # 还有哪些量需要进行均值为零，方差为 1 的处理呢？
        # 1. 神经网络的输入特征 (Input Scaling)。
        # 2. 隐藏层的激活值 (Hidden Activations)，LayerNorm。
        # 3. 强化学习中的“状态价值” (Value Function / Critic) 以及奖励输出（Reward）。
        # 4. 损失函数中的梯度 (Gradient Clipping / Scaling)。
        # 5. 多模态对齐中的 Embedding (如 CLIP)。
        # 6. 权重初始化，kaiming 初始化。

        # ⭐️⭐️⭐️⭐️⭐️ 整个深度学习领域，进行如上标准化的核心目的就是为了防止梯度数值太大，从而影响学习率对梯度更新的控制。从数学公式看：\Delta w = \eta \cdot \nabla L（其中 \eta 是学习率，\nabla L 是梯度）。如果某个 Batch 的梯度因为输入没对齐或奖励没标准化，突然飙升到 1,000,000。即便我们把学习率 \eta 设得很小（如 10^{-5}），最终的更新步长 \Delta w 依然高达 10。
        # ⭐️⭐️⭐️⭐️⭐️ Grad Clipping 就是进行梯度缩减的最后保底手段！
        # 现在来说明一下为什么标准化能够降低梯度，也就是说，梯度 \nabla L 是什么算出来的？为什么标准化会影响他的数值？假设我们有三层网络：
        # 第一层（隐藏层 1）：$z = w_1 \cdot x + b_1$
        # 第二层（隐藏层 2）：$y = w_2 \cdot z + b_2$ （注意：这里的输入是上一层的输出 $z$）
        # 第三层（输出/损失）：$L = w_3 \cdot y + b_3$ （这里计算最终的 Loss）

        # 那么如果我们要更新 w_1，那么 L 对 w1 求导就是如下等式：
        # \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_1}
        # 最后的结果就是 x * w2 * w3。
        # ⭐️⭐️⭐️⭐️⭐️ 梯度的本质就是“路径上所有权重的连乘”再乘以“最底层的输入 x”。
        # 所以这就是影响梯度大小的关键——x，在训练的某一瞬间，权重 w 是固定的，我们唯一能动的手脚，就是数据的分布。
        # x 是输入。如果 x 很大，它和 w 一乘，再加上链式法则的连乘效应，梯度会像海啸一样拍过来，让我们对 x 做了标准化之后，它的数值就会比较小了，从而梯度数值就可控了。
        
        # ⭐️⭐️⭐️⭐️⭐️ 深度学习的“三位一体”控制，为了保住**学习率（LR）**的指挥权，工程师们构筑了三道防线：
        # 1. 初始化 (Initialization)：让初始的 $w$ 尽量靠近 $1$（比如 Xavier 或 Kaiming 初始化）。
        # 2. 标准化 (Normalization)：让每一层的输入 $x$ 始终保持在 $1$ 左右。（这就是我们发现的重点）
        # 3. 激活函数 (Activation)：让梯度的传导率 $\sigma'$ 也在 $1$ 附近。
        # 这三者合力，才让那个连乘公式 $w_n \dots w_1 \cdot x$ 不至于变成天文数字。
 
        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
        # 在所有的生成 Token（completion_ids）中，寻找哪个位置是 EOS（结束符）。
        # 生成一个和回答区域等大的 布尔矩阵。如果是 EOS，该位为 True；其他位（文字或 Padding）为 False。

        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        # 创建一个长度等于 B * G 的向量，初始值全部设为最大长度 R。

        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 假设我们的最大生成长度 R = 8，Batch 中有 3 个样本。
        # 样本 A：回答很短，在索引 3 处就说完了（输出了 EOS）。
        # 样本 B：回答较长，在索引 7 处输出了 EOS。
        # 样本 C：模型是个“话痨”，直到最后（索引 7）都没输出过 EOS。
        # 那么 eos_idx 的数值结果会是 tensor([3, 7, 8], dtype=torch.long)

        # 这行代码的详解为：
        # is_eos 是一个形状为 [Batch, R] 的布尔矩阵（True 代表该位置是 EOS）。
        # is_eos.any(dim=1)，在每一行（dim=1）检查是否存在任何一个 True。得到一个长度为 Batch * G 的布尔向量。True 表示这个样本在生成过程中输出了 EOS，它提前下班了。False 表示这个样本直到最大长度都没输出过 EOS，它还在“加班”。
        # is_eos.int().argmax(dim=1)，.int() 把 True/False 变成 1/0，.argmax(dim=1) 在寻找每一行中最大值（即第一个 1）出现的下标。如果一行里有多个 EOS（虽然少见但可能），argmax 永远只返回第一个。这非常重要，因为第一个 EOS 之后的所有内容都是无效的。is_eos.int().argmax(dim=1) 的一个示例是
        # is_eos.int() 的返回值如下 =>
        # [[0, 0, 1, 0, 1, 0],
        #  [0, 0, 0, 0, 0, 0],
        #  [1, 0, 0, 0, 0, 0]]
        # argmax 的逻辑是 “返回每一行中，数值最大的那个元素第一次出现的索引。”
        # 在这个 0/1 矩阵中，最大值就是 1。
        # 行 0：最大值是 1，出现在索引 2 和 4。argmax 取第一个，结果是 2。
        # 行 1：全都是 0，最大值是 0。argmax 取第一个 0 出现的索引，结果是 0。（注意：这是个误区！全 0 行也会返回 0）
        # 行 2：最大值是 1，出现在索引 0。结果是 0。
        # 最终结果就是 tensor([2, 0, 0])。

        # 需要确定清楚 shape，is_eos 是 [B*num_gen, R]，eos_idx 是 B*num_gen，is_eos.any(dim=1) 是 B*num_gen，is_eos.int().argmax(dim=1) 也是 B*num_gen。
        # eos_idx[is_eos.any(dim=1)] 那些没有生成 EOS 的位置（对应的布尔值为 False），在这行代码里会被完全跳过，保持它们原来的初始值（即最大长度）。只有生成 EOS 位置的 eos_idx[i] 会被取出来被赋值。
        # is_eos.int().argmax(dim=1)[is_eos.any(dim=1)] 也是一样的，同样基于布尔值进行过滤，只取出那些真正有 EOS 的行对应的索引值，用于赋值。

        # 伪代码如下，对于 Batch 里的每一个样本 i：
        # if 样本_i_中出现了_EOS:
        #     把样本_i_的结束位置(eos_idx[i]) 更新为 第一个_EOS_出现的真实位置
        # else:
        #     保持样本_i_的结束位置(eos_idx[i]) 不变（默认是最大长度 R）
        
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B*num_gen, R]
        # torch.arange(is_eos.size(1), device=args.device) 在生成一个从 0 到 R-1 的整数序列。假设最大长度 R=5，它生成 [0, 1, 2, 3, 4]。这是一把标尺，代表每一个 Token 在回答中对应的位置编号。

        # .expand(is_eos.size(0), -1) 用于把这把刻度尺复制给 Batch 里的每一个回答。如果 BatchSize 为 3，它变成一个 3 * 5 的矩阵，
        # [[0, 1, 2, 3, 4],
        # [0, 1, 2, 3, 4],
        # [0, 1, 2, 3, 4]]
        # 下面是整个处理流程：
        # 1. 制造一把尺子 [0, 1, 2, ..., 511] (Shape: [512])
        # 2. 调用 .expand(16, -1)
        # 3. 系统自动推导为 .expand(16, 512)
        # 4. 结果：16 把同样的尺子整齐排开 (Shape: [16, 512])

        # eos_idx.unsqueeze(1)，之前 eos_idx 是 [B*G]，在这行之后就是，[B*G, 1]。
        # <=（以及所有的比较运算符如 ==, >, !=）在 PyTorch 中都会触发自动广播机制（Broadcasting）。因此 eos_idx.unsqueeze(1) 会转变为 [B*num_gen, R]，然后进行比较。
        # 最后产出的内容示例为：
        # [[1, 1, 1, 0, 0],  <-- 只有前3个词参与梯度更新
        # [1, 1, 1, 1, 1],  <-- 全部参与梯度更新
        # [1, 1, 0, 0, 0]]  <-- 只有前2个词参与梯度更新

        # 这行代码把一个抽象的索引数字（比如“在位置 2 结束”），具象化成了一个物理的开关矩阵。
        # 有效区 (1)：这些 Token 会携带我们辛苦算出来的、标准化后的 Advantage 信号去更新模型参数。
        # 屏蔽区 (0)：这些是 EOS 之后的 Padding 杂讯。通过乘以 0，它们的梯度被物理性掐断。

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]

        # 在计算 KL 散度的时候，是逐个 token 计算的，一样地，在蒸馏 loss 的 KL 散度计算中，也是逐 token 的。
        # 但是和蒸馏的 KL 散度不一样的是，ref_per_token_logps 的 shape 是 [B*num_gen, R]，而蒸馏的 shape 是 B*num_gen, R, vocab_size]。
        # 这是因为前者只关注一个点，而后者关注一个面。
        # 强化学习（RL/GRPO）中的 KL 只关注模型实际采样出（写出来）的那个 Token 的概率，是一个“点”对“点”的对比。它衡量的是：“我们写这个词的概率，比老模型变大了多少？”
        # 蒸馏（Distillation）中的 F.kl_div，关注的是整个词表（Vocab）上所有可能词的分布。计算的是“向量”对“向量”的对比。它衡量的是：“我们对‘苹果’、‘香蕉’、‘手机’等所有词的预测概率，是否和教师模型一模一样？”

        # 为什么蒸馏的 KL 散度会和 GRPO 的 KL 散度有这样的差异？核心是优化的目的不同：
        # 1. RL 关心的是决策的后果。既然我们已经选了这个词，我只需要看这个词的概率变动，以及这个词带来的奖励（Advantage）就够了。所以它是一个“点”。
        # 2. 而在蒸馏中，我们希望学生模型全盘继承教师的思维方式。教师觉得“苹果”和“梨”很像，这种“类比能力”就藏在非正确答案的概率分布里。如果我们只看“选中的那个点”，就会丢失教师模型最精华的“长尾知识”。所以它必须是一个“面”。
       
        # 本质上，这行代码并不是严格的 KL 散度，他只是 KL 的采样。
        # 严格的 KL 散度定义是 KL(P \| Q) = \sum_{v \in \text{Vocab}} P(v) \log \frac{P(v)}{Q(v)}
        # 上述代码实现仅仅是 \log P(\text{selected}) - \log Q(\text{selected})。这只是在当前采样到的那个 Token 上，两个模型对数概率的差值。
        # 因此，这行代码本身确实不是完整的 KL 散度，它只是计算 KL 散度的一个“中间组件”或者说“采样估计项”。

        # 既然它只是个“偏移量”，为什么代码注释和论文都管它叫 KL 约束呢？这里利用了一个蒙特卡洛采样（Monte Carlo Sampling）的技巧。
        # 1. 采样即分布：当我们按照当前模型的分布去生成（Sample）Token 时，我们其实已经在“模拟”那个求和符号 \sum 了。
        # 2. 期望等价：在数学期望上，对采样到的 Token 计算 log-ratio 的均值，在无穷多次采样下，等于对整个词表计算完整的 KL 散度。所以，这行代码其实是“用单点采样的偏移量，作为对整体 KL 散度的无偏估计。”

        # 有一个形象的例子。
        # 1. 全量 KL（上帝视角）：我们把餐厅菜单上所有的菜（词表 V）都吃一遍，对比新老厨师对每一道菜评分的差异。这很累，耗时耗力。
        # 2. 采样估计（单点偏移）：我们让厨师随机给我们上几道他最擅长的菜（模型采样的 Token）。我们只品尝这几道菜，看看新厨师做得比老厨师好多少。核心逻辑是只要我们点的菜足够多，或者这些菜确实代表了厨师的风格（概率分布），我们通过这几道菜得到的“平均评价差距”，就等于对整个餐厅水平差距的无偏估计。

        # 理论上，KL 散度永远是非负的（$\ge 0$）。但上述实现会让 KL 散度变成负数，如果我们直接用 kl_div = ref_logps - per_token_logps。当新模型的概率比老模型大时，kl_div 是负数；当新模型的概率比老模型小时，kl_div 是正数。
        # 为了解决上面的问题，这行代码引入了一个巧妙的函数。令 $x = \text{kl\_div}$。
        # 1. 始终非负：根据数学不等式 $e^x \ge x + 1$，这个函数 $e^x - x - 1$ 的值永远 $\ge 0$。
        # 2. 最小值在原点：当 $x=0$（新旧模型完全一致）时，结果恰好为 $0$。
        # 3. 对称的惩罚：如果 $x$ 是正的（偏离了），它会增长。如果 $x$ 是负的（也偏离了），$e^x$ 虽然减小，但 $-x$ 会让整体变大。不管我们是变激进还是变保守，只要我们偏离了老模型，这行代码就会产生一个正向的惩罚值。

        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)  # [B*num_gen, R]
        # 这行代码其实就是在定义 GRPO 的损失函数。
        # torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) 是奖励项。torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) 整体的 shape 是 [B*num_gen, R]，这里发生了广播。

        # 下面解释一下 per_token_logps - per_token_logps.detach()
        # 它计算的是重要性采样比率 (Importance Sampling Ratio)，衡量这个 Token 的概率相对于刚才采样时，是变大了还是变小了。
        # 需要关注的是 per_token_logps - per_token_logps.detach()，这里两个都是 per_token_logps。这里没有写错代码。这行代码处理的是 “新旧策略的重要性采样比例”，而不是 “模型与参考模型的偏差”。
        
        # ⭐️⭐️⭐️⭐️⭐️ 第一个问题是，这里为什么要用 .detach()？如果不加 .detach()，这行代码在数学上会直接失效，在工程上会导致梯度归零。让我们看看如果不加 .detach()，公式会变成什么样：
        # \text{ratio} = \exp(\text{per\_token\_logps} - \text{per\_token\_logps})
        # \text{ratio} = \exp(0) = 1
        # 在计算图中，如果我们不 detach，分子和分母其实是同一个变量。当我们对这个“1”求导时，常数的导数永远是 0。结果就是无论 advantages 有多大，1 * advantages 的导数依然是 0。我们的优化器接收不到任何更新信号，模型参数纹丝不动。
        # 如果我们进行了 .detach()，那么从数值（Value）上看，在计算那一刻它们确实是一模一样的；但从计算图（Computational Graph）的角度看，它们是截然不同的两个东西。
        # 既然数值一样，减出来不是 0 吗？exp(0) 不是 1 吗？关键就在这里。
        # 在反向传播时，PyTorch 看到的公式是：
        # \text{Ratio}(\theta) = \exp(\text{LogProb}(\theta) - \text{Constant})
        # 前向传播（Forward）状态下，的确 \exp(\text{数值} - \text{数值}) = 1，所以这一项的初始值确实是 1。反向传播（Backward）时，当我们对 \theta 求导时，
        # \frac{d}{d\theta} \exp(\text{LogProb}(\theta) - C) = \exp(\dots) \cdot \frac{d}{d\theta} \text{LogProb}(\theta)。
        # 结论就是因为左边的 LogProb 是带梯度的，所以导数不是 0！这个导数会带着 Advantages 的信号，告诉模型：“喂！虽然现在比率是 1，但如果我们往这个方向改参数，比率就会变大，我们就能拿到更多分！”

        # 第二个问题是什么是重要性采样？为什么这里要计算重要性采样？
        # 首先我们要知道 old 模型在每个训练步（Step）都会随着当前模型的更新而“同步”变动，这和 DQN 这种定期更新模型的 RL 算法是不一样的（⭐️⭐️⭐️⭐️⭐️ DQN 是将 old model 的输出当做 label，这里的并不是将 old model 的输出当做 label 因此是不一样的，所以为什么这里要用重要性采样而不是 MSE？因为它不是在规定“什么是对的”（Label），而是在规定“怎么利用旧的经验去尝试更好的方向”）。

        # 其次回答什么是重要性采样？
        # 在强化学习训练中，采样（Rollout）是非常耗时的（需要大模型推理）。每次梯度更新（New Model 变了一点点）之后，按理说旧数据就“作废”了，因为分布变了。但我们不重新采样，而是对比 New Model 和 Old Model 对同一个动作的“看法”（概率）：
        # Ratio = \frac{\text{New Model 产生该动作的概率}}{\text{Old Model 产生该动作的概率}}
        # 如果 Ratio > 1：说明 New Model 比 Old Model 更看好这个动作，我们要加大这个动作带来的奖励影响。如果 Ratio < 1：说明 New Model 没那么看好它，我们要缩小影响。
        # 这里的 Ratio 就是重要性采样，用来依据旧模型的数据估计新模型的表现。

        # ⭐️⭐️⭐️⭐️⭐️ 然后是为什么要有重要性采样，一是为了让 Advantage 直接接管梯度的大小，让 Reward 对梯度的控制权最大。二是为了弥补优势出自于旧模型的输出，需要计算一个比例对优势进行新模型参数下的兼容计算。
        # 但是目前由于 GRPO 的实现是“采样一次，更新一次”，因此目前第二个目的没有意义，只有第一个目的在起作用。

        # per_token_logps - per_token_logps.detach() 就是重要性采样的实现。对应的数学表达式是：
        # \log(\pi_{\theta}) - \log(\pi_{\theta_{\text{old}}})
        # 套用对数公式：\log \left( \frac{\pi_{\theta}}{\pi_{\theta_{\text{old}}}} \right)
        # 现在，我们对整个式子取指数 torch.exp(...)：\exp \left( \log \left( \frac{\pi_{\theta}}{\pi_{\theta_{\text{old}}}} \right) \right)
        # 因为 \exp 和 \log 是互逆运算，它们抵消了，最终结果就是：text{ratio} = \frac{\pi_{\theta}}{\pi_{\theta_{\text{old}}}}

        # ⭐️⭐️⭐️⭐️⭐️ 关于 GRPO 及其变体 GSPO、DAPO，这里有一篇好文 https://zhuanlan.zhihu.com/p/1937513263362471465。
        # 依据这篇文章和 Gemini，我们有如下几点结论：
        # 1. GRPO 的奖励评分通常是“序列级”的，但它的损失函数（Loss）和梯度更新是“逐 token”的，计算出来的优势是标量，他会为梯度做加权。
        # 2. 为了防止策略更新步长过大，GRPO 的损失函数引入了 clip 操作。当比率触发截断时，损失函数的值变成了一个常量，常量的导数是 0。结果就是既然导数为 0，那么在反向传播时，梯度 \nabla_{\theta} L 也就变成了 0。这意味着，尽管这个 token 产生了损失值，但它不会对参数 \theta 的更新产生任何“推力”。但因为在 token 维度计算 Loss 的，因此 clip 只是丢掉了某个 token 的奖励信号。
        # 3. token 的奖励信号在数学实现上，它是在鼓励每一个 Token 的生成或者降低，但在宏观效果上，它是在鼓励这个序列的生成或者降低。

        # ⭐️⭐️⭐️⭐️⭐️ 但这里面其实是有问题的，由于 GRPO 的奖励一般都是序列级的，因此 GRPO 默认认为，一个好结果里的每个 token 都是有功的，坏结果里的每个 token 都是有过的（⭐️⭐️⭐️⭐️⭐️ 这其实就是信度分配问题）。
        # 这使得如果整个序列的 \hat{A}_i > 0（表现优于平均），那么该序列中所有的 token 都会受到正向激励，通过梯度上升调高它们的出现概率；如果一个很长的推理链，前面 90% 都对，只有最后一步写错了，导致最终 r_i 极低。在 GRPO 看来，前面那 90% 正确的推导也会被惩罚。
        # 再比如，“1+1=2”这个推理链，如果模型写出了正确的逻辑，但中间夹杂了一个废话 token（比如“嗯...”）。
        # 结果就是在 GRPO 看来，这个“嗯...”也对拿高分有功，所以模型也会被鼓励以后多说“嗯...”。
        # 这也就是为什么强化学习需要极大的样本量（Group Size）。在成千上万次的采样中，正确的第 1 步可能会出现在很多不同的序列里。有的序列最终对了（受到奖励），有的最终错了（受到惩罚）。通过统计意义上的平均，那些真正有用的 Token 最终会获得正向的净梯度，而那些“凑数”的 Token 信号则会互相抵消。

        # ⭐️⭐️⭐️⭐️⭐️ 这也是为什么奖励函数可以控制模型的表现。一是因为 Advantage 能够控制 token 是要鼓励还是抑制，如果 A > 0，那么就要鼓励，如果 A < 0，那么就要抑制。而是因为 Advantage 还给出了梯度强度，如果奖励函数设计得非常激进，那么 advantages 的数值就会非常大。这个巨大的数值会像乘法因子一样，直接作用在梯度上。

        # ⭐️⭐️⭐️⭐️⭐️ 此外，clip 和 min 会有一个非对称性质。那么在 A > 0 的时候，r 可以取得无限小，A < 0 的时候，r 可以取得无限大。
        # 反应在 loss 中就是，如果这个 token 有好处，但是 r 很小，那么鼓励的程度微乎其微，这代表新模型认为旧模型不该生成这个 token；如果这个 token 有坏处，但是 r 很大，那么就会强烈的抑制这个 token 的生成，这代表新模型比旧模型更想生成这个坏 token。
        # 所以 clip 操作其实是：
        # 1. 所以当优势为正时，只有增幅被限制，增幅限制由 e_high 实现。
        # 2. 负优势动作的减幅被限制，减幅限制由 e_low 实现。

        # - args.beta * per_token_kl 是约束项。
        # 在强化学习问题中，有一个臭名昭著的问题是 Reward Hacking。大模型非常聪明，如果它发现输出一串乱码或者特定的特殊字符能骗过奖励模型（RM）拿到高分，它会毫不犹豫地抛弃人类语言，进化成一个“复读机”或“乱码生成器”。
        # 而这里的 KL 散度则能够保证模型不会因为追逐高分而变得不像人话。
        # 这里的 KL 散度计算是发生在 Token 级别，这意味着每一个生成的词都会受到约束。beta 这是一个超参数（系数）。\beta 越大，模型就越“保守”，越不敢偏离原始模型；\beta 越小，模型就越“激进”，可能探索出更强的逻辑，但也更容易崩坏。
        # 如果没有 KL 项，模型的分布会迅速坍缩。坍缩后果就是模型会发现某一个答案组合分最高，从此以后无论我们问什么，它都只吐出那一个标准答案，丧失了生成的多样性和灵活性。而由于 KL 惩罚的存在，模型如果想把某个词的概率推到极高（接近 1.0），会面临巨大的 KL 惩罚成本。这迫使模型在追求高分的同时，尽量保持概率分布的平滑。

        # ⭐️⭐️⭐️⭐️⭐️ 在这里介绍一下模型的熵。在信息论和大模型训练的语境下，熵（Entropy）可以直观地理解为模型输出的不确定性，或者说“选择的多样性”。如果把 Token 的分布比作一个调色盘：
        # 1. 高熵（High Entropy）：调色盘里颜色很多，分布均匀。模型觉得好几个词（Token）都挺合适，说话比较“油滑”且富有变化。
        # 2. 低熵（Low Entropy）：调色盘里只有一种颜色。模型认定只有一个词是对的，说话非常“死板”且单一。

        # 在我们的代码里，每一个位置（Position）上的 Token 都有一个概率分布。如果分布是平摊的，每个词概率都是 0.01，熵就很大。模型在生成时随机性强，不容易预测。如果分布是尖锐的，某个词概率 0.99，其他都是 0.0001，熵就很小。模型几乎处于“锁死”状态，每次生成的词都一样。

        # ⭐️⭐️⭐️⭐️⭐️ 这里也科普一下 entropy_loss。简单来说，Entropy Loss（信息熵损失）的唯一目标是防止模型太快变得“盲目自信”，强迫模型保持探索（Exploration）的热情。

        # ⭐️⭐️⭐️⭐️⭐️ GRPO 的 clip 中的 1+e 和 1-e 也会对熵造成影响。
        # 1-e 越大，劣化 token 的门槛被抬高了，模型很难把概率从坏的统治级 token 上转移走，确实不利于注入新的熵，反之，下界越低（例如 0.7），劣化 token 的门槛越低，释放出的概率空间就越多，更加可以注入新的熵；1+e 越大，优化 token 的空间越大，让模型能够给予这些低概率的好 token 更高幅度的优化，降低原来这个位置上的确定性，从而增加熵。

        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # per_token_loss 的 shape 是 [B*num_gen, R]，completion_mask 是 [B*num_gen, R]。由于在生成时，Batch 里的每个句子长度不一，短句后面会补齐 <pad> Token。这步操作把所有 <pad> 位的 Loss 全部归零。我们只关心模型真正生成的、属于“回答”部分的那些 Token 表现如何。
        # (per_token_loss * completion_mask).sum(dim=1) 变成了 [B*num_gen]，表示序列级的 loss，completion_mask.sum(dim=1) 变成了 [B*G]。这行代码先把每一行（即每一个生成的回答序列）有效的 Loss 加起来，再除以该行有效 Token 的数量。
        # 如果不除以数量，长句子产生的 Loss 绝对值会远大于短句子（因为加和项多）。这会导致模型产生一种错觉：只要说得越长，受到的奖惩就越剧烈。通过这一步，我们将每个句子的贡献缩放到同一个量级（即该句子的平均 Token 损失）。⭐️⭐️⭐️⭐️⭐️ 这一步先按行归一化，确保了每个回答（Completion）对梯度的贡献是均等的，无论它是 10 个词还是 100 个词。
        # .mean() 是在做批次聚合。对 Batch 中所有生成的回答（B * G 个回答）取算术平均。将这一组（Group）中所有样本的反馈浓缩成一个单一的浮点数。这个最终的 policy_loss 就是传给 loss.backward() 的那个值。

        # ⭐️⭐️⭐️⭐️⭐️ policy_loss 是先在 sample 内求平均，再在 batch 内求平均。那么由于梯度和 Loss 的线性关系，对 Loss 的任何加权、求和或除法操作，都会完全线性地反映在梯度（Gradient）上。所以梯度也会先在 sample 内求平均，再在 batch 内求平均。当然，这是因为 loss 此处的计算没有非线性计算，所以 loss 和梯度的关系才如此的线性。

        # ⭐️⭐️⭐️⭐️⭐️ policy_loss 是基于前文的 “条件概率”，表示在给定当前上下文（Context）时，模型选择特定 Token x_k 所贡献的那一份“功劳”或“过失”。per_token_loss 的绝对值大小决定了模型在更新参数时，对“改掉”或“强化”这个 Token 的迫切程度（梯度强度）。

        # 仔细去看 policy_loss 的计算，他其实有两个 mean 的过程：
        # 1. 第一个 mean 是序列级的 mean。这是为了解决长句子的歧视，因为梯度会天然向“长句子”倾斜。长句子 Token 多，产生的总 Loss 大，梯度就粗。但是 GRPO 任务，无论模型写 10 个词还是 100 个词，你作为一个“完整的回答（Completion）”，在这一组（Group）竞争中，投票权应该是相等的。这是为了让强化学习的 Advantage（优势信号） 能准确、均匀地作用于不同长度的回答上。
        # 2. 第二个 mean 是 batch 级的 mean。这意味着在这一次参数更新中，模型收到的信号不是来自某一个特定的好句子，也不是来自某一个特定的 Token，而是整个 Batch 表现的“平均风向”。按 Batch 平均，相当于在这一步里，让所有句子的优缺点互相抵消，只留下最稳健的优化方向。

        # ⭐️⭐️⭐️⭐️⭐️ SFT 的 loss 也会按照 batch 级别去做 mean，这几乎是现代深度学习所有监督学习任务的标配动作。如果不做 Batch 级的 mean，梯度量级会随 Batch Size 爆炸，我们无法在不改学习率的情况下把 Batch 从 4 增加到 64。此外多卡同步梯度时，如果各卡样本数不同且没取平均，模型会直接跑飞。
        # SFT 的 mean 是 global token 的 mean，和 GRPO 认为每个“回答”都是策略，都要评分不同，SFT 认为每个“字”都是知识，都要学对。因此他对所有的 token 都是一视同仁的，因此对于长句，肯定它的关注会更高。

        # 新的问题来了，既然 mean 我们知道了为什么要做，那么为什么要做 sum 呢？
        # ⭐️⭐️⭐️⭐️⭐️ 从最终的目的来说，这里的 sum 也是为了解决长短句歧视，核心其实就是为了在序列维度聚合这个序列的梯度，然后方便后续做除法，然后消除长短句的梯度歧视。
        # 那么，假设原始的 per_token_loss，其中的 wait token loss 很高，按道理来说，wait token 的梯度强度应该很大，这里做了 (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1) 之后，wait token 的更新强度不会变吗？
        # 1. 绝对强度，确实变小了。如果这个句子很长（比如 100 个词），wait 产生的梯度会被那个分母（100）稀释 100 倍。从物理意义上讲，模型更新参数时，花在 wait 上的“力气”确实变小了。
        # 2. 相对强度，它依然是整条街“最靓的仔”。尽管整体强度被分母除小了，但在这一行产生的梯度分配中，wait 拿走的份额依然远高于 the 或 is。模型依然知道，这个序列之所以要大改，主要是因为 wait 没处理好。反向传播时，梯度依然会顺着求和路径，成比例地回传给各个 Token（这是反向传播计算梯度时，自动实现的）。
        # ⭐️⭐️⭐️⭐️⭐️ 一个 loss 由 k 个 loss 组成，反向传播时，k 个 loss 会也会对应 k 个梯度，但是对于单个 token 来说，loss 大不代表梯度大，和 sigmoid 的位置也有关。
        # 1. 比如 policy_loss 和 per_token_loss，这里的 loss 和梯度就是线性的。
        # 2. 但是 per_token_loss 和 per_token_logps，loss 和梯度就是非线性的关系。
        
        # 如何理解梯度，对于 \frac{\partial \text{Logits}_i}{\partial \theta}，我们可以理解为参数 \theta 的微小变动对第 i 个位置输出的影响。

        loss = (policy_loss + aux_loss) / args.accumulation_steps 
        # 做梯度累积。
        loss.backward()
        # 反向传播。

        # 下面来讲讲 GSPO 和 DAPO 的优化。首先是从 GRPO 到 DAPO。
        # ⭐️⭐️⭐️⭐️⭐️ 为什么 DAPO 提高了 1 + e 的上界？

        # 作者发现，如果 clip 的上界 e 设置过小，会出现这样的问题：当 old policy 对某个 token 的概率很低，而该 token 的 advantage 又是正值（即 old model 恰好采样得非常好），此时当前 policy model 的上涨空间就会受到很大限制，而上涨恰恰是我们希望发生的。
        # 举例来说，如果 old policy 的概率是 0.9（这代表在这个位置上，选择某个 token 的概率是 0.9），e = 0.2 ，clip 上界为 1.08（意思是 new model 在这个位置上选择同样 token 的概率最大可以为 1.08），已超过概率的最大值 1.0，这种情况是绝对不会被 clip 的；但如果 old policy 的概率是 0.2，clip 上界仅为 0.24，即便当前模型将其概率提升到 0.4（一个不是非常激进且恰到好处的改进），也会因 e 过小而被 clip，导致该 token 的训练信号被废弃。为了解决这一问题，DAPO 引入 Clip-Higher，提高上界以提升 token 利用效率。

        # ⭐️⭐️⭐️⭐️⭐️ DAPO 的动态采样
        # DAPO 的第二个创新是动态采样（Dynamic Sampling）。这项技术的背景是 => 假如一个 query 我们 sample 了 10 次，这 10 次每次都答得很好/或者很差，都取得了 max reward/zero reward，这个时候由于 GRPO 的计算方法，导致这 10 次采样的 advantage 都是 0，所以这些采样所带来的 gradient 就也都是 0。
        # 这样做的一个后果就是，实际的有梯度的 sample 要远低于名义 sample 数，导致最后梯度汇集的时候没有收集到足够的信息，从而形成高方差（梯度忽大忽小）、不稳定的训练，以及 sample 的浪费。
        # 需要注意的是，这种现象是在训练初期；以及后期随着训练的进行在不断加强的，因为刚开始时模型效果很差，而训练越到后边模型效果越好，给出满分回答的几率就越大。因此，DAPO 在采集样本时，额外做了一件事：保证每次采样出来的回答，reward 不全是 0 或者 1，如果采样出来的回答全是 0 或者 1 就继续采样，直到不满足为止，它保证同一输入下的采样集合中既包含正确回答，也包含错误回答。
        # NGRPO 与这里的动态采样是类似的，都是为了保证 Advantage 不为 0，从而有梯度。

        # ⭐️⭐️⭐️⭐️⭐️ DAPO - Token-Level Gradient Loss
        # DAPO 第三个方面的创新是为了解决 GRPO 在训练长回答时 gradient 的权重会随着采样回答的长度变长而下降的问题。首先解释为什么采样长度变长权重会下降。假设采样了 2 次，有一次回答一共有 200 个 token，而另一次回答有 10 个 token。那么根据 GRPO 的计算公式，每次回答的梯度先在 sample 内求平均，再在 batch 内求平均。第一次回答每个 token 的权重是 1/200 * 1/2，而第二个回答每个 token 的权重是 1/10 * 1/2，所以第二次回答的 token 的影响要明显高于第一次回答。
        # 再来说采样长度变长权重下降的危害：对于一些比较难的问题，长回答本身就很正常，如果这些回答本身非常好，那么由于长度平均就会导致本来非常有用的梯度信号被稀释；假如回答是不好的，长度长仅仅也是因为重复单词，或者回答冗余词太多，长度归一就导致这次采样本该带来的纠正信号没法正确传递到 policy model 上。总的来说，长高质量回答的有用信号被稀释；长低质量回答的纠正信号也被稀释（长只是因为冗余或重复）。
        # 所以 DAPO 采用的方法是：把一次梯度计算时所有采样生成的 token 总数加起来求平均，回到上边这个例子，第一次采样和第二次采样每个 token 的权重都是 1/(200 + 10)，即对不同回答中的 token 一视同仁。这样就能改善 GRPO 在长样本训练中的效率低下的问题。所以他的核心就是将序列级别的 mean 拆解为了 Global Token 级别的 mean。

        # ⭐️⭐️⭐️⭐️⭐️ DAPO - Overlong Reward Shaping
        # DAPO 的第四个改进是在奖励设计中引入 软惩罚机制（Soft Punishment）来处理过长回答。具体来说，当生成长度超过第一个预设阈值时，惩罚会随长度线性增加；一旦超过第二个阈值，惩罚将抵消因回答正确获得的所有奖励，相当于将该回答视为无效。这种惩罚是按 token 作用在 reward（即 advantage）上的。
        
        # 进入 GSPO，解决 MoE 训练中 GRPO 不稳定的问题。
        # ⭐️⭐️⭐️⭐️⭐️ https://qwen.ai/blog?id=gspo。

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step() # 学习率更新。
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
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
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # 根据参数决定加载哪种预训练权重。如果开启了推理模式，则使用带有“reasoning”能力的权重，否则使用常规的全量微调（SFT）权重。

    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # 初始化 Policy 模型（策略模型）。这是我们要通过强化学习不断更新、学习如何生成更好答案的主体模型。

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    # 初始化 Reference 模型（参考模型）。通常其权重与初始状态的 Policy 模型一致。
    ref_model = ref_model.eval().requires_grad_(False)
    # 将参考模型设为评估模式并冻结参数（不计算梯度）。它的作用是在训练中提供对比，防止 Policy 模型为了追求高奖励而偏移初始分布太远（计算 KL 散度约束）。

    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    # 这里的奖励模型用的是 InternLM2-1.8B-Reward，其是在 InternLM2-Chat-1.8B-SFT 基础上训练得到的奖励模型。
    # 它负责给 Policy 模型生成的答案打分，判断好坏。
    # Reward 模型（奖励模型）和 SFT 模型（指令微调模型）确实是“同根同源”的亲兄弟。 它们拥有完全相同的“身体”（Transformer 骨干网络），只是换了一个“头”（输出层）和一套“灵魂”（训练目标）。

    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    # 将奖励模型移动到指定设备，并同样设为评估模式且冻结梯度。奖励模型在 GRPO 训练过程中只负责“评分”，不参与参数更新。

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    # 加载奖励模型专用的分词器（因为 Reward 模型的词表或预处理方式可能与 Policy 模型不同）。


    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    # 这个加载器在这里的主要目的是为了“数数”。它结合了数据集大小、batch_size 和采样器逻辑，用于确定在一个 Epoch 中模型会接收到多少次数据批次。

    iters = len(loader_for_count)
    # 获取单个 Epoch 内的总迭代（Iteration）次数。

    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    # 计算整个训练生命周期内，优化器真正执行参数更新的总次数。

    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    # cosine 学习率调度器。
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
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
        # Sampler 负责“点名”（生成索引列表），DataLoader 负责“派人”（调度 Worker）去 Dataset 这个“大仓库”里把对应名字的货取出来打包。
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, ref_model, reward_model, reward_tokenizer, start_step, wandb)
        else:
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()