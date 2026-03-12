from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # add_system_ratio 一个概率值，表示有 20% 的概率会给这条对话加上系统提示词。
    # 如果你的训练集里全是用户问、助手答，模型可能不知道自己叫什么。通过这种方式，你强行把 "minimind" 这个名字刻进了模型的记忆里。
    SYSTEM_PROMPTS = [
        # 这包含了一组预定义的身份描述（中英文都有）。
        # 核心目的是通过训练，让模型“记住”自己叫 minimind，并设定其语调（专业、友好、可靠）。
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            # 为什么要随机？ 如果 100% 都加，模型可能会产生依赖，没有系统提示词就不会说话了；保持一定比例的随机，能让模型更鲁棒。
            # 这里回答两个问题：
            # 1. 为什么要在 SFT 数据集中添加 System Prompt？
            # A：大模型本质上是概率预测。如果不告诉它“你是 MiniMind”，它可能会根据训练数据里残余的信息，一会儿觉得自己是 GPT-4，一会儿觉得自己是文心一言。通过反复训练包含“你是 MiniMind”的 Prompt，模型会建立起牢固的自我认同。
            # 2. 为什么要“随机”加（Why Random?）
            # 如果模型只见过带 System Prompt 的数据，当你实际使用时，万一忘写了 System Prompt，或者用户直接问“你好”，模型可能会因为找不到那个“启动开关”而变得语无伦次，甚至拒绝回答。随机添加能训练模型在有或没有系统指令的情况下都能正常工作。
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
# System Prompt 是在 SFT 阶段被“教会”的，但在推理阶段被“激活”使用。
# 一般来讲，System Prompt 不会在 pretrain 阶段加入，只会在 SFT 阶段加入，尤其是指令微调阶段。

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    # 这行在检查字符串中是否包含一个完全没有内容的思考标签。只有当随机数大于 0.05 时（即 95% 的概率），才会进入删除逻辑。
    # 换句话说，有 5% 的概率，即使思考块是空的，我们也会原样保留它。
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
        # 把那个占地方但没内容的 <think>\n\n</think>\n\n 彻底删掉，让 Prompt 变得更紧凑。
    return prompt_content
# 这段代码的目的：
# 1. 如果你的模型模板里默认带了 <think> 标签，但在某些对话中（比如简单的打招呼），模型并没有产生实际的思考逻辑。保留一对空的 <think> 标签会白白占用 Token 位。删掉它们可以减少显存占用，提高推理/训练效率。
# 2. 为什么要 5% 的保留？一是如果训练数据里 100% 的空思考块都被删掉了，模型可能会产生一种错觉：“只要出现 <think> 标签，里面就必须有字”。
# 二是保留 5% 的空块，是告诉模型：“有时候我也可能只是想了一下但啥也没想出来，这也是正常的。”这样模型在推理遇到空块时，不会因为没见过这种格式而崩掉。

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    # 继承 Dataset 有什么好处？
    # 1. 可以深度集成 DataLoader，包括了可以使用多进程并行加载（利用 CPU 多核并行读取和预处理数据，防止 GPU 闲置等待数据）、自动批处理（自动把单个样本组装成 Batch）、数据打乱（每一个 Epoch 自动打乱数据，增加训练的随机性）、自动内存固定（速数据从内存拷贝到显存的速度）。
    # 2. 实现按需加载数据。如果你的数据集有 100GB，你不可能一次性读入内存。
    # - 在 __init__ 中，你只需要保存文件路径列表。
    # - 在 __getitem__ 中，你才真正读取当前索引的文件。
    # 好处是无论数据集多大，内存占用始终保持在极低的水平。
    # 3. 逻辑分离。将“如何读取数据”与“如何训练模型”完全分开。你可以轻松更换数据集（比如从本地 JSON 改为数据库读取），而不需要修改一行训练循环代码。
    # 4. 通用性：几乎所有 PyTorch 相关的第三方库（如 Hugging Face, PyTorch Lightning）都默认你使用的是 Dataset 对象。
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer # 获取 tokenizer。
        self.max_length = max_length # 设置序列的最大长度。
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        # 使用 Hugging Face 的 datasets 库加载本地的 JSONL 数据。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        # 获取“助手开始回答”的标识符 ID（如 <|im_start|>assistant\n）。
        # 这里的 self.bos_id 一般是三个，例如 [200264, 173781, 198]
        # 访问 https://tiktokenizer.vercel.app/ 可知，<|im_start|> 和 assistant 两个 token。
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        # 获取“回答结束”的标识符 ID（如 <|im_end|>\n）。

    def __len__(self):
        # self.samples 返回的是 Hugging Face datasets 库中的 Dataset 对象。
        # Hugging Face 的 Dataset 类在内部已经实现了 __len__ 方法。它本质上像是一个高度优化的列表（List）或表格。
        return len(self.samples)
    # 在 Python 中，一个对象能否被 len()，取决于它内部是否定义了 __len__ 这个魔术方法。

    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        # 这里的 .copy() 只是一个浅拷贝。
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        # 判断当前对话是否包含“外部工具/函数调用”。
        # 如果满足条件，就把这些函数定义赋值给 tools；否则设为 None。这通常用于训练模型学会何时调用 API（如查询天气、计算器等）。

        # apply_chat_template 的好处在于它会自动读取模型配置文件里的 chat_template 脚本。你不需要手动拼接这些复杂的特殊符号，代码会自动适配你加载的任何模型。
        return self.tokenizer.apply_chat_template(
            messages,
            # 传入刚才复制的对话列表
            tokenize=False,
            # 告诉分词器只返回转换后的字符串，先不要急着转成数字 ID。
            add_generation_prompt=False,
            # 如果是推理阶段（让模型回答），我们会设为 True，它会在字符串末尾强行加上 <|im_start|>assistant\n 来诱导模型开口。
            # 在 SFT 训练中，我们要训练模型自己学会生成这些标识符，因此设为 False，保持对话的完整闭环。
            tools=tools
            # 如果刚才提取到了工具定义，模板会自动将其格式化到 System Prompt 中。
        )

    def generate_labels(self, input_ids):
        # 这段函数是整个 SFT（监督微调）逻辑中最硬核的部分。
        # 它的目标是制作一个“遮罩”：让模型在看到用户问题时“闭嘴”（不计算损失），只有在看到助手回答时才“学习”（计算损失）。
        labels = [-100] * len(input_ids)
        # 创建一个和输入序列等长的列表，初始值全部设为 -100。
        # 在 PyTorch 的 CrossEntropyLoss 中，ignore_index 默认就是 -100。这意味着对应的位置在训练时会被完全忽略，不产生梯度。
        # 这是因为模型在训练时依然会预测用户问题本身，但我们通过 -100 “假装”它没预测，或者说“不关心”它预测得对不对。
        i = 0
        while i < len(input_ids):
            # 使用 while 循环手动控制指针 i，因为一旦找到目标，我们需要跳跃式前进。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 检查从当前位置 i 开始的片段，是否匹配预先定义的助手起始符（如 <|im_start|>assistant\n）。
                # 如果匹配成功，说明从这里开始到回答结束，都是我们要学习的内容。
                start = i + len(self.bos_id) # 学习的起点。
                end = start # 学习的终点。
                while end < len(input_ids):
                    # 不断向后移动 end 指针，直到撞上助手回答的结束符（如 <|im_end|>\n）。
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    # 将 start 到 end（包括结束符）这一段的标签从 -100 替换为 真实的 input_ids。
                    # max_length 表示输入 + 输出 + 模板标识符的总和。
                    labels[j] = input_ids[j]
                    # 这里的 labels 就是正确答案。
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
                # 既然这一段助手的话已经处理完了，直接把指针 i 移到 end 之后，继续寻找下一轮对话。
                # 所以这里是对 i 的更新。
            else:
                i += 1
                # 如果当前位置不是助手开头，指针往后挪一格继续找。
        return labels
    # 假设输入为：[User提示词] [Assistant开始符] [我是人工智能] [Assistant结束符] [Padding...]
    # 那么 labels 则为：[-100, ..., -100] [-100, ..., -100] [我是人工智能] [Assistant结束符] [-100, ...]

    # 在 Python 中，双下划线开头和结尾的方法（也叫 Dunder Methods）都是为了让你的类能像内置对象一样工作。
    # 当你写 dataset[0] 时，Python 解释器在底层会自动将其转换成 dataset.__getitem__(0)。
    # 还有的例子为 for item in obj => 调用 obj.__iter__()
    def __getitem__(self, index):
        # 这段 __getitem__ 函数是数据加载的“流水线”，它定义了当你从数据集取第 index 条数据时，如何把原始的 JSON 文本变成模型能直接“吞掉”的张量（Tensor）。
        # 虽然我们定义了 __getitem__，但是并不是由我们来调用 dataset 的，主要调用链如下：
        # 1. DataLoader (核心调度者)：当你启动训练循环并遍历 DataLoader 时，它内部会产生一系列索引（index）。DataLoader 同时也会负责维持整个流程。他从 Sampler 那里拿到索引，然后把任务分派给具体的搬运工（Workers）。
        # 2. Sampler (索引提供者)：如果是分布式训练，DistributedSampler 会告诉 DataLoader 当前卡需要哪些索引（比如 0, 8, 16...）。
        # 3. Worker (多进程搬运工)：如果你设置了 num_workers > 0，DataLoader 会开启多个子进程。这些子进程会并发地调用你 SFTDataset 里的 __getitem__(index)。
        
        # __getitem__ 是一个多进程的实例。当你设置 num_workers > 0 时，PyTorch 确实是在多个独立的进程中并行运行 __getitem__ 的。
        # 当你启动 DataLoader 时，主进程会使用 spawn 或 fork 创建出 N 个子进程（Workers）。每个子进程都完整复制（或共享映射）了一份你的 SFTDataset 实例。每个子进程都在独立运行自己的 Python 解释器。当 DataLoader 需要索引 [10, 11, 12, 13] 时，它会把 10 发给 Worker A，11 发给 Worker B…… 它们同时开始执行各自的 __getitem__。

        # 因此在 __getitem__ 的写入操作是状态不共享的，因此最佳实践是 __getitem__ 应该是只读的。
        # class SFTDataset(Dataset):
        #     def __init__(self):
        #         self.counter = 0

        #     def __getitem__(self, index):
        #         self.counter += 1  # ⚠️ 这里的修改只在当前的 Worker 进程生效
        #         return ...

        sample = self.samples[index]
        # 从 load_dataset 加载好的数据集对象中，取出索引为 index 的那一条原始 JSON 数据。
        # 这里的 sample 如下所示：
        # {"conversations": [{"role": "user", "content": "你好，我是第一次使用你们的平台，我该怎么使用你来帮助我解决问题呢？"}, {"role": "assistant", "content": "你好！很高兴能为你提供帮助。使用我来解决问题非常简单，你只需要告诉我你遇到的具体问题或需要帮助的领域，比如学习、工作、技术问题、生活建议等。我会根据你的需求，提供相应的信息、建议或解决方案。如果你有任何疑问，或者需要更具体的指导，也可以随时向我提问。现在，你可以告诉我你具体需要帮助的内容了。"}]}
        
        conversations = pre_processing_chat(sample['conversations']) # 加入 System Prompt。
        prompt = self.create_chat_prompt(conversations) # 将 conversations 加上 bos、eos 等 special token。
        prompt = post_processing_chat(prompt) # 去除空思考标签。
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # token 化，同时去除超出的部分。
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 如果对话比较短，不足 max_length，就在列表末尾填充 pad_token_id（通常是 0）。
        # 目的是矩阵整齐化。在 GPU 计算时，同一个 Batch（批次）里的所有数据必须长度一模一样，模型才能像处理表格一样并行计算。
        # 结果就是现在 input_ids 的长度不多不少，正好等于 max_length。

        # 这里的将 input_ids 拼接成 max_len 是为了和别的样本保持对齐，实现静态填充。
        # 在 __getitem__ 运行完后，DataLoader 会调用一个叫 collate_fn 的函数，把多个样本堆叠（Stack）在一起。如果样本 A 长度是 100，样本 B 长度是 200，它们是没法拼成一个长方形矩阵的。
        # 但是这种静态填充比较浪费 GPU，还有一种方法是在 __getitem__ 里不补齐（让每个样本保持原始长度），然后在 DataLoader 的 collate_fn 阶段，根据当前这个 Batch 里最长的那条数据动态地补齐。这叫 Dynamic Padding。这样做的好处是如果一个 Batch 里大家都短，那就只对齐到短的长度，训练速度能提升好几倍！

        labels = self.generate_labels(input_ids)
        # 将刚刚准备好的、整齐的 input_ids 丢进我们之前讨论过的 generate_labels 函数里。
        # 产出得到一个同样长度为 max_length 的 labels 列表。
        # 这个列表里，用户问题对应的位置全是 -100（被掩盖），只有助手回答的部分是真实的数字 ID。

        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        # 返回最终的结果，input_id 都是整数。 


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }

if __name__ == "__main__":
    pass