import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        # in_features: 输入特征的维度（例如大模型某层的输入宽度）。
        # out_features: 输出特征的维度。
        # rank: 秩（通常设为 4, 8, 16 等）。这是 LoRA 的核心参数，数值越小，训练的参数量就越少，计算也越快。
        super().__init__()
        self.rank = rank  
        # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  
        # 低秩矩阵A，它将高维输入（in_features）降维到低维空间（rank）。
        self.B = nn.Linear(rank, out_features, bias=False)  
        # 低秩矩阵B，它将低维特征（rank）重新升维到原始输出空间（out_features）。

        # 通过 B * A 的结构，我们得到了一个和原始权重矩阵大小一致、但参数量极小的结构。

        # 矩阵A高斯初始化，矩阵 A 使用高斯分布随机初始化。这为模型提供了训练的初始梯度和随机性，确保模型可以开始学习。
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化，初始化为全 0。
        self.B.weight.data.zero_()

        # - B 被初始化成 0 是因为需要保证 LoRA 模块在训练初期对原始模型的权重 W_0 没有任何改动，即 W_updated=W_0+0=W_0。 
        # - A 采用高斯分布随机初始化是因为需要引入随机性，保证打破对称性，使得 A 的不同列（或行）可以学习到不同的特征，有助于模型探索更广阔的参数空间。

        # 为什么必须是 A 采用高斯分布，B 采用 0 初始化，而不是 B 采用高斯分布初始化，而 A 采用 0 初始化？
        # 在链式法则中，参数矩阵 W_loraB 和 W_loraA 会被逐个更新。这里我们需要介绍一下对于一个矩阵运算 Y = WX 而言，其梯度更新只和两个东西有关：
        # 1. 该参数在当前操作中的“输入值”，譬如对于 Y = WX，我们要更新 W，那么就和 X 有关。
        # 2. 上游梯度值，也就是损失函数 L 对 Y 的梯度（也就是损失对运算结果的梯度），梯度最上游是损失函数本身对神经网络输出的梯度。
        # L = f(W)。
        # 那么 dL/dW = dL/dY * dY/dW = dL/dY * x。
        
        # 如果理解了这一点，就知道了为什么需要让 A 随机初始化，然后让 B 为 0 了，因为正向传播时，矩阵操作如下：
        # 1. X1 = Wa（不为 0）* X0，此时 X1 不为 0。
        # 2. X2 = Wb（为 0）* X1。
        # 那么反向传播时，就会先根据 B 的梯度来更新 B，其梯度计算为 X1 * L 对 X2 的梯度（上游梯度，一般不为 0），然后更新 Wb 后，继续反向传播更新 Wa，其梯度计算为 X0 * L 对 X1 的梯度（这里的 L/X1 = L/X2 * X2/X1（也就是 Wb）），此时正好都不为 0（Wb 刚刚更新也不为 0），因此 Wa 的梯度也不为 0，从而能够更新 Wa。 

        # 如果相反过来，也就是 X1 为 0，X2 不为 0，那么反向传播过程如下：
        # 1. X1 = Wa（为 0）* X0，此时 X1 为 0。
        # 2. X2 = Wb（不为 0）* X1。
        # 先根据 B 的梯度来更新 B，其梯度计算为 X1 * L 对 X2 的梯度（上游梯度，一般不为 0），此时 Wb 的梯度是 0，Wb 无法得到更新，继续反向传播更新 Wa，其梯度计算为 X0 * L 对 X1 的梯度（这里的 L/X1 = L/X2 * X2/X1（也就是 Wb == 0）），此时正好为 0（Wb 刚刚无法更新，依旧为 0），因此 Wa 的梯度也为 0，从而不能够更新 Wa。
    
    def forward(self, x):
        # 前向传播计算，顺序为 x * Wa * Wb。
        return self.B(self.A(x))
        # 在实际的大模型推理中，这个结果会加到原始线性层的输出上，也就是 y = W0x + x * Wa * Wb。


def apply_lora(model, rank=8):
    # 这段代码展示了如何动态地将 LoRA 模块注入到现有的预训练模型中。
    # 它利用了 Python 的动态特性，在不重写整个模型类的情况下，“切入”并修改了指定层的行为。
    for name, module in model.named_modules():
        # 递归地遍历模型中的所有子模块（如卷积层、线性层、注意力层等）。name 是层的路径名，module 是具体的层对象。
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 必须是线性层 (nn.Linear)，具体来说就是 Attention 中的 qkvo，ffn 中的 gate、up、down，以及最后输出的分类头（llm_head）。
            # 必须是方阵（输入维度等于输出维度，例如 Transformer 中的自注意力投影层）。这通常是为了简化演示或针对特定的权重设计。
            # 这个条件排除了ffn 中的 gate、up、down，以及最后输出的分类头（llm_head）。
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            # 实例化我们定义的 LoRA 类，其输入输出维度与当前线性层对齐，并确保它被移动到与主模型相同的计算设备（CPU 或 GPU）上。
            setattr(module, "lora", lora)
            # 在当前的 module 对象中新增一个名为 "lora" 的成员变量。现在，这个线性层对象内部持有了我们新创建的 LoRA 层。
            # setattr(obj, "name", value) 等同于执行 obj.name = value。
            # - 执行前：module（比如一个 nn.Linear 层）只有自带的属性，如 weight 和 bias。
            # - 执行后：module 现在多了一个名为 lora 的属性。我们可以通过 module.lora 直接访问到我们创建的那个低秩矩阵模块。

            # 在 PyTorch 中，几乎所有构建神经网络的“层”（如 nn.Linear, nn.Conv2d, nn.LSTM）都继承自 nn.Module。
            # 而 nn.Parameter 不继承自 nn.Module。

            # 由于 module 是一个 nn.Module 对象，当我们给它赋予一个同样也是 nn.Module 的对象（即我们的 LoRA 实例）时，PyTorch 的内部机制（__setattr__ 魔法方法）会自动做两件事：
            # 1. 自动注册为子模块：它会将这个 lora 实例添加进 module 的 _modules 字典中。
            # 2. 建立追踪关系。当我们调用 model.parameters() 时，新加入的 LoRA 参数也会被包含在内。当我们执行 model.to("cuda") 时，这个 lora 模块也会跟着一起移动到 GPU。当我们保存模型 torch.save(model.state_dict()) 时，LoRA 的权重也会被自动保存进文件里。
            
            # 方法劫持。
            original_forward = module.forward
            # 把该层（nn.Linear）原本的计算逻辑“备份”起来。为什么要备份？因为我们接下来要覆盖 module.forward。如果不先存下来，我们就永远失去了调用原始矩阵 $W_0x$ 计算的能力，会导致模型彻底失效。

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 这里定义了一个全新的函数，用来替代旧的计算过程。最关键的是默认参数（layer1=..., layer2=...）。
                # 通过把 original_forward 和 lora 赋值给函数的默认参数，我们就把当前的这两个对象**“锁死”**在了这个函数的局部作用域里。每个线性层都会拥有自己专属的 layer1（原权重）和 layer2（对应的 LoRA 插件）。
                return layer1(x) + layer2(x)
            
            # 还有一种关于 forward_with_lora 的错误实现，
            # 是一个非常经典且隐蔽的 Python 闭包（Closure）陷阱，通常被称为 “延迟绑定”（Late Binding）。

            # for name, module in model.named_modules():
            #     if isinstance(module, nn.Linear):
            #         original_forward = module.forward
            #         lora = LoRA(...)
            #         # 错误写法：直接引用外部变量
            #         def forward_with_lora(x):
            #             return original_forward(x) + lora(x) 
            #         module.forward = forward_with_lora

            # 在 Python 中，嵌套函数（forward_with_lora）在定义时并不会保存外部变量 original_forward 和 lora 的具体值。它只保存一个“变量名”的引用。
            # 想象一下循环过程：
            # - 第一次循环：处理 Layer_1。original_forward 指向 Layer_1 的原始方法。
            # - 第二次循环：处理 Layer_2。此时，全局（或函数作用域内）的 original_forward 变量被更新了，指向了 Layer_2 的原始方法。
            # ...
            # 循环结束：假设处理了 100 层。此时，变量 original_forward 最终停留在 第 100 层 的引用上。
            # 灾难发生了。当我们开始训练模型并调用 Layer_1 时，它去查找 original_forward。因为它存的是引用，它会找到当前变量最后的值——也就是 第 100 层的方法。
            # 结果就是无论我们调用哪一层，它们运行的其实全是最后一层的逻辑。 我们的模型坍塌了。

            # 默认参数（显式绑定）可以解决这个问题，默认参数是在函数“定义时”立即求值的。
            # 当我们执行这一行时，Python 会立即查看当前这一刻 original_forward 指向谁，并把这个具体的“对象地址”存入该函数私有的 layer1 槽位里。

            module.forward = forward_with_lora
            # 正式完成“劫持”。我们把 module（那个 nn.Linear 对象）的 forward 方法指向了我们刚刚定义的新函数。


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    # 使用 PyTorch 加载磁盘上的模型权重文件。
    # map_location=model.device 确保权重被直接加载到模型所在的设备（如 GPU 或 CPU）上，防止显存或内存溢出。
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # 移除参数名称中多余的 module. 前缀。
    # 如果你之前用 DataParallel 或 DistributedDataParallel 进行分布式训练，PyTorch 会自动在所有层名前加上 module.。为了让这些权重能加载到非分布式模型中，需要把这 7 个字符切掉。

    for name, module in model.named_modules():
        # 递归地遍历模型中的每一个子模块（Layer）。
        # name 是层的名字（如 layers.0.self_attn），module 是该层的对象。
        if hasattr(module, 'lora'):
            # 检查当前这个层是否包含一个名为 lora 的属性（这通常意味着该层被注入了 LoRA 结构）。
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 从巨大的 state_dict 中筛选出属于当前这个层的 LoRA 权重。
            # 假设有一个模型，里面有一层叫 layers.0.attention.q_proj。在这个层里，注入了一个名为 lora 的模块。
            # 那么在我们的 state_dict 中，关于这一层的参数名可能是这样的：
            # - "layers.0.attention.q_proj.lora.lora_A.weight"
            # - "layers.0.attention.q_proj.lora.lora_B.weight"

            # 当循环运行到这一层时，name 是 layers.0.attention.q_proj
            # f'{name}.lora.': 变成了字符串 "layers.0.attention.q_proj.lora."
            # 这行字典推导式会执行以下逻辑，首先是筛选 (if 部分) =>
            # 它在巨大的 state_dict 字典里扫描，只保留那些键名中包含 "layers.0.attention.q_proj.lora." 的项。
            # 然后是裁剪 (replace 部分) =>
            # 它把匹配到的键名中的前缀去掉。"layers.0.attention.q_proj.lora.lora_A.weight" => "lora_A.weight"
            # "layers.0.attention.q_proj.lora.lora_B.weight" => "lora_B.weight"
            # 最终生成的 lora_state 字典长这样：
            # {
            #     "lora_A.weight": tensor([...]), 
            #     "lora_B.weight": tensor([...])
            # }
            
            # 在 PyTorch 中，当你调用 module.lora.load_state_dict() 时，该子模块（module.lora）只认识它内部的参数名。它并不关心自己在整个模型大图景里的全路径名。

            module.lora.load_state_dict(lora_state)
            # 筛选出的局部权重真正加载到当前层的 lora 模块中。


def save_lora(model, path):
    # 只保存 LoRA 相关的参数。
    raw_model = getattr(model, '_orig_mod', model)
    # 如果你使用了 torch.compile()（PyTorch 2.0+ 的特性），模型会被封装在一个 _orig_mod 对象里。
    # 这行代码确保无论模型是否被编译过，我们都能拿到最原始的模型结构，避免层名多出 _orig_mod. 前缀。
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 创建一个容器，用来存放所有筛选出来的 LoRA 参数。
            # 只处理那些被注入了 LoRA 结构的层。这样保存下来的文件会非常小（通常只有几 MB），因为我们跳过了基座模型那几十亿个原始参数。
            clean_name = name[7:] if name.startswith("module.") else name
            # 如果模型层名以 module. 开头（通常是因为使用了 DDP 分布式训练），就把它切掉。
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            # 这是最关键的一步，它把局部参数名“还原”成全局参数名。
            # 假设 clean_name 是 layers.0.q_proj。
            # 该层内部的 LoRA 参数名原本只是 lora_A.weight。
            # 经过这行代码，它变成了："layers.0.q_proj.lora.lora_A.weight"。
            # 这样做的意义是确保加载时，程序知道这个 lora_A 到底属于哪一层。

            state_dict.update(lora_state)
            # 将这一层处理好的 LoRA 权重合并到总的 state_dict 中。
            # 当执行 dict_a.update(dict_b) 时，会发生两件事：
            # - 添加新成员：如果 dict_b 中的键在 dict_a 中不存在，则直接添加。
            # - 覆盖旧值：如果 dict_b 中的键在 dict_a 中已经存在，则 dict_a 中对应的值会被 dict_b 的新值覆盖。

    torch.save(state_dict, path)
    # 最终搜集到的所有 LoRA 权重保存到指定的 path 路径下。
