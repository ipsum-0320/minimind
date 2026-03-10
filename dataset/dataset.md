# MiniMind Datasets

将所有下载的数据集文件放置到当前目录.

Place the downloaded dataset file in the current directory.

dpo.jsonl --RLHF阶段数据集
lora_identity.jsonl --自我认知数据集（例如：你是谁？我是minimind...），推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
lora_medical.jsonl --医疗问答数据集，推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
pretrain_hq.jsonl✨ --预训练数据集，整合自jiangshu科技
r1_mix_1024.jsonl --DeepSeek-R1-1.5B蒸馏数据，每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
sft_1024.jsonl --整合自Qwen2.5蒸馏数据（是sft_2048的子集），每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
sft_2048.jsonl --整合自Qwen2.5蒸馏数据，每条数据字符最大长度为2048（因此训练时设置max_seq_len=2048）
sft_512.jsonl --整合自匠数科技SFT数据，每条数据字符最大长度为512（因此训练时设置max_seq_len=512）
sft_mini_512.jsonl✨ --极简整合自匠数科技SFT数据+Qwen2.5蒸馏数据（用于快速训练Zero模型），每条数据字符最大长度为512（因此训练时设置max_seq_len=512）