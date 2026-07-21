# train.py —— LoRA 微调 Qwen2.5-1.5B，打造情感陪伴小模型
# 用法: python train.py   (GPU 约3-5分钟，纯CPU 约20-40分钟)
import json
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # 5亿参数小模型，笔记本可跑
SYSTEM = "你是一位温柔、善解人意的情感陪伴助手，善于共情和倾听，回答温暖真诚。"
use_gpu = torch.cuda.is_available()

# ---------- 1. 加载模型和分词器 ----------
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if use_gpu else torch.float32)

# ---------- 2. 挂上 LoRA（只训练约0.5%的参数，省显存） ----------
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                  task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# ---------- 3. 准备数据：套用对话模板并分词 ----------
with open("data.json", encoding="utf-8") as f:
    raw = json.load(f)

def to_ids(ex):
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": ex["instruction"]},
         {"role": "assistant", "content": ex["output"]}],
        tokenize=False)
    return tok(text, truncation=True, max_length=512)

ds = Dataset.from_list(raw).map(to_ids, remove_columns=["instruction", "output"])

# ---------- 4. 训练 ----------
args = TrainingArguments(
    output_dir="ckpt",
    num_train_epochs=4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="no",
    fp16=use_gpu,
    report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
trainer.train()

# ---------- 5. 保存 LoRA 权重（只有几MB） ----------
model.save_pretrained("emo-lora")
tok.save_pretrained("emo-lora")
print("训练完成！权重已保存到 ./emo-lora，运行 python chat.py 开始对话")
