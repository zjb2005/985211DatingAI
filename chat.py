# chat.py —— 加载训练好的情感小模型，命令行对话
# 用法: python chat.py   (输入 quit 退出)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "你是一位温柔、善解人意的情感陪伴助手，善于共情和倾听，回答温暖真诚。"
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"

tok = AutoTokenizer.from_pretrained("emo-lora")
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16 if use_gpu else torch.float32).to(device)
model = PeftModel.from_pretrained(model, "emo-lora")   # 叠加 LoRA 权重
model.eval()

history = [{"role": "system", "content": SYSTEM}]
print("情感陪伴小助手已就绪，说说你的心事吧（输入 quit 退出）\n")

while True:
    user = input("你: ").strip()
    if not user:
        continue
    if user.lower() in ("quit", "exit", "q"):
        break
    history.append({"role": "user", "content": user})
    prompt = tok.apply_chat_template(history, tokenize=False,
                                     add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256,
                             do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][inputs.input_ids.shape[1]:],
                       skip_special_tokens=True).strip()
    print(f"助手: {reply}\n")
    history.append({"role": "assistant", "content": reply})
    if len(history) > 9:                      # 只保留最近4轮，防止上下文过长
        history = [history[0]] + history[-8:]
