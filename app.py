# app.py —— 情感小助手网页版界面（兼容新旧版 Gradio）
# 用法: python app.py，会自动打开浏览器
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # 必须和训练时的基座一致
SYSTEM = "你是一位温柔、善解人意的情感陪伴助手，善于共情和倾听，回答温暖真诚。"
use_gpu = torch.cuda.is_available()

print("正在加载模型，请稍候...")
tok = AutoTokenizer.from_pretrained("emo-lora")
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16 if use_gpu else torch.float32)
model = PeftModel.from_pretrained(model, "emo-lora")
model.eval()

def reply(message, history):
    msgs = [{"role": "system", "content": SYSTEM}]
    for h in history:                      # 兼容新旧两种历史记录格式
        if isinstance(h, dict):            # 新版: {"role":..., "content":...}
            if h.get("content"):
                msgs.append({"role": h["role"], "content": str(h["content"])})
        else:                              # 旧版: (用户说的, 助手答的) 成对
            u, a = h[0], h[1]
            if u:
                msgs.append({"role": "user", "content": str(u)})
            if a:
                msgs.append({"role": "assistant", "content": str(a)})
    msgs.append({"role": "user", "content": message})
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256,
                             do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:],
                      skip_special_tokens=True).strip()

gr.ChatInterface(
    reply,
    title="情感陪伴小助手",
    description="说说你的心事吧，我在听。",
).launch(inbrowser=True)
