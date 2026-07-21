# rag_app.py —— 带 RAG 检索层的情感小助手网页版
# 原理: 用户提问 → 在 data.json 里检索最相近的问答对 → 塞给模型当参考 → 生成回答
# 用法: pip install sentence-transformers   然后 python rag_app.py
import json
import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # 必须和训练时的基座一致
SYSTEM = "你是一位温柔、善解人意的情感陪伴助手，善于共情和倾听，回答温暖真诚。"
TOP_K = 2        # 每次检索几条参考
THRESHOLD = 0.45  # 相似度低于此值就不参考(说明库里没有相近问题)
use_gpu = torch.cuda.is_available()

# ---------- 1. 加载对话模型 ----------
print("正在加载对话模型...")
tok = AutoTokenizer.from_pretrained("emo-lora")
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16 if use_gpu else torch.float32)
model = PeftModel.from_pretrained(model, "emo-lora")
model.eval()

# ---------- 2. 构建向量索引（RAG的"数据库"） ----------
print("正在构建检索索引...")
emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")  # 中文向量模型,约100MB
with open("data.json", encoding="utf-8") as f:
    kb = json.load(f)                                       # 知识库=你的问答对
kb_emb = emb_model.encode([x["instruction"] for x in kb],
                          normalize_embeddings=True)        # 每个问题→一个向量
print(f"索引完成，知识库共 {len(kb)} 条")

def retrieve(query):
    """把用户问题转成向量，和库里所有问题算相似度，返回最相近的几条"""
    q = emb_model.encode([query], normalize_embeddings=True)
    scores = (kb_emb @ q.T).ravel()          # 余弦相似度
    idx = scores.argsort()[::-1][:TOP_K]
    return [(kb[i], float(scores[i])) for i in idx if scores[i] >= THRESHOLD]

# ---------- 3. 生成回答：检索结果拼进提示词 ----------
def reply(message, history):
    hits = retrieve(message)
    system = SYSTEM
    if hits:
        refs = "\n\n".join(f"用户问：{x['instruction']}\n你答：{x['output']}"
                           for x, s in hits)
        system += ("\n\n以下是知识库中与当前问题最相近的标准问答，"
                   "请优先参考其内容和语气来回答：\n" + refs)

    msgs = [{"role": "system", "content": system}]
    for h in history:                        # 兼容新旧版 Gradio 历史格式
        if isinstance(h, dict):
            if h.get("content"):
                msgs.append({"role": h["role"], "content": str(h["content"])})
        else:
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
    answer = tok.decode(out[0][inputs.input_ids.shape[1]:],
                        skip_special_tokens=True).strip()
    if hits:                                  # 底部标注命中情况，方便你调试
        note = "、".join(f"“{x['instruction'][:12]}…”({s:.2f})" for x, s in hits)
        answer += f"\n\n[参考了知识库: {note}]"
    return answer

gr.ChatInterface(
    reply,
    title="情感陪伴小助手（RAG版）",
    description="回答会优先参考 data.json 知识库中最相近的问答。",
).launch(inbrowser=True)
