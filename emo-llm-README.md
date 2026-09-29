# Emo-LLM · 情感陪伴对话小模型

> 基于 Qwen2.5-1.5B + LoRA 微调的中文情感陪伴助手，从数据构造、模型微调到 Web 部署的端到端实现。
> An end-to-end Chinese emotional-support chatbot: synthetic data generation → LoRA fine-tuning → web deployment.

[中文](#中文) | [English](#english)

---

## 中文

### 项目简介

Emo-LLM 是一个能够共情倾听、温柔回应的中文对话助手。用户可以向它倾诉压力、失恋、焦虑等情绪困扰，它会先认可情绪、再表达理解、最后给出温和的建议或追问，而不是生硬地讲道理。

做这个项目的目的，是完整走通「用一个开源小模型 + 少量数据，打造一个有特定风格与人格的对话 AI」这条主流技术路径，并把它部署成人人可用的网页应用。项目刻意选用参数量仅 15 亿的小模型，证明在消费级笔记本（无需专业 GPU）上也能完成从训练到上线的全流程。

### 技术栈

| 环节 | 技术选型 | 说明 |
|---|---|---|
| 基座模型 | Qwen2.5-1.5B-Instruct | 阿里开源，中文能力强，可在本地运行 |
| 微调方法 | LoRA (PEFT) | 仅训练约 0.14% 参数，消费级设备即可完成 |
| 训练框架 | Hugging Face Transformers + Trainer | 标准化训练流程 |
| 数据构造 | 大模型蒸馏 (DeepSeek API) | 按主题批量生成高质量情感对话 |
| 推理部署 | Gradio | 快速搭建 Web 聊天界面 |
| 运行环境 | Python 3 + PyTorch (CPU/MPS/CUDA 自适应) | 有无 GPU 均可运行 |

### 实现原理

整个项目分为三个阶段：

**1. 数据构造（`gen_data.py`）**
采用「模型蒸馏」策略：调用 DeepSeek API，围绕工作压力、失恋、孤独、焦虑等 12 个情感主题批量生成对话数据，每条为「用户倾诉 → 共情式回复」的问答对，自动去重后汇入 `data.json`。这种方式以极低成本获得了风格统一、覆盖全面的训练语料。

**2. 模型微调（`train.py`）**
在 Qwen2.5-1.5B 基座上挂载 LoRA 适配器，仅对注意力层的 q/k/v/o 投影矩阵进行低秩微调。训练时套用模型的对话模板，注入统一的系统人格设定。全程只训练约 200 万参数（占总参数 0.14%），产出的适配器权重仅数 MB，可与基座模型解耦分发。

```
data.json ──► apply_chat_template ──► Qwen2.5-1.5B + LoRA ──► emo-lora/ (适配器权重)
```

**3. 推理部署（`chat.py` / `app.py`）**
加载基座模型并叠加训练好的 LoRA 权重进行推理。`chat.py` 提供命令行对话，`app.py` 基于 Gradio 提供网页聊天界面，支持多轮上下文记忆。生成采用温度采样（temperature=0.7）以保证回复的自然与多样。

### 核心功能

- **情感共情对话**：识别用户情绪，给出「认可情绪 → 表达理解 → 温和建议」结构化的温暖回应
- **多轮上下文记忆**：记住对话历史，追问时结合前文给出连贯回答
- **可扩展知识增强（RAG，可选）**：`rag_app.py` 额外实现了检索增强，回答时可参考知识库中最相近的标准问答，提升风格一致性
- **网页交互界面**：基于 Gradio 的聊天页面，浏览器直接访问，手机端亦可使用
- **完全本地训练**：无需云端 GPU，消费级笔记本约 1 小时内完成微调

### 项目结构

```
emo-llm/
├── gen_data.py      # 数据构造：调用大模型 API 批量生成情感对话
├── data.json        # 训练数据集（情感倾诉—共情回复 问答对）
├── train.py         # LoRA 微调脚本
├── chat.py          # 命令行推理对话
├── app.py           # Gradio 网页聊天界面
├── rag_app.py       # （可选）带 RAG 检索增强的网页版
├── requirements.txt # 依赖清单
└── emo-lora/        # 训练产出的 LoRA 适配器权重
```

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. （可选）构造更多训练数据
export DEEPSEEK_API_KEY="your-key"
python gen_data.py

# 3. 微调模型
python train.py          # 产出 emo-lora/ 适配器权重

# 4. 启动对话
python app.py            # 网页版，浏览器自动打开
# 或 python chat.py      # 命令行版
```

### 设计取舍与思考

- **为什么用小模型**：项目意在验证「小模型 + 微调」范式的可行性，1.5B 模型在共情对话这类风格化任务上表现足够，且训练成本极低。
- **微调 vs RAG**：微调改变的是模型「说话的风格与人格」，知识固化在权重中；本项目同时实现了 RAG 版本作为对比，二者代表大模型应用开发的两大主流范式。
- **已知局限**：受参数量限制，模型在事实性问答、复杂推理上能力有限；身份类问题（如「你是谁」）因训练数据未覆盖可能产生幻觉，可通过补充相应训练数据修正。

---

## English

### Overview

Emo-LLM is a Chinese conversational assistant designed for empathetic listening and emotional support. Users can share stress, heartbreak, or anxiety, and the model responds by first validating the emotion, then expressing understanding, and finally offering gentle suggestions or follow-up questions — rather than delivering blunt advice.

The goal of this project was to work through the complete, mainstream pipeline of **building a persona-specific conversational AI from an open-source small model and a modest amount of data**, and to deploy it as an accessible web application. A 1.5B-parameter model was chosen deliberately to demonstrate that the entire workflow — from training to deployment — runs on a consumer laptop without a dedicated GPU.

### Tech Stack

| Stage | Choice | Notes |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | Open-source, strong Chinese ability, runs locally |
| Fine-tuning | LoRA (PEFT) | Trains ~0.14% of parameters; consumer-grade hardware |
| Training | Hugging Face Transformers + Trainer | Standard training loop |
| Data | LLM distillation (DeepSeek API) | Topic-driven synthetic dialogue generation |
| Deployment | Gradio | Rapid web chat interface |
| Runtime | Python 3 + PyTorch (CPU/MPS/CUDA adaptive) | Runs with or without GPU |

### How It Works

**1. Data Generation (`gen_data.py`)** — A distillation strategy: the DeepSeek API generates dialogue pairs across 12 emotional themes (work stress, breakups, loneliness, anxiety, etc.), each a "user confides → empathetic reply" pair, deduplicated and merged into `data.json`. This yields stylistically consistent, broadly-covering training data at minimal cost.

**2. Fine-tuning (`train.py`)** — A LoRA adapter is attached to the Qwen2.5-1.5B base, applying low-rank updates only to the attention q/k/v/o projection matrices. Training applies the model's chat template with a unified system persona. Only ~2M parameters (0.14% of the total) are trained; the resulting adapter is a few MB and ships decoupled from the base model.

**3. Deployment (`chat.py` / `app.py`)** — Inference loads the base model and overlays the trained LoRA weights. `chat.py` offers a CLI; `app.py` provides a Gradio web chat with multi-turn memory. Temperature sampling (0.7) keeps replies natural and varied.

### Features

- **Empathetic dialogue** with a structured "validate → understand → gently advise" response pattern
- **Multi-turn memory** for coherent follow-up answers
- **Optional RAG augmentation** (`rag_app.py`) that references the closest reference Q&A to reinforce style consistency
- **Web interface** via Gradio, accessible from browser and mobile
- **Fully local training** — no cloud GPU; ~1 hour on a consumer laptop

### Project Structure

```
emo-llm/
├── gen_data.py      # Data generation via LLM API
├── data.json        # Training dataset (confide–empathize pairs)
├── train.py         # LoRA fine-tuning
├── chat.py          # CLI inference
├── app.py           # Gradio web chat
├── rag_app.py       # (Optional) RAG-augmented web version
├── requirements.txt
└── emo-lora/        # Trained LoRA adapter weights
```

### Quick Start

```bash
pip install -r requirements.txt

export DEEPSEEK_API_KEY="your-key"   # optional, to generate more data
python gen_data.py

python train.py     # produces emo-lora/
python app.py       # web UI, opens automatically
```

### Design Notes

- **Why a small model** — to validate the "small model + fine-tuning" paradigm; 1.5B is sufficient for a stylistic task like empathetic conversation, at very low training cost.
- **Fine-tuning vs RAG** — fine-tuning shapes *how the model speaks* (persona baked into weights); a RAG variant is included for contrast, together covering the two mainstream paradigms of LLM application development.
- **Known limitations** — bounded by parameter count on factual and reasoning-heavy tasks; identity questions ("who are you") may hallucinate where training data is absent, correctable by adding such examples.

---

*Author: [@zjb2005](https://github.com/zjb2005) · Built with Qwen2.5, PEFT, Transformers & Gradio*
