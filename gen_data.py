# gen_data.py —— 调用大模型 API 批量生成情感对话训练数据（蒸馏）
# 用法: 1) pip install openai   2) 填入下方 API_KEY   3) python gen_data.py
# 生成结果自动合并进 data.json（去重），之后重新 python train.py 即可
import json
import re
from openai import OpenAI

# ======== 只需改这三行 ========
API_KEY = "sk-aa901d8ebbd947dba66f4341dd2493c6"
BASE_URL = "https://api.deepseek.com"  # DeepSeek。用通义千问则改为:
#            https://dashscope.aliyuncs.com/compatible-mode/v1 (模型名 qwen-plus)
GEN_MODEL = "deepseek-chat"
# =============================
        
PER_TOPIC = 50   # 每个主题生成条数，12个主题 x 50 = 600/次
TOPICS = ["工作压力", "失恋分手", "家庭矛盾", "孤独社恐", "学业考试压力",
          "自卑内耗", "失眠焦虑", "亲人或宠物离世", "婚姻育儿", "职场人际",
          "容貌身材焦虑", "迷茫与无意义感"]

PROMPT = """请生成{n}条中文情感陪伴对话训练数据，主题：{topic}。
要求：
1. instruction 是用户的一句倾诉，口语化、场景具体、长短不一
2. output 是助手的温暖回复，3-5句：先共情认可情绪，再表达理解，最后给轻建议或温和提问，不能有明显AI痕迹，语言结构不能太结构化，不能死板，要语气温柔友善感性可人，像一位知心大姐姐一样安慰用户的情绪，体现出人类的灵魂和情感
3. {n}条内容彼此不重复
严格输出 JSON 数组，格式 [{{"instruction":"...","output":"..."}}]，不要输出任何其他文字。"""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
new_data = []

for t in TOPICS:
    print(f"正在生成主题: {t} ...")
    try:
        r = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user",
                       "content": PROMPT.format(n=PER_TOPIC, topic=t)}],
            temperature=1.0)
        text = r.choices[0].message.content
        items = json.loads(re.search(r"\[.*\]", text, re.S).group())
        good = [x for x in items if x.get("instruction") and x.get("output")]
        new_data += good
        print(f"  成功 {len(good)} 条")
    except Exception as e:
        print(f"  该主题失败，已跳过: {e}")

# 合并旧数据并按 instruction 去重
try:
    with open("data.json", encoding="utf-8") as f:
        old = json.load(f)
except FileNotFoundError:
    old = []

seen, merged = set(), []
for x in old + new_data:
    if x["instruction"] not in seen:
        seen.add(x["instruction"])
        merged.append(x)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=1)
print(f"完成！本次新增 {len(merged) - len(old)} 条，data.json 现共 {len(merged)} 条")
