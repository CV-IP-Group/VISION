# 该脚本旨在对训练数据集进行增强，请勿重复使用
import random, re

HOMO = {
  "什么": ["甚么","神马"], "广告": ["廣告","ad","AD"],
  "前面": ["面前","正前","前便"], "写": ["寫","写着","寫住"],
  "看看": ["睇下","看下","看一看"], "读": ["讀","读下","念下"],
}
PUNC = r"[，。！？、,.!?]"

def augment(sent):
    s = re.sub(PUNC, "", sent)
    for k, vs in HOMO.items():
        if k in s and random.random() < 0.4:
            s = s.replace(k, random.choice(vs))
    # 随机删除/插入
    if random.random() < 0.2 and len(s) > 4:
        i = random.randrange(len(s))
        s = s[:i] + s[i+1:]
    if random.random() < 0.2:
        s = s + random.choice([" 呀", " 啊", " 呢", ""])
    return s
