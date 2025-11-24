
import joblib, json
from typing import Any, Dict, Optional

# 意图类别到函数映射
# 原始基于 sklearn 的意图到函数映射
MAP = {
    "object_ahead":    {"function":"vision.detect_objects", "defaults":{}},
    "read_sign_text":  {"function":"vision.ocr_read",       "defaults":{}},
    "describe_scene":  {"function":"vision.describe_scene",  "defaults":{}},
    "guide_direction": {"function":"vision.describe_scene",  "defaults":{}},
    "hazard_alert":    {"function":"vision.check_hazards",   "defaults":{}},
    "distance_query":  {"function":"vision.distance_query",  "defaults":{}},
    "repeat_last":     {"function":"system.repeat",         "defaults":{}},
    "none":            {"function":"system.ignore",         "defaults":{}}
}

# 加载模型和阈值
def load_model(model_path='models/intent_sklearn.joblib',threshold_path='models/intent_threshold.json'):
    mdl = joblib.load(model_path)
    with open(threshold_path,'r',encoding='utf-8') as f:
        TAU = json.load(f)['tau_on']
    return mdl, TAU

#  预测意图
def predict_intent(text: str, mdl, TAU=0.5):
    X = mdl['vec'].transform([text])
    proba = mdl['clf'].predict_proba(X)[0]
    classes = mdl['classes']
    idx = proba.argmax()
    label, p = classes[idx], float(proba[idx])
    if p < TAU: label = 'none'
    return label, p

_RULES = [
    {
        "keywords": [
            "在哪", "找", "左边", "右边", "哪里", "位置", "搜", "查",
            "find", "where", "locate", "position", "left", "right",
            "边度", "喺边", "揾", "左手边", "右手边", "稳"
        ],
        "intent": "locate",
        "function": "vision.detect_objects",
        "action": "call"
    },
    {
        "keywords": [
            "读", "念", "字", "写了什么", "文字", "文章", "书",
            "read", "text", "say", "what does it say", "scan",
            "读下", "写乜", "念出黎"
        ],
        "intent": "read",
        "function": "vision.ocr_read",
        "action": "call"
    },
    {
        "keywords": [
            "危险", "障碍", "能走吗", "撞", "安全", "路况", "过马路",
            "safe", "hazard", "obstacle", "watch out", "clear",
            "危唔危险", "会唔会撞", "行唔行得", "阻住"
        ],
        "intent": "safety",
        "function": "vision.check_hazards",
        "action": "call"
    },
    {
        "keywords": [
            "描述", "环境", "周围", "前面", "前边", "面前", "看见什么", "场景",
            "describe", "what is", "surrounding", "scene", "front",
            "讲下周围", "环境点样", "前面有乜", "睇下", "附近有乜"
        ],
        "intent": "describe",
        "function": "vision.describe_scene",
        "action": "call"
    }
]


def _rule_route(text: str) -> Optional[Dict[str, Any]]:
    t = text.lower()
    for rule in _RULES:
        if any(k.lower() in t for k in rule["keywords"]):
            return {
                "action": rule.get("action", "call"),
                "function": rule["function"],
                "intent": rule["intent"],
                "args": {"raw_text": text, "confidence": 1.0},
            }
    return None


def keyword_route(text: str) -> Optional[Dict[str, Any]]:
    return _rule_route(text)


# 路由到具体函数调用(返回JSON)
def route_mcp(text: str, mdl, TAU=0.5, function_map=MAP, skip_rules: bool = False):
    # 1) 规则优先处理核心用例
    if not skip_rules:
        routed = _rule_route(text)
        if routed is not None:
            return routed

        # 2) 回退到原来的 sklearn 模型
        label, p = predict_intent(text, mdl, TAU)
        spec = function_map.get(label, function_map['none'])
        if not spec['function']:
                return {"action":"clarify","function":"None", "msg":"请问你是想让我‘读广告’还是‘看看前面’？", "p":p}
        payload = {**spec['defaults'], "raw_text": text, "confidence": p}
        return {"action":"call", "function":spec['function'], "args":payload}
