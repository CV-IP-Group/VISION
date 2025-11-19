
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
        "keywords": ["障碍", "前面有没有", "危险", "路上有什么"],
        "intent": "hazard_alert",
        "function": "vision.check_hazards",
    },
    {
        "keywords": ["帮我读", "上面写了什么", "读这个", "这句话是什么"],
        "intent": "read_sign_text",
        "function": "vision.ocr_read",
    },
    {
        "keywords": ["描述", "描述一下周围", "环境", "环境怎么样", "describe"],
        "intent": "describe_scene",
        "function": "vision.describe_scene",
    },
    {
        "keywords": ["前面有什么", "帮我看看", "周围有什么", "describe"],
        "intent": "object_ahead",
        "function": "vision.detect_objects",
    },
]


def _rule_route(text: str) -> Optional[Dict[str, Any]]:
    t = text.lower()
    for rule in _RULES:
        if any(k.lower() in t for k in rule["keywords"]):
            return {
                "action": "call",
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
