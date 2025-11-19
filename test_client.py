# This is a test client for the MCP service.
# And is out of date since the MCP serve changed.
import requests

# 原版本：发送旧的 query 字段请求
# response = requests.post("http://localhost:9000/mcp", json={
#     "query": "前方有没有行人？"
# })

# 新版本（2025-11-11）：发送新的 text/camera_index 结构
payload = {
    # "text": "前方有没有行人？",
    # "camera_index": 1,
    # --- NEW VERSION ---
    "api_version": "v1",
    "text": "前方有没有行人？",
    "camera_index": 1,
}

# response = requests.post("http://localhost:9000/mcp", json=payload)
# --- NEW VERSION ---
response = requests.post("http://localhost:9000/api/v1/mcp", json=payload)

print("Raw response text:", response.text)  # for DEBUG purpose only
print(response.json())