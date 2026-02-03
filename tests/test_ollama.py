import requests

prompt = "Explain Scope 3 emissions briefly."

res = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    },
    timeout=60
)

res.raise_for_status()
print(res.json()["response"])
