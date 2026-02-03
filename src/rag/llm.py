import requests
import json

OLLAMA_URL = "http://localhost:11434"
MODEL = "phi3:mini"

def call_llm(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True
        },
        stream=True,
        timeout=300
    )

    output = []

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode("utf-8"))
        token = data.get("response", "")
        output.append(token)

        if data.get("done", False):
            break

    return "".join(output)
