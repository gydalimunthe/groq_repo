import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()
API_URL = "https://api.groq.com/openai/v1/chat/completions"
BASE_DIR = Path(__file__).resolve().parent


class ChatRequest(BaseModel):
    question: str


def load_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key.strip()

    for filename in ("api_key.env", ".env"):
        file_path = BASE_DIR / filename
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                if raw.startswith("GROQ_API_KEY="):
                    return raw.split("=", 1)[1].strip().strip("\"'")
                # Also support file containing only the key value.
                if raw.startswith("gsk_"):
                    return raw

    return ""


def call_groq(question: str) -> str:
    api_key = load_api_key()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ API key not found. Set GROQ_API_KEY or create api_key.env.",
        )

    payload = {
        "messages": [
            {"role": "assistant", "content": "Be a straightforward assistant"},
            {"role": "user", "content": question},
        ],
        "model": "openai/gpt-oss-120b",
        "temperature": 1,
        "max_completion_tokens": 8192,
        "top_p": 1,
        "stream": False,
        "reasoning_effort": "medium",
        "stop": None,
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        return "No response choices returned."
    return choices[0].get("message", {}).get("content", "") or "(Empty response)"


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Groq Chatbot</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 800px; margin: 24px auto; padding: 0 16px; }
          #chat { border: 1px solid #ccc; border-radius: 8px; padding: 12px; min-height: 280px; margin-bottom: 12px; overflow-y: auto; }
          .msg { margin: 8px 0; white-space: pre-wrap; }
          form { display: flex; gap: 8px; }
          input { flex: 1; padding: 10px; }
          button { padding: 10px 14px; }
        </style>
      </head>
      <body>
        <h1>Groq Chatbot</h1>
        <p>Type <b>quit</b> to end the chat.</p>
        <div id="chat"></div>
        <form id="chat-form">
          <input id="question" placeholder="Ask a question..." autocomplete="off" />
          <button type="submit">Send</button>
        </form>
        <script>
          const form = document.getElementById("chat-form");
          const input = document.getElementById("question");
          const chat = document.getElementById("chat");
          let ended = false;

          function append(role, text) {
            const div = document.createElement("div");
            div.className = "msg";
            div.textContent = role + ": " + text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
          }

          form.addEventListener("submit", async (e) => {
            e.preventDefault();
            if (ended) return;
            const question = input.value.trim();
            if (!question) return;
            input.value = "";
            append("You", question);

            const res = await fetch("/chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ question })
            });
            let data = {};
            try {
              data = await res.json();
            } catch {
              data = { detail: "Invalid server response." };
            }
            if (!res.ok) {
              append("Assistant", data.detail || "Error");
              return;
            }
            append("Assistant", data.answer);
            if (data.ended) ended = true;
          });
        </script>
      </body>
    </html>
    """


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if question.lower() == "quit":
        return {"answer": "Chat ended. Refresh the page to start again.", "ended": True}

    try:
        answer = call_groq(question)
        return {"answer": answer, "ended": False}
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        detail = body
        try:
            parsed = json.loads(body)
            detail = parsed.get("error", {}).get("message") or parsed.get("message") or body
        except json.JSONDecodeError:
            detail = body
        raise HTTPException(status_code=err.code, detail=detail) from err
    except urllib.error.URLError as err:
        raise HTTPException(status_code=502, detail=f"Network error: {err.reason}") from err
