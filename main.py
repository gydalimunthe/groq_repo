import json
import os
import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

app = FastAPI(title="Groq Chat App")


class ChatRequest(BaseModel):
    question: str


def call_groq(user_input: str, api_key: str) -> str:
    payload = {
        "messages": [
            {"role": "assistant", "content": "Be a straightforward assistant"},
            {"role": "user", "content": user_input},
        ],
        "model": MODEL,
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
        return f"No response choices returned. Raw response: {data}"

    message = choices[0].get("message", {})
    return message.get("content", "") or "(Empty response)"


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Groq Chat</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 760px; margin: 32px auto; padding: 0 16px; }
      #chat { border: 1px solid #ddd; border-radius: 8px; padding: 12px; min-height: 260px; margin-bottom: 12px; overflow-y: auto; }
      .msg { margin: 8px 0; white-space: pre-wrap; }
      .you { color: #1f2937; }
      .assistant { color: #0f766e; }
      .muted { color: #666; font-size: 13px; }
      form { display: flex; gap: 8px; }
      input { flex: 1; padding: 10px; }
      button { padding: 10px 14px; cursor: pointer; }
    </style>
  </head>
  <body>
    <h1>Groq Chat</h1>
    <p class="muted">Ask anything. Type <b>quit</b> to stop chatting.</p>
    <div id="chat"></div>
    <form id="chat-form">
      <input id="question" placeholder="Type your question..." autocomplete="off" />
      <button type="submit">Send</button>
    </form>

    <script>
      const chat = document.getElementById("chat");
      const form = document.getElementById("chat-form");
      const questionInput = document.getElementById("question");
      let ended = false;

      function append(role, text, cssClass) {
        const div = document.createElement("div");
        div.className = "msg " + cssClass;
        div.textContent = role + ": " + text;
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
      }

      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (ended) return;

        const question = questionInput.value.trim();
        if (!question) return;
        questionInput.value = "";
        append("You", question, "you");

        try {
          const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
          });
          const data = await res.json();
          if (!res.ok) {
            append("Assistant", data.detail || "Request failed", "assistant");
            return;
          }
          append("Assistant", data.answer, "assistant");
          if (data.ended) ended = true;
        } catch (err) {
          append("Assistant", "Network error: " + err, "assistant");
        }
      });
    </script>
  </body>
</html>
"""


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if question.lower() == "quit":
        return {"answer": "Session ended. Refresh page to start again.", "ended": True}

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set.")

    try:
        answer = call_groq(question, api_key)
        return {"answer": answer, "ended": False}
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=err.code, detail=body) from err
    except urllib.error.URLError as err:
        raise HTTPException(status_code=502, detail=f"Network error: {err.reason}") from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {err}") from err
