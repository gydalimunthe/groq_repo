import json
import os
import sys
import urllib.error
import urllib.request

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"


def call_groq(user_input: str, api_key: str) -> str:
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": "Be a straightforward assistant",
            },
            {
                "role": "user",
                "content": user_input,
            },
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


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: set GROQ_API_KEY before running this script.")
        sys.exit(1)

    print("Groq chat loop started. Type 'quit' to exit.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() == "quit":
            print("Exiting.")
            break

        if not user_input:
            continue

        try:
            answer = call_groq(user_input, api_key)
            print(f"Assistant: {answer}\n")
        except urllib.error.HTTPError as err:
            error_body = err.read().decode("utf-8", errors="replace")
            print(f"HTTP error {err.code}: {error_body}\n")
        except urllib.error.URLError as err:
            print(f"Network error: {err.reason}\n")
        except Exception as err:
            print(f"Unexpected error: {err}\n")


if __name__ == "__main__":
    main()
