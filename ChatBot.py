import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from file_attachments import prepare_attachments
from image_generation import generate_image


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
MAX_REQUEST_SIZE = 32 * 1024 * 1024
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Give clear, friendly, concise answers."
)
HOST = "127.0.0.1"
PORT = 8000
PAGE_PATH = Path(__file__).resolve().with_name("index.html")


class ChatHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404, "Not found")
            return

        try:
            body = PAGE_PATH.read_bytes()
        except OSError:
            self.send_error(500, "Could not read index.html")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/api/image":
            self._handle_image_generation()
            return
        if self.path != "/api/chat":
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 0 or content_length > MAX_REQUEST_SIZE:
                self._send_json(413, {"error": "The request is too large."})
                return
            request_body = self.rfile.read(content_length)
            data = json.loads(request_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid JSON request."})
            return

        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            self._send_json(400, {"error": "Messages must be a list."})
            return

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        for message in messages[-12:]:
            if (
                not isinstance(message, dict)
                or message.get("role") not in ("user", "assistant")
                or not isinstance(message.get("content"), str)
            ):
                self._send_json(400, {"error": "Invalid chat message."})
                return
            try:
                attachment_text = prepare_attachments(message.get("files", []))
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            content = message["content"]
            if attachment_text:
                content = (
                    f"{content}\n\n{attachment_text}" if content else attachment_text
                )
            conversation.append({"role": message["role"], "content": content})

        payload = {
            "model": MODEL,
            "messages": conversation,
            "stream": False,
        }
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = result["message"]["content"]
            if not isinstance(answer, str):
                raise ValueError("Ollama returned an invalid message.")
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            self._send_json(
                502,
                {"error": f"Ollama returned HTTP {error.code}: {details}"},
            )
            return
        except urllib.error.URLError:
            self._send_json(
                502,
                {
                    "error": (
                        "Cannot connect to Ollama. Make sure Ollama is running "
                        "and the model is installed."
                    )
                },
            )
            return
        except (TimeoutError, json.JSONDecodeError, KeyError, ValueError):
            self._send_json(
                502,
                {
                    "error": (
                        "The AI response timed out or could not be read. "
                        "Please try again."
                    )
                },
            )
            return

        self._send_json(200, {"answer": answer})

    def _handle_image_generation(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 0 or content_length > 64 * 1024:
                self._send_json(413, {"error": "The image prompt is too large."})
                return
            data = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid image-generation request."})
            return

        prompt = data.get("prompt") if isinstance(data, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            self._send_json(400, {"error": "Enter a prompt for the image."})
            return
        if len(prompt) > 2000:
            self._send_json(
                400,
                {"error": "The image prompt must be 2000 characters or fewer."},
            )
            return

        try:
            image = generate_image(prompt)
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            self._send_json(
                502,
                {
                    "error": (
                        f"Image service returned HTTP {error.code}: "
                        f"{details[:1000]}"
                    )
                },
            )
            return
        except urllib.error.URLError:
            self._send_json(
                502,
                {
                    "error": (
                        "Cannot connect to the image service at "
                        "http://127.0.0.1:7860. Start Stable Diffusion WebUI "
                        "with --api and check its endpoint."
                    )
                },
            )
            return
        except TimeoutError:
            self._send_json(
                504,
                {"error": "Image generation timed out. Please try again."},
            )
            return
        except (json.JSONDecodeError, ValueError) as error:
            self._send_json(502, {"error": str(error)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header(
            "Content-Disposition",
            'attachment; filename="generated-image.png"',
        )
        self.send_header("Content-Length", str(len(image)))
        self.end_headers()
        self.wfile.write(image)

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"Chatbot is running at {url}")
    print("Press Ctrl+C to stop the server.")

    import webbrowser

    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping chatbot server.")
    finally:
        server.server_close()
