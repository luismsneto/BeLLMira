import requests
import json
import shlex

class ModelClient:
    """HTTP client for interacting with an OpenAI-compatible inference server."""

    def __init__(self, base_url="http://localhost:8080/"):
        """
        Args:
            base_url: Base URL of the inference server.
        """
        self.base_url = base_url
        self.session = requests.Session()

    def get_model_name(self) -> str:
        """Return the ID of the first model reported by the server."""
        url = self.base_url + "v1/models/"
        response = self.session.get(url)
        response.raise_for_status()

        try:
            return response.json()["data"][0]["id"]
        except (KeyError, IndexError, ValueError) as e:
            raise Exception(f"Could not parse model name: {e}")

    def get_models_name_list(self) -> list[str]:
        """Return the IDs of all models reported by the server."""
        url = self.base_url + "v1/models/"
        response = self.session.get(url)
        response.raise_for_status()
        try:
            return [model['id'] for model in response.json()["data"]]
        except(KeyError, IndexError, ValueError) as e:
            raise Exception(f"Could not parse model name: {e}")
        
    @staticmethod
    def is_valid_json(data):
        """Return True if *data* can be serialised to JSON without error."""
        try:
            json.dumps(data)
            return True
        except (ValueError, TypeError):
            return False

    def build_chat_request(
        self,
        user_prompt,
        system_prompt=None,
        model_name="/app/model/model",
        temperature=0.0,
        max_tokens=1000,
        json_schema=None,
        use_guided_json: bool = False,
        assistant_messages=None,
        image_prompt=None,
        enable_thinking: bool = None,
    ):
        """Build an unsigned POST request for the chat completions endpoint.

        Args:
            user_prompt: The user turn text.
            system_prompt: Optional system message prepended to the conversation.
            model_name: Model path or identifier served by the backend.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_schema: JSON schema dict for structured output.
            use_guided_json: When True, send schema via `guided_json` (vLLM)
                instead of `response_format`.
            assistant_messages: Prior assistant turns to include for multi-turn chat.
            image_prompt: Base-64 encoded PNG to attach as a vision input.
            enable_thinking: When set, enables or disables chain-of-thought via
                `chat_template_kwargs`.

        Returns:
            requests.Request ready to be prepared and sent.
        """
        url = self.base_url + "v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if assistant_messages:
            messages.extend(assistant_messages)

        if image_prompt:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_prompt}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": user_prompt})

        headers = {'Content-Type': 'application/json'}
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature
        }
        if json_schema is not None:
            if use_guided_json:
                data["guided_json"] = json_schema
            else:
                data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": json_schema
                    }
                }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if enable_thinking is not None: 
            #data["enable_thinking"] = enable_thinking
            data["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if not self.is_valid_json(data):
            raise Exception("Invalid request data for chat")
        return requests.Request("POST", url, headers=headers, json=data)

    def build_embedding_request(
        self,
        input_text,
        model_name="/app/model/embedding",
        user_id=None
    ):
        """Build an unsigned POST request for the embeddings endpoint.

        Args:
            input_text: Text or list of texts to embed.
            model_name: Embedding model path or identifier.
            user_id: Optional end-user identifier forwarded to the server.

        Returns:
            requests.Request ready to be prepared and sent.
        """
        url = self.base_url + "v1/embeddings"
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": model_name,
            "input": input_text
        }
        if user_id:
            data["user"] = user_id

        if not self.is_valid_json(data):
            raise Exception("Invalid request data for embeddings")

        return requests.Request("POST", url, headers=headers, json=data)

    def build_rerank_request(
        self,
        query,
        documents,
        modelname="/app/model/rerank",
        top_n=None
    ):
        """Build an unsigned POST request for the reranking endpoint.

        Args:
            query: Query string used to score relevance.
            documents: List of candidate document strings to rerank.
            modelname: Reranker model path or identifier.
            top_n: If set, return only the top-n ranked documents.

        Returns:
            requests.Request ready to be prepared and sent.
        """
        url = self.base_url + "v1/rerank"
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": modelname,
            "query": query,
            "documents": documents
        }
        if top_n is not None:
            data["top_n"] = top_n

        if not self.is_valid_json(data):
            raise Exception("Invalid request data for reranking")

        return requests.Request("POST", url, headers=headers, json=data)

    def send_request(self, request):
        """Prepare and send a requests.Request, returning the response."""
        prepared = self.session.prepare_request(request)
        return self.session.send(prepared)

    def stream_chat_response(self, req: requests.Request):
        """
        Streams a chat response from a requests.Request.

        Args:
            req: requests.Request created via build_chat_request.

        Yields:
            str: text chunks generated by the model.
        """
        req.json["stream"] = True
        prepped = self.session.prepare_request(req)

        with self.session.send(prepped, stream=True) as response:
            response.raise_for_status() 

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: "):
                    decoded = decoded[len("data: "):]
                if decoded == "[DONE]":
                    break
                try:
                    chunk = json.loads(decoded)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0]["delta"]
                text = delta.get("content")
                if text:
                    yield text

    def print_curl(self, request):
        """Return the request as an equivalent curl command string."""
        prepared = self.session.prepare_request(request)
        parts = ["curl", "-X", prepared.method]

        for k, v in prepared.headers.items():
            if k.lower() == "content-length":
                continue  # Skip Content-Length
            parts += ["-H", f"{shlex.quote(f'{k}: {v}')}"]

        if prepared.body:
            body = prepared.body
            if isinstance(body, bytes):
                body = body.decode('latin-1')
            parts += ["--data", shlex.quote(body)]

        parts += [shlex.quote(prepared.url)]

        return " ".join(parts)

    def print_json(self, request):
        """Return the request body as a pretty-printed JSON string."""
        prepared = self.session.prepare_request(request)
        body = prepared.body
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        try:
            parsed = json.loads(body)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return body or "{}"

    def print_invoke_webrequest(self, request):
        """Return the request as an equivalent PowerShell Invoke-WebRequest command string."""
        prepared = self.session.prepare_request(request)

        parts = ["$body = ("]

        body = prepared.body
        if isinstance(body, bytes):
            body = body.decode('latin-1')

        parsed = json.loads(body)

        def to_ps(obj):
            if isinstance(obj, dict):
                return "@{ " + "; ".join(f"{k} = {to_ps(v)}" for k, v in obj.items()) + " }"
            elif isinstance(obj, list):
                return "@(" + ", ".join(to_ps(i) for i in obj) + ")"
            elif isinstance(obj, str):
                return '"' + obj.replace('"', '`"') + '"'
            else:
                return json.dumps(obj)
        ps_hashtable = to_ps(parsed)
        parts.append(ps_hashtable + ") | ConvertTo-Json -Depth 10 -Compress")

        # Now the command
        cmd = [
            "Invoke-WebRequest",
            "-Uri", f'"{prepared.url}"',
            "-Method", prepared.method,
        ]

        headers = {
            k: v for k, v in prepared.headers.items()
            if k.lower() not in ["content-length", "connection"]
        }
        if headers:
            hdrs = "; ".join(f'"{k}" = "{v}"' for k, v in headers.items())
            cmd += ["-Headers", f"@{{ {hdrs} }}"]

        cmd += ["-Body", "([System.Text.Encoding]::UTF8.GetBytes($body))"]

        return "\n".join(parts + [""] + [" ".join(cmd)])

    def print_request_cmds(self, request):
        """Print the request as both a curl command and a PowerShell Invoke-WebRequest command."""
        print("# curl")
        print(self.print_curl(request))
        print()
        print("# Invoke-WebRequest (PowerShell)")
        print(self.print_invoke_webrequest(request))