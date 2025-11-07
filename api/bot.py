import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

gpt_model = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store_id = os.getenv("VECTORDBID")
ENABLE_GUARDRAILS = False

INSTRUCTIONS = """
You are the official chatbot for the Santa Clara University Engineering Computing Center (ECC) Lab. Your role is to generate accurate, copy-ready responses using only verified information from the ECC Lab knowledge base documents.

Behavior Rules:

1. Interaction Modes:
- Most of the time, the user will paste a client's question directly. In this case, generate a professional, concise response that the user can copy and paste to reply to the client. Write it in a clear, helpful tone appropriate for technical support.
- Other times, the user may ask you a question directly. In this case, answer the user clearly and factually, following the same grounding and citation rules.

2. Knowledge Restriction:
- Use only information found within the ECC Lab knowledge base (uploaded internal documents).
- Never use or infer information from outside sources, memory, or general knowledge.
- If the knowledge base does not contain an answer, respond exactly with:
  "No entry found in the ECC Lab knowledge base for this query."

3. Retrieval Policy:
- Search all ECC Lab knowledge base documents.
- Extract or paraphrase relevant sections accurately.
- If multiple documents are used, list all filenames at the end.

4. Answer Format:
- Responses must be concise, factual, and easy to copy and send to clients.
- Use Markdown formatting for bullet points, file paths, or commands.
- At the end of every answer, include:
  Documents Referred:
  - <file1>
  - <file2>
"""


def _json_response(status: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(payload),
    }


def _parse_body(request: Any) -> Optional[Dict[str, Any]]:
    if hasattr(request, "get_json"):
        try:
            return request.get_json()
        except Exception:
            pass

    body = getattr(request, "body", None) or getattr(request, "data", None)
    if isinstance(body, bytes):
        body = body.decode("utf-8")

    if isinstance(body, str) and body.strip():
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None

    return None


def _check_guardrails(query: str) -> bool:
    objective = (
        "Santa Clara University (SCU), Provost, Education advising, courses, "
        "academic policies, student advising and support or a related question"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You will receive a user query and your task is to classify if a "
                    f"given user request is related to {objective}. If it is relevant, "
                    "return `1`. Else, return `0`"
                ),
            },
            {"role": "user", "content": query},
        ],
        seed=0,
        temperature=0,
        max_tokens=1,
        logit_bias={"15": 100, "16": 100},
    )
    return int(response.choices[0].message.content) == 1


def _generate_response(query: str) -> Dict[str, Any]:
    if not vector_store_id:
        return _json_response(
            500, {"error": "VECTORDBID environment variable not set."}
        )

    try:
        response = client.responses.create(
            model=gpt_model,
            input=query,
            temperature=0.0,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                    "max_num_results": 10,
                }
            ],
            include=["file_search_call.results"],
            instructions=INSTRUCTIONS,
        )

        message_text = next(
            (
                item.content[0].text
                for item in response.output
                if item.type == "message"
            ),
            None,
        )

        if message_text:
            return _json_response(200, {"response": message_text})

        return _json_response(500, {"response": "Error in generating response."})
    except Exception as exc:
        return _json_response(500, {"error": str(exc)})


def handler(request: Any) -> Dict[str, Any]:
    method = getattr(request, "method", "GET").upper()

    if method == "OPTIONS":
        return {
            "statusCode": 204,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "body": "",
        }

    if method != "POST":
        return _json_response(
            405, {"error": "Method not allowed", "allowed": ["POST", "OPTIONS"]}
        )

    payload = _parse_body(request)
    if not payload or "query" not in payload:
        return _json_response(
            400, {"error": "Request body must be JSON and contain a 'query' key."}
        )

    query = payload["query"]
    if ENABLE_GUARDRAILS and not _check_guardrails(query):
        return _json_response(
            200, {"response": "Sorry, i cannot help you with that!"}
        )

    return _generate_response(query)

