from datetime import datetime, timezone
from typing import Any, Dict


def handler(request: Any) -> Dict[str, Any]:
    """
    Sample endpoint served at `/` (rewritten via vercel.json) that verifies server-side
    execution within the Vercel deployment.
    """
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "no-store, max-age=0",
        },
        "body": (
            '{"ok": true, "message": "index handler pong", "timestamp": "'
            + datetime.now(timezone.utc).isoformat()
            + '"}'
        ),
    }
from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
vector_store_details = {
    "id": os.getenv("VECTORDBID"),
}
vector_store_id = os.getenv("VECTORDBID")
app = Flask(__name__)
gpt_model = "gpt-4o"
api_key=os.getenv("OPENAI_API_KEY")
#print(api_key)
client = OpenAI(api_key=api_key)

#toggle guard rails here if any complications in the generations.
#The main idea is that we use a request to determine relevancy first, then pass on to generation.
enableGuardrails = False

def checkInput(input):
    objective = "Santa Clara University (SCU), Provost, Education advising, courses, academic policies, student advising and support or a related question"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": f"You will receive a user query and your task is to classify if a given user request is related to {objective}. If it is relevant, return `1`. Else, return `0`"},
        {"role": "user", "content": input},
        ],
        seed=0,
        temperature=0,
        max_tokens=1,
        logit_bias={"15": 100, #token ID for `0` 
                    "16": 100})  #token ID for `1`
    #print(response)
    return int(response.choices[0].message.content)

def generateResponse(query,vector_store_id):
        
    instructions = """
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

Example 1:
User: "Client: I cannot connect to my ECC lab machine remotely."
Assistant:
Please verify that the client is connected to the SCU VPN, then use the SSH command:
ssh your_username@linux.dc.engr.scu.edu
If the error persists, ensure the account credentials are correct.
Documents Referred:
- ECC_Remote_Login_Guide.pdf

Example 2:
User: "What time does the ECC lab close?"
Assistant:
The ECC Lab is open Monday to Friday, 8:00 AM to 8:00 PM during academic terms. Hours may vary on holidays.
Documents Referred:
- ECC_Lab_Hours_and_Policies.docx
"""


    try:
        app.logger.info("Generating response for query: %s", query)
        response = client.responses.create(
            model=gpt_model,
            input=query,
            temperature=0.0,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": 10,
            }],
            include=["file_search_call.results"],
            instructions=instructions,
        )
        
        # Extract the message text from the response
        message_text = next((item.content[0].text for item in response.output if item.type == 'message'), None) # type: ignore

        if message_text:
            app.logger.info("Response generated successfully for query.")
            return jsonify({"response": message_text})
        else:
            app.logger.error("Response missing message text for query.")
            return jsonify({"response": "Error in generating response."}), 500

    except Exception as e:
        # Handle potential API errors
        app.logger.exception("Error generating response: %s", e)
        return jsonify({"error": str(e)}), 500

# Define the main endpoint for processing user queries
@app.route('/bot', methods=['POST', 'OPTIONS'])
def get_response():
    if request.method == 'OPTIONS':
        return ('', 204)

    # Get the JSON data from the request body
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Request body must be JSON and contain a 'query' key."}), 400

    query = data['query']
    isValid = 0
    if enableGuardrails == True:
        isValid = checkInput(query)

    #print(isValid)
    if not vector_store_id:
        return jsonify({"error": "VECTORDBID environment variable not set."}), 500

    if (enableGuardrails == True and isValid == 1) or enableGuardrails == False:
        return generateResponse(query,vector_store_id)
    else:
        return jsonify({"response": "Sorry, i cannot help you with that!"})

# Apply permissive CORS headers to all responses
@app.after_request
def apply_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Define a health check endpoint, default
@app.route('/')
def check():
    """
    A simple endpoint to confirm that the Flask application is running.
    """
    return jsonify({"status": "ok", "message": "Provost Bot API is running."})


# Note: For local development, you might add the following lines.
#if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3001, debug=True)


