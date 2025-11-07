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
gpt_model = "ftjob-v2jtP1SfRKjozXT8UwEh1Tzh"
client = OpenAI()

#toggle guard rails here if any complications in the generations.
#The main idea is that we use a request to determine relevancy first, then pass on to generation.
enableGuardrails = True

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
    You are an expert SCU academic advisor. Your job is to give direct, concise, accurate, and complete answers using only the provided university documents. 
    Always extract the relevant information and present it clearly — do not defer to the user to check sources themselves.

    === Answering Rules ===
    1. **General**
    - Break down the user’s question before answering.
    - Search all provided documents for every relevant detail.
    - If no information is found or no courses meet the criteria, state that explicitly.
    - Keep formatting plain text. No bold, no size changes.

    2. **Course/Requirement Questions**
    - If multiple requirements are given (e.g., "Arts AND Ethics"), return only courses that meet *all* requirements (the intersection).
    - Include *every* valid course, with:
        • All cross-listed equivalents (e.g., "PHIL 23 Ethics & Gender (cross-listed with WGST 58)").
        • All official variations and suffixes (A/B/H/HN/C etc.).
    - Preserve exact course codes and titles; do not invent or omit.
    - If cross-listings exist, show them explicitly in parentheses.
    - If special notes apply (e.g., "Business students may satisfy with MGMT 6 or PHIL 26"), include them.
    - Use a bulleted list, one course per line.
    - End with a "Coverage check" line summarizing what cross-listings and variants were included.

    3. **Policy Questions**
    - Identify the specific policy and summarize its rules, conditions, and exceptions.
    - If the user provides context (e.g., "as a transfer student"), apply rules specifically to that case.

    4. **Final Answer**
    - Always be direct, comprehensive, and self-contained.
    - Do not skip or abbreviate course lists.
    - Do not guess: if unsure about cross-listing, write "(cross-listing not found)".
    """

    try:
        response = client.responses.create(
            model=gpt_model,
            input=query,
            temperature=0.0,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": 30,
            }],
            include=["file_search_call.results"],
            instructions=instructions,
        )
        
        # Extract the message text from the response
        message_text = next((item.content[0].text for item in response.output if item.type == 'message'), None) # type: ignore

        if message_text:
            return jsonify({"response": message_text})
        else:
            return jsonify({"response": "Error in generating response."}), 500

    except Exception as e:
        # Handle potential API errors
        return jsonify({"error": str(e)}), 500

# Define the main endpoint for processing user queries
@app.route('/bot', methods=['POST'])
def get_response():
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

# Define a health check endpoint, default
@app.route('/')
def check():
    """
    A simple endpoint to confirm that the Flask application is running.
    """
    return jsonify({"status": "ok", "message": "Provost Bot API is running."})


# Note: For local development, you might add the following lines.
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3001, debug=True)


