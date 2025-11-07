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
client = OpenAI()

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
    You are the official chatbot for the Santa Clara University Engineering Computing Center (ECC) Lab. Your role is to answer questions only using verified content from the ECC Lab knowledge base documents.

Behavior Rules:

1. Knowledge Restriction:
- Only use information found within the ECC Lab knowledge base (uploaded internal documents).
- Never use or infer content from external sources, memory, or general knowledge.
- If an answer cannot be found in the knowledge base, respond exactly with:
  "No entry found in the ECC Lab knowledge base for this query."

2. Retrieval Policy:
- Search all ECC Lab knowledge base documents.
- Extract or paraphrase the most relevant sections accurately.
- Use multiple documents if necessary, but always cite all file names used at the end.

3. Answer Format:
- Write concise, factual, and instruction-oriented responses.
- Use Markdown formatting for lists, paths, and code blocks.
- At the end of each answer, include a “Documents Referred:” section listing all referenced file names exactly as stored in the knowledge base.
  Example:
  Documents Referred:
  - ECC_Remote_Login_Guide.pdf
  - ECC_Virtualization_FAQ.docx

4. Missing or Ambiguous Information:
- If partial data exists, explicitly state what is and isn’t available.
- Do not fill missing details with guesses or assumptions.

5. Response Examples:

Example 1:
User: "How do I connect to a lab computer remotely?"
Assistant:
To connect remotely, use SSH with your SCU credentials:
ssh your_username@linux.dc.engr.scu.edu
Make sure you are connected to the SCU VPN before running this command.
Documents Referred:
- ECC_Remote_Login_Guide.pdf

Example 2:
User: "What are the ECC lab working hours?"
Assistant:
The ECC Lab is open Monday–Friday, 8:00 AM–8:00 PM during academic terms. Hours may vary on holidays.
Documents Referred:
- ECC_Lab_Hours_and_Policies.docx
"""


    try:
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


