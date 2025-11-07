from openai import OpenAI
from tqdm import tqdm
import concurrent
import os
from dotenv import dotenv_values
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"

if not env_path.exists():
    raise FileNotFoundError(f"Unable to locate .env file at {env_path}")

env_vars = dotenv_values(env_path)
api_key = env_vars.get("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError(
        f"Missing OPENAI_API_KEY entry in {env_path}. Please add OPENAI_API_KEY=your_api_key to the .env file."
    )

client = OpenAI(api_key=api_key)

dir_pdfs = 'PAGES/cleaned'
pdf_files = [os.path.join(dir_pdfs, f) for f in os.listdir(dir_pdfs)]
print(pdf_files)

def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

def upload_single_pdf(file_path: str, vector_store_id: str):
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(file=open(file_path, 'rb'), purpose="assistants")
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}

def upload_pdf_files_to_vector_store(vector_store_id: str):
    pdf_files = [os.path.join(dir_pdfs, f) for f in os.listdir(dir_pdfs)]
    stats = {"total_files": len(pdf_files), "successful_uploads": 0, "failed_uploads": 0, "errors": []}

    print(f"{len(pdf_files)} PDF files to process. Uploading in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(upload_single_pdf, file_path, vector_store_id): file_path for file_path in pdf_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_files)):
            result = future.result()
            if result["status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)

    return stats

store_name = "ECC_Knowledge_Base"
vector_store_details = create_vector_store(store_name)

# upload all files to the vector-store
#I've converted the pdf to txt to improve embed results
upload_pdf_files_to_vector_store(vector_store_details["id"])