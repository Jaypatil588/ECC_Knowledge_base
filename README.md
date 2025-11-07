## ECC Knowledge Base Chatbot

This project powers the ECC knowledge base assistant. It provides:

- `BOT_BACKEND/index.py`: API entry point for handling chat requests.
- `STORAGE/addVectorStorage.py`: utility for building the OpenAI vector store from PDF/doc assets under `PAGES/cleaned`.
- `FINETUNE/file-ft-provost.jsonl`: example fine-tuning data used to guide responses.

### Local Setup

1. Duplicate `example.env` to `.env` and fill in required secrets (`OPENAI_API_KEY`, etc.). Keep this file private; it is ignored by git.
2. Install Python dependencies listed in `BOT_BACKEND/requirements.txt`.
3. Run storage scripts or backend services as needed.

### Notes

- Source documents reside in `PAGES/cleaned` and drive model responses.
- Deployments on Vercel should inject secrets via environment variables rather than committed files.