How to run:

1. Open command prompt at the working repository root. In command prompt, run
- pip install -r requirements.txt
- set OPENAI_API_KEY=*your open api key*
- uvicorn src.api:app --reload

2. Open http://127.0.0.1:8000/docs to see an interactive API page to manually test the functionality