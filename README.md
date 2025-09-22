How to run:

1. Open command prompt at the working repository root. In command prompt, run
- pip install -r requirements.txt
- set OPENAI_API_KEY=*your open api key*
- uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

2. Open http://127.0.0.1:8000/docs to see an interactive API page to manually test the functionality

3. To enable debugging use DEBUG for normal logging use INFO:
- Command Prompt: set log_level=INFO
- Powershell: $env:log_level = 'INFO'
- macOS/Linux: export log_level=INFO