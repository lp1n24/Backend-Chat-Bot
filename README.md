Project Overview
This project is a backend question-answering chatbot. Here is the list of waht this chatbot can do:
- Read and store information from PDFs.
- Answer user questions by looking inside the stored PDFs.
- Ask for clarification if the question is not clear.
- Ability to search the web and use the search results if the answer is not found.
- It can do web search on user demands.
- Remember past conversation in the same session.
- It can answer follow-up questions as it keeps track of past conversations (to some extend).

The systems uses LangGraph to connect multiple agents (router, PDF agent, clarify agent, web agent).
It also uses FastAPI to provide an HTTP API that we can call to test the system.

Files included:
- src/ : All python code
- data/papers/papers/ : PDF files store location
- data/processed : CSV file created from ingest.py
- Dockerfile : container setup
- docker-compose.yml : To run the system
- requirements.txt : to install python dependencies

Here is a simple architecture diagram that shows how the system works.

                       User Question
                             |
                             v
                          FastAPI
                             |
                             v
                     Router (router.py)
         - TF-IDF match with PDF text from pages.csv
         - Rule checks (vague, out of scope, direct)
                             |
                             v
                        Action chosen
         + - - - - - - - - - + - - - - - - - - - +
         |                   |                   |
         v                   v                   v
      PDF Agent         Clarify Agent         Web Agent
    - search pages.csv   - ask user        - use snippets from
    for answer           questions         web search for answer
         |                   |                   |
         + - - - - - > Session Memory <- - - - - +
                             |
                             v
                        Final Answer

Agents:
- Router: Decides where to send the question (PDFs, clarification, or web).
- PDF Agent: Uses TF-IDF retriever to find best text parts from ingested PDFs (saved as pages.csv) and ask the LLM to answer
- Clarify Agent: Uses the LLM to ask user for more details if the question is too vague.
- Web Agent: Uses DuckDuckGo search to find answers online when PDFs are not enough.


How to run locally with docker-compose:
1. Make sure you have Docker and docker-compose installed.
2. Clone this repository and open a terminal in the project folder.
   - git clone https://github.com/lp1n24/backend-chat-bot.git
   - cd Backend-Chat-Bot
3. Edit docker-compose.yml file to enable LLM using your OpenAI API key.
   - In your opened terminal, use notepad for simple editing:
     - notepad docker-compose.yml
   - Inside the docker-compose.yml file you will see OPENAI_API_KEY: "your_api_key_here".
   - Insert your own OpenAI API key in place of "your_api_key_here" (e.g., OPENAI_API_KEY: "sk-...").
4. Inside terminal type these command
   - For Windows: 
     - docker compose build
     - docker compose up
   - For Linux/Mac:
     - docker-compose build
     - docker-compose up
5. After the container starts, go to web browser and type:
   - http://127.0.0.1:8000/docs
   This will navigate you to where the API runs on.

How to use our "Backend-Chat-Bot":
1. Make sure you are inside the web page (http://127.0.0.1:8000/docs)
2. To ask questions:
   - Click and expand the "POST/ask" inside default list (it should be with Get /status and POST /clear)
   - Click "Try it out" then you will see a huge text block that contains "session_id" and "question" strings.
   - Change "string" in "session_id" to something like s1 or s2 (your unique session_id value).
   - Change "string" in "question" to your questions. E.g., "What is OpenAI?"
   - Click "Execute" button to let your question go through the system.
3. You should see the answer and all the necessary details inside Response body text block.
   - Note that if you run this chatbot without OpenAI API key, the response will be less informative as it doesn't use LLM to answer the question. The string "mode" inside Response body tells you which mode are you talking with (llm or local).
4. To clear session history:
   - Click and expand the "POST/clear" inside the same list as "POST/ask"
   - Click "Try it out" then you will see a huge text block that contains session_id string.
   - Change "string" in "session_id" to the session you want to clear.
   - Click "Execute" button to clear that session's history.
5. You can also see the system's action on the terminal. E.g., "2025-09-22 22:20:24,250 INFO api: ASK start session=s1". You can set to Debug level for debugging with "set LOG_LEVEL=DEBUG".

Design Trade-offs:
1. TF-IDF retriever:
   - We used TF-IDF instead of embeddings for retrieval.
   - It is fast, lightweight, and easy to set up.
   - It is suitable in academic context.
   - The trade-off is lower accuracy for semantic queries since it only matches keywords but not the actual meaning.
2. In-memory memory store:
   - We used a python dictionary for session memory.
   - This ensure simplicity of the prototype (can easily run in Docker without setting up a database).
   - The trade-off is that the memory resets when the container restarts, so it's not production-ready.
3. Router with rules:
   - The Router agent uses simple logic to decide between actions (PDF search, clarify, web search).
   - It is easy to understand and transparent to user.
   - The trade-off is that it is less flexible than using policy-based like reinforcement learning or advanced planning.
   - It can also occasionally misrouting for tricky queries (especially when we use keywords based not semantic based).
4. Minimal logging:
   - We used lighweight logging for normal observation and debugging (API calls, graph decisions, LLM responses).
   - This gives enough information without overloading terminal with bunch of text.
   - The trade-off is that it doesn't give enough information when compared to structured logs or tracing in production level.

Future Work:
1. Use embeddings retriever:
   - Use something like OpenAI embeddings or HuggingFace models would improve accuracy, especially in semantic queries.
   - It increases overall capability of the chatbot because semantic similarity works better than keyword matching (can handle long and complex queries).
2. Use reinforcement learning for router:
   - We can train the RL agent with feedback signals to reduce the chance of misrouting.
   - This allow system to adapt better to tricky queries and feel more autonomous. Simply just make the system feels smarter and more capable.
3. Use the actual database for memory:
   - With the actual database, we can now store memory without having them gone every time we restarted the container.
   - This makes the system more reliable in real production with many users.
4. Implement confidence scoring:
   - Add a confidence score for each answer.
   - This is a better way to let the agent automatically ask the user for clarification.
   - This can also serve as a good performance matrix for developer to fine tune the system better.
5. Frontend web UI:
   - Create a simple custom UI for chatbot.
   - This helps non-technical users to navigate through whole system more smoothly.
6. Evaluation system:
   - Use automatic evaluation system that check for quantitative aspect like accuracy, relevance, latency, etc.
   - This let developer to measure the overall performance of the system after every updates.
   - it prevents developer to push worse updates


