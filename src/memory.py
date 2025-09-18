from collections import defaultdict, deque

class SessionMemory:

    # Deque is used to automatically drop unrelevant turns (count both user and agent response)
    def __init__(self, max_turns=10):
        self.history = defaultdict(lambda: deque(maxlen=max_turns))
    
    # Add a new user question to the history
    def add_user(self, session_id, question):
        self.history[session_id].append(("user", question))
    
    # Add a new agent answer to the history
    def add_agent(self, session_id, answer):
        self.history[session_id].append(("agent", answer))
    
    # Return history as a list of (role, text) because it is easier to JSON serialize than deque
    def get(self, session_id):
        return list(self.history.get(session_id, []))
    
    # Completely clear the history for a given session
    def clear(self, session_id):
        if session_id in self.history:
            del self.history[session_id]