import time
from typing import List, Dict, Optional

class MemoryAgent:
    def __init__(self, max_conversations: int = 10, max_conversation_age: int = 3600):
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.current_conversation_id: Optional[str] = None
        self.max_conversations = max_conversations
        self.max_conversation_age = max_conversation_age  # in seconds

    def start_new_conversation(self) -> str:
        conversation_id = str(time.time())
        self.conversations[conversation_id] = []
        self.current_conversation_id = conversation_id
        self._cleanup_old_conversations()
        return conversation_id

    def add_exchange(self, question: str, response: str) -> None:
        if self.current_conversation_id is None:
            self.start_new_conversation()
        
        self.conversations[self.current_conversation_id].append({
            "question": question,
            "response": response,
            "timestamp": time.time()
        })

    def get_current_conversation(self) -> List[Dict[str, str]]:
        if self.current_conversation_id is None:
            return []
        return self.conversations[self.current_conversation_id]

    def get_all_conversations(self) -> Dict[str, List[Dict[str, str]]]:
        return self.conversations

    def clear_conversations(self) -> None:
        self.conversations.clear()
        self.current_conversation_id = None

    def _cleanup_old_conversations(self) -> None:
        current_time = time.time()
        conversations_to_remove = []

        for conv_id, conversation in self.conversations.items():
            if not conversation:  # Skip empty conversations
                continue
            last_exchange_time = conversation[-1]["timestamp"]
            if current_time - last_exchange_time > self.max_conversation_age:
                conversations_to_remove.append(conv_id)

        for conv_id in conversations_to_remove:
            del self.conversations[conv_id]

        # If we still have too many conversations, remove the oldest ones
        while len(self.conversations) > self.max_conversations:
            oldest_conv_id = min(self.conversations.keys(), key=lambda k: self.conversations[k][-1]["timestamp"])
            del self.conversations[oldest_conv_id]