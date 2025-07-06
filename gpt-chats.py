import json
from datetime import datetime
import uuid

with open("conversations.json", "r") as f:
    conversations = json.load(f)

combined_chats = []

for conv in conversations:
    chat_id = str(uuid.uuid4())
    title = conv.get("title", "No Title")
    timestamp = conv.get('create_time')
    conversation_text = []

    for value in conv.get("mapping", {}).values():
        msg = value.get('message')
        if msg:
            parts = msg.get('content', {}).get('parts', [])
            cleaned_parts = []
            for part in parts:
                if isinstance(part, dict):
                    cleaned_parts.append(json.dumps(part)) 
                elif isinstance(part, str):
                    cleaned_parts.append(part)
            if cleaned_parts:
                conversation_text.append(" ".join(cleaned_parts).replace('*', '').replace('\n', ''))

    combined_chats.append({
        "id": chat_id,
        "title": title,
        "conversation": "\n".join(conversation_text),
        "timestamp": timestamp
    })

output_path = "combined_chats.json"
with open(output_path, "w") as f:
    json.dump(combined_chats, f, indent=2)
