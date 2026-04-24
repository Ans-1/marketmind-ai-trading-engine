# core/agents/__init__.py
import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Shared Groq client - used by all agents
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,  # Low temp = consistent, structured JSON outputs
    max_tokens=500,
)

def safe_parse_json(content: str) -> dict:
    """Parse LLM JSON response, extracting the JSON block if hidden in text."""
    try:
        # Hunt down the JSON brackets, ignoring any chatty text before or after
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_string = content[start_idx:end_idx+1]
            return json.loads(json_string)
            
        # Fallback if no brackets are found
        return json.loads(content)
        
    except json.JSONDecodeError:
        return {
            "signal": "NEUTRAL", 
            "confidence": 0.0, 
            # Adding a snippet of the raw text so we can see what it actually said next time!
            "summary": f"Failed to parse LLM response. Raw: {content[:50]}..."
        }