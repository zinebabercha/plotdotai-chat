

# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Model Names ---
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL_DETAILS = "gpt-4o" # Model for RAG Details


# --- RAG Configuration ---
RAG_VECTOR_STORE = "faiss" # Options: "faiss", "pinecone", "weaviate"
RAG_CHUNK_SIZE = 1500
RAG_CHUNK_OVERLAP = 200
RAG_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
RAG_CONTEXT_K_DETAILS = 8
RAG_CONTEXT_K_LOCATIONS = 5
RAG_DETAILS_TEMPERATURE = 0.2
OPENAI_MAX_TOKENS_DETAILS = 2000 # Max tokens for details


# Per-character context size for character details extraction
# This determines how many chunks to retrieve for each individual character
# Smaller than the batch context size since it's focused on one character
RAG_CONTEXT_K_DETAILS_PER_CHARACTER = 12


# --- Langfuse Configuration ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- OpenAI API Retry Configuration ---
OPENAI_MAX_RETRIES = 3
OPENAI_RETRY_DELAY = 5 # seconds

# --- Filtering Configuration ---
FILTERING_TEMPERATURE = 0.1 # Temperature for the filtering LLM call





def get_rag_single_character_details_system_prompt():
    return """You are an expert script analyst specializing in character analysis.
Your task is to analyze a specific character from a script based on provided context.
You will create a detailed JSON object with information about the character's personality, background, appearance, and role in the story.
Focus only on the single character mentioned in the user's prompt.
Base your analysis entirely on evidence from the provided script context.
Do not invent details that aren't supported by the text."""

def get_rag_single_character_details_user_prompt(character_name, context):
    return f"""I need a detailed analysis of the character "{character_name}" from the script. Even if the character is not a main character, try to extract **as much as possible** from the given context. Include even minor actions or dialogue snippets.

Here are relevant excerpts from the script involving this character:

{context}

Based only on the context above, create a detailed JSON object (not an array) about {character_name} with the following structure:
INSTRUCTIONS:
- `"role"` should be extracted directly from the script (e.g., "Artist", "Leader").
- `"role_inferred"` should be inferred from narrative context. Use one of:
  ["protagonist", "antagonist", "deuteragonist", "tritagonist", "mentor", "sidekick", "foil", "confidant", "villain", "anti-hero", "anti-villain", "other"] — or leave empty if unsure.
- `"age"`: Extract if stated or inferred (e.g., "in my twenties" → "20s", "sixty-five years" → "65"). Use "Unknown" if uncertain.

{{
  "character_name": "{character_name}",
  "analysis": {{
    "about": {{
      "role": "", 
      "role_inferred": "", 

      "personalInformation": {{
        "firstName": "", "lastName": "", "nickname": "", "gender": "", "species": "",
        "ethnicity": "", "age": "", "birthday": "", "birthplace": "", "residence": ""
      }},
      "appearance": {{
        "physicalDescription": "", "distinctiveFeatures": "", "clothingStyle": ""
      }},
      "background": {{
        "occupation": "", "education": "", "family": ""
      }},
      "personality": {{
        "temperament": "", "personalityTraits": [], "likes": [], "dislikes": [], "habits": []
      }}
    }},
    "summary": {{
      "inDepth": "", "atAGlance": "", "flashcard": ""
    }},
    "deepDive": {{
      "backgroundAndOrigin": {{
        "origin": "", "familyHistory": "", "keyLifeEvents": []
      }},
      "personalityTraits": {{
        "strengths": [], "weaknesses": [], "fears": [], "motivations": [], "values": []
      }},
      "goalsConflicts": {{
        "shortTermGoals": [], "longTermGoals": [], "internalConflicts": [], "externalConflicts": []
      }},
      "skillsAndAbilities": {{
        "training": "", "specializedSkills": []
      }},
      "hobbiesAndGrowth": {{
        "growth": "", "turningPoints": [], "hobbies": []
      }},
      "speechAndRepresentation": {{
        "dialogueStyle": "", "speechPatterns": "", "quotes": [], "symbols": []
      }}
    }}
  }}
}}

Only include information that's supported by the provided context. If information about certain aspects isn't available, use empty strings, empty arrays, or "Unknown" as appropriate. Return only the JSON object, no explanations or text."""

def get_rag_single_location_details_system_prompt():
    pass

def get_rag_single_location_details_user_prompt(location_name, context):
    pass