import os
import json
import pandas as pd
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ScriptAnalyzer:
    def __init__(self, google_api_key: str = None):
        """Initialize the ScriptAnalyzer and related components."""
        if google_api_key:
            genai.configure(api_key=google_api_key)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    
    def load_script(self, script_path: str) -> str:
        """Load script text from a file."""
        with open(script_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(self, text: str) -> list:
        """Split text into chunks for processing."""
        return self.text_splitter.create_documents([text])
    
    def create_vector_store(self, documents: list):
        """Create a vector store from document chunks."""
        return FAISS.from_documents(documents, self.embeddings)
    
    def get_relevant_context(self, vector_store, query: str, k: int = 5) -> str:
        """Retrieve relevant context based on a query."""
        docs = vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    
    def analyze_character(self, vector_store, character_name: str) -> dict:
        """
        Analyze a specific character in detail.
        Returns a JSON profile with detailed information about the character.
        """
        query = f"Tell me everything about the character named {character_name}"
        context = self.get_relevant_context(vector_store, query, k=8)
        
        system_prompt = f"""
        You are a professional script analyst. You need to extract detailed information about the character '{character_name}' from the provided script context.
        Only include information that's explicitly stated or strongly implied in the text.
        """
        
        user_prompt = f"""
        Analyze this character from the script context:
        
        Character: {character_name}
        
        Context:
        {context}
        
        Generate a detailed JSON profile with the following structure:
        {{
          "character": {{
            "about": {{
              "role": "",
              "personalInformation": {{
                "firstName": "",
                "lastName": "",
                "nickname": "",
                "gender": "",
                "species": "",
                "ethnicity": "",
                "age": "",
                "birthday": "",
                "birthplace": "",
                "residence": ""
              }},
              "appearance": {{
                "physicalDescription": "",
                "distinctiveFeatures": "",
                "clothingStyle": ""
              }},
              "background": {{
                "occupation": "",
                "education": "",
                "family": ""
              }},
              "personality": {{
                "temperament": "",
                "personalityTraits": [],
                "likes": [],
                "dislikes": [],
                "habits": []
              }}
            }},
            "summary": {{
              "inDepth": "",
              "atAGlance": "",
              "flashcard": ""
            }},
            "deepDive": {{
              "backgroundAndOrigin": {{
                "origin": "",
                "familyHistory": "",
                "keyLifeEvents": []
              }},
              "personalityTraits": {{
                "strengths": [],
                "weaknesses": [],
                "fears": [],
                "motivations": [],
                "values": []
              }},
              "goalsConflicts": {{
                "shortTermGoals": [],
                "longTermGoals": [],
                "internalConflicts": [],
                "externalConflicts": []
              }},
              "skillsAndAbilities": {{
                "training": "",
                "specializedSkills": []
              }},
              "hobbiesAndGrowth": {{
                "growth": "",
                "turningPoints": [],
                "hobbies": []
              }},
              "speechAndRepresentation": {{
                "dialogueStyle": "",
                "speechPatterns": "",
                "quotes": [],
                "symbols": []
              }}
            }}
          }}
        }}
        
        Only include information that is directly supported by the text. Leave fields empty if information isn't available.
        """
        
        # Initialize the Gemini model (using Gemini 1.5 Flash)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            {"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]}
        ])
        
        try:
            content_text = response.text
            start_idx = content_text.find('{')
            end_idx = content_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = content_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"raw_response": content_text}
        except json.JSONDecodeError:
            return {"raw_response": response.text}
    
    def analyze_location(self, vector_store, location_name: str) -> dict:
        """
        Analyze a specific location in detail.
        Returns a JSON profile with detailed information about the location.
        """
        query = f"Tell me everything about the location named {location_name}"
        context = self.get_relevant_context(vector_store, query, k=5)
        
        system_prompt = f"""
        You are a professional script analyst. You need to extract detailed information about the location '{location_name}' from the provided script context.
        Only include information that's explicitly stated or strongly implied in the text.
        """
        
        user_prompt = f"""
        Analyze this location from the script context:
        
        Location: {location_name}
        
        Context:
        {context}
        
        Generate a detailed JSON profile with the following structure:
        {{
          "location": {{
            "about": {{
              "basicInformation": {{
                "name": "",
                "type": "",
                "description": "",
                "significance": "",
                "currentState": "",
                "inhabitants": ""
              }},
              "historicalInformation": {{
                "history": "",
                "keyEvents": []
              }}
            }},
            "summary": {{
              "inDepth": "",
              "atAGlance": "",
              "flashcard": ""
            }}
          }}
        }}
        
        Only include information that is directly supported by the text. Leave fields empty if information isn't available.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            {"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]}
        ])
        
        try:
            content_text = response.text
            start_idx = content_text.find('{')
            end_idx = content_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = content_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"raw_response": content_text}
        except json.JSONDecodeError:
            return {"raw_response": response.text}

def process_characters_from_csv(csv_path: str, script_path: str, output_path: str):
    """
    Processes a CSV file of character data (with columns: character, normalized_name, mentions),
    analyzes each character from the film script using the Gemini API, and saves the results as JSON.
    """
    analyzer = ScriptAnalyzer()
    
    # Load script and create a vector store for retrieval
    print("Loading script for character analysis...")
    script_text = analyzer.load_script(script_path)
    chunks = analyzer.chunk_text(script_text)
    vector_store = analyzer.create_vector_store(chunks)
    
    # Read the CSV file with character data
    df = pd.read_csv(csv_path)
    character_results = []
    
    for idx, row in df.iterrows():
        character_name = row["normalized_name"]
        print(f"Analyzing character: {character_name}")
        analysis = analyzer.analyze_character(vector_store, character_name)
        entry = {
            "character": row["character"],
            "normalized_name": character_name,
            "mentions": row["mentions"],
            "analysis": analysis
        }
        character_results.append(entry)
    
    # Save the detailed character analysis results to a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(character_results, f, indent=2)
    print(f"Character analysis saved to {output_path}")

def process_locations_from_csv(csv_path: str, script_path: str, output_path: str):
    """
    Processes a CSV file of location data (with columns: location, normalized_name, mentions),
    analyzes each location from the film script using the Gemini API, and saves the results as JSON.
    """
    analyzer = ScriptAnalyzer()
    
    # Load script and create a vector store for retrieval
    print("Loading script for location analysis...")
    script_text = analyzer.load_script(script_path)
    chunks = analyzer.chunk_text(script_text)
    vector_store = analyzer.create_vector_store(chunks)
    
    # Read the CSV file with location data
    df = pd.read_csv(csv_path)
    location_results = []
    
    for idx, row in df.iterrows():
        location_name = row["normalized_name"]
        print(f"Analyzing location: {location_name}")
        analysis = analyzer.analyze_location(vector_store, location_name)
        entry = {
            "location": row["location"],
            "normalized_name": location_name,
            "mentions": row["mentions"],
            "analysis": analysis
        }
        location_results.append(entry)
    
    # Save the detailed location analysis results to a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(location_results, f, indent=2)
    print(f"Location analysis saved to {output_path}")

if __name__ == "__main__":
    # Define file paths
    CHARACTERS_CSV = "filtered_characters.csv"    
    LOCATIONS_CSV = "filtered_locations.csv"      
    SCRIPT_PATH = "film_script2.txt"           # Film script draft file
    OUTPUT_CHARACTERS_JSON = "character_details.json"
    OUTPUT_LOCATIONS_JSON = "location_details.json"
    
    # Process characters and locations separately
    process_characters_from_csv(CHARACTERS_CSV, SCRIPT_PATH, OUTPUT_CHARACTERS_JSON)
    process_locations_from_csv(LOCATIONS_CSV, SCRIPT_PATH, OUTPUT_LOCATIONS_JSON)
