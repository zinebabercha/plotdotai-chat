import os
import json
import pandas as pd
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables and configure OpenAI
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ScriptAnalyzer:
    def __init__(self):
        """Initialize the ScriptAnalyzer and related components."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
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
    
    def analyze_all_characters(self, vector_store, character_names: list) -> list:
        """Analyze all characters in one prompt."""
        query = "Tell me everything about the following characters: " + ", ".join(character_names)
        context = self.get_relevant_context(vector_store, query, k=8)
        
        system_prompt = """
        You are a professional script analyst. You need to extract detailed information about multiple characters from the provided script context.
        Only include information that's explicitly stated or strongly implied in the text.
        """
        
        user_prompt = f"""
        Analyze these characters from the script context:
        
        Characters: {', '.join(character_names)}
        
        Context:
        {context}
        
        Generate a list of detailed JSON profiles, one for each character, with the following structure for each:
        {{
          "character": {{
            "about": {{
              "role": (one of : "protagonist", "antagonist", "deuteragonist", "tritagonist", "mentor", "sidekick", "foil", "confidant", "villain", "anti-hero", "anti-villain" ),
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
        
        Return a JSON array containing one object per character. Only include information that is directly supported by the text. 
        Leave fields empty if information isn't available.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        content_text = response.choices[0].message.content
        try:
            start_idx = content_text.find('[')
            end_idx = content_text.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = content_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return [{"raw_response": content_text}]
        except json.JSONDecodeError:
            return [{"raw_response": content_text}]
    
    def analyze_all_locations(self, vector_store, location_names: list) -> list:
        """Analyze all locations in one prompt."""
        query = "Tell me everything about the following locations: " + ", ".join(location_names)
        context = self.get_relevant_context(vector_store, query, k=5)
        
        system_prompt = """
        You are a professional script analyst. You need to extract detailed information about multiple locations from the provided script context.
        Only include information that's explicitly stated or strongly implied in the text.
        """
        
        user_prompt = f"""
        Analyze these locations from the script context:
        
        Locations: {', '.join(location_names)}
        
        Context:
        {context}
        
        Generate a list of detailed JSON profiles, one for each location, with the following structure for each:
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
        
        Return a JSON array containing one object per location. Only include information that is directly supported by the text. 
        Leave fields empty if information isn't available.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        content_text = response.choices[0].message.content
        try:
            start_idx = content_text.find('[')
            end_idx = content_text.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = content_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return [{"raw_response": content_text}]
        except json.JSONDecodeError:
            return [{"raw_response": content_text}]

def process_characters_from_csv(csv_path: str, script_path: str, output_path: str):
    """
    Processes a CSV file of character data and analyzes all characters in one prompt.
    """
    analyzer = ScriptAnalyzer()
    
    # Load script and create a vector store for retrieval
    print("Loading script for character analysis...")
    script_text = analyzer.load_script(script_path)
    chunks = analyzer.chunk_text(script_text)
    vector_store = analyzer.create_vector_store(chunks)
    
    # Read the CSV file with character data
    df = pd.read_csv(csv_path)
    character_names = df["normalized_name"].tolist()
    character_results = []
    
    print(f"Analyzing all characters: {', '.join(character_names)}")
    analyses = analyzer.analyze_all_characters(vector_store, character_names)
    
    # Match analyses with original CSV data
    for idx, row in df.iterrows():
        character_name = row["normalized_name"]
        analysis = next((item for item in analyses if item["character"]["about"]["personalInformation"].get("firstName", "").lower() == character_name.lower() or 
                        item["character"]["about"]["personalInformation"].get("lastName", "").lower() == character_name.lower()), 
                        analyses[idx] if idx < len(analyses) else {"raw_response": "No analysis found"})
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
    Processes a CSV file of location data and analyzes all locations in one prompt.
    """
    analyzer = ScriptAnalyzer()
    
    # Load script and create a vector store for retrieval
    print("Loading script for location analysis...")
    script_text = analyzer.load_script(script_path)
    chunks = analyzer.chunk_text(script_text)
    vector_store = analyzer.create_vector_store(chunks)
    
    # Read the CSV file with location data
    df = pd.read_csv(csv_path)
    location_names = df["normalized_name"].tolist()
    location_results = []
    
    print(f"Analyzing all locations: {', '.join(location_names)}")
    analyses = analyzer.analyze_all_locations(vector_store, location_names)
    
    # Match analyses with original CSV data
    for idx, row in df.iterrows():
        location_name = row["normalized_name"]
        analysis = next((item for item in analyses if item["location"]["about"]["basicInformation"]["name"].lower() == location_name.lower()), 
                        analyses[idx] if idx < len(analyses) else {"raw_response": "No analysis found"})
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
    SCRIPT_PATH = "script.txt"
    OUTPUT_CHARACTERS_JSON = "character_details.json"
    OUTPUT_LOCATIONS_JSON = "location_details.json"
    
    # Process characters and locations separately
    process_characters_from_csv(CHARACTERS_CSV, SCRIPT_PATH, OUTPUT_CHARACTERS_JSON)
    process_locations_from_csv(LOCATIONS_CSV, SCRIPT_PATH, OUTPUT_LOCATIONS_JSON)