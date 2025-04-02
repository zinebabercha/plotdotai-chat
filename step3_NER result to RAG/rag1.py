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
            chunk_size=2000,  # Increased chunk size to handle more context
            chunk_overlap=400,  # Increased overlap to maintain context across chunks
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
            model="gpt-4",  # Ensure using GPT-4
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
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

# Update process functions as necessary to incorporate optimizations and handling for the full GPT model.

if __name__ == "__main__":
    # Define file paths
    CHARACTERS_CSV = "filtered_characters.csv"
    SCRIPT_PATH = "film_script2.txt"
    OUTPUT_CHARACTERS_JSON = "character_details2.json"
    
    # Initialize and process characters
    analyzer = ScriptAnalyzer()
    script_text = analyzer.load_script(SCRIPT_PATH)
    chunks = analyzer.chunk_text(script_text)
    vector_store = analyzer.create_vector_store(chunks)
    
    df = pd.read_csv(CHARACTERS_CSV)
    character_names = df["normalized_name"].tolist()
    print(f"Analyzing characters: {', '.join(character_names)}")
    character_analysis = analyzer.analyze_all_characters(vector_store, character_names)
    
    with open(OUTPUT_CHARACTERS_JSON, "w", encoding="utf-8") as f:
        json.dump(character_analysis, f, indent=2)
    print(f"Character analysis saved to {OUTPUT_CHARACTERS_JSON}")