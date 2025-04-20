import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import argparse

# Load environment variables
load_dotenv()

class ScriptAnalyzer:
    def __init__(self, openai_api_key: str = None):
        """Initialize the ScriptAnalyzer and related components."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.api_key
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

    def extract_plot_with_rag(self, script_path: str, output_json: str = "plot_output.json"):
        """Extract the plot in a structured three-act format using RAG and OpenAI."""
        script_text = self.load_script(script_path)
        script_chunks = self.chunk_text(script_text)
        vector_store = self.create_vector_store(script_chunks)

        plot_query = "Summarize the film script using a Three Act Structure: setup, confrontation, resolution"
        relevant_context = self.get_relevant_context(vector_store, plot_query, k=10)

        prompt = (
            "You are an expert story analyst. Your job is to analyze a film script and extract its full plot "
            "using the Three-Act Structure. The format must be strict JSON with this structure:\n\n"
            "{"
            "\"Genre\": \"\", \"Subgenre\": \"\", \"Theme\": \"\", \"Style\": \"\", \"Structure\": \"Three Acts Structure\", "
            "\"Acts\": {"
            "  \"Act I\": {\"Setup\": {\"Opening Scene\": \"\", \"Inciting Incident\": \"\", \"Major Characters Introduced\": \"\", "
            "\"First Plot Point\": \"\", \"Sequences\": [{\"Title\": \"\", \"Description\": \"\", "
            "\"Scenes\": [{\"Title\": \"\", \"Description\": \"\", \"Beats\": [{\"Title\": \"\", \"Description\": \"\"}]}]}]}}, "
            "  \"Act II\": {\"Confrontation\": {\"Rising Conflict\": \"\", \"Increasing Tension\": \"\", \"Midpoint\": \"\", "
            "\"Second Plot Point\": \"\", \"Sequences\": [{\"Title\": \"\", \"Description\": \"\", "
            "\"Scenes\": [{\"Title\": \"\", \"Description\": \"\", \"Beats\": [{\"Title\": \"\", \"Description\": \"\"}]}]}]}}, "
            "  \"Act III\": {\"Resolution\": {\"Climax\": \"\", \"Falling Action\": \"\", \"Denouement\": \"\", "
            "\"Sequences\": [{\"Title\": \"\", \"Description\": \"\", "
            "\"Scenes\": [{\"Title\": \"\", \"Description\": \"\", \"Beats\": [{\"Title\": \"\", \"Description\": \"\"}]}]}]}}"
            "}}"
            "Only use information you can confidently retrieve from the provided script context.\n"
            "If information is unavailable, leave the field empty.\n\n"
            "Here is the relevant context from the script:\n\n"
            f"{relevant_context}"
        )

        print("Sending plot extraction prompt to OpenAI API...")
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in film script analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=3000
        )
        response_text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")

        with open(output_json, "w", encoding="utf-8") as f:
            f.write(response_text)
        print(f"Plot extracted and saved to {output_json}")

        return json.loads(response_text)


def main():
    parser = argparse.ArgumentParser(description="Script Analysis: Plot Extraction with RAG")
    parser.add_argument("--script", type=str, required=True, help="Path to the .txt script file")
    parser.add_argument("--output", type=str, default="plot_output.json", help="Output JSON filename")
    args = parser.parse_args()

    analyzer = ScriptAnalyzer()
    analyzer.extract_plot_with_rag(script_path=args.script, output_json=args.output)

if __name__ == "__main__":
    main()
