import os
import json
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI 
from tqdm import tqdm
import time
import uuid

# Import Langfuse
from langfuse import Langfuse

import config 
import utils 

class RagProcessor:
    """Handles RAG-based analysis for characters and locations."""

    def __init__(self):
        """Initializes embeddings, text splitter, and OpenAI client."""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables or config.")

        print("Initializing RAG Processor components...")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.RAG_CHUNK_SIZE,
            chunk_overlap=config.RAG_CHUNK_OVERLAP,
            separators=config.RAG_SEPARATORS
        )
        print(f"Using Embedding Model: {config.OPENAI_EMBEDDING_MODEL}")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_EMBEDDING_MODEL
        )
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.vector_store = None
        
        # Initialize Langfuse if keys are available
        if hasattr(config, 'LANGFUSE_PUBLIC_KEY') and hasattr(config, 'LANGFUSE_SECRET_KEY') and config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
            self.langfuse = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=getattr(config, 'LANGFUSE_HOST', "https://cloud.langfuse.com")
            )
            print("Langfuse monitoring initialized.")
        else:
            self.langfuse = None
            print("Langfuse not configured, continuing without monitoring.")
            
        print("RAG Processor initialized.")

    def _build_vector_store(self, script_text: str, character_names: list[str] = None):
        """Builds FAISS vector store with optional character tagging."""
        # Create a trace for this operation
        trace = None
        if self.langfuse:
            trace = self.langfuse.trace(
                name="build_vector_store",
                metadata={"character_count": len(character_names) if character_names else 0}
            )
        
        print("Chunking script text...")
        docs = self.text_splitter.create_documents([script_text])

        if character_names:
            for doc in docs:
                lower_text = doc.page_content.lower()
                matched_chars = [name for name in character_names if name.lower() in lower_text]
                if matched_chars:
                    doc.page_content = f"[Characters: {', '.join(matched_chars)}]\n\n" + doc.page_content
        
        print(f"Created {len(docs)} document chunks.")
        print("Building FAISS vector store...")
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        print("Vector store built successfully.")

    def _get_relevant_context(self, query: str, k: int) -> str:
        """Retrieves relevant context from the vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not built. Call process_script first.")
            
        trace = None
        if self.langfuse:
            trace = self.langfuse.trace(
                name="context_retrieval",
                metadata={"query": query[:100], "k": k}
            )
            
        print(f"Retrieving top {k} relevant chunks for query: '{query[:50]}...'")
        docs = self.vector_store.similarity_search(query, k=k)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
        print(f"Retrieved context length: {len(context)} characters.")
        return context

    def _call_openai_chat(self, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
        """Calls the OpenAI Chat API and returns the message content."""
        print(f"Calling OpenAI Chat API (model: {model}, temp: {temperature})...")
        max_retries = 3
        retry_delay = 5 # seconds
        
        # Create a trace for OpenAI call
        trace = None
        if self.langfuse:
            trace = self.langfuse.trace(
                name="openai_chat_completion",
                metadata={
                    "model": model,
                    "temperature": temperature,
                    "system_prompt_length": len(system_prompt),
                    "user_prompt_length": len(user_prompt)
                }
            )
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature
                )
                end_time = time.time()
                content = response.choices[0].message.content
                
                if self.langfuse:
                    # Log the completion using generation object
                    self.langfuse.generation(
                        name="openai_chat_completion",
                        model=model,
                        prompt=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        completion=content,
                        usage={
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        },
                        metadata={
                            "duration_ms": int((end_time - start_time) * 1000),
                            "temperature": temperature
                        }
                    )
                
                print("Received response from OpenAI.")
                return content
            except openai.RateLimitError:
                print(f"Rate limit exceeded. Retrying chat call in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying chat call in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Failed to call OpenAI API for chat.")
                    return "" 
        return ""

    def _parse_json_response(self, response_text: str) -> list:
        """Parses JSON object from OpenAI response, handling markdown wrapping or formatting issues."""
        if response_text.startswith("```json"):
            response_text = response_text.strip()[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()

        try:
            parsed_json = json.loads(response_text)
            if isinstance(parsed_json, dict):
                return [parsed_json]
            elif isinstance(parsed_json, list):
                return parsed_json
            else:
                return [{"error": "Parsed JSON is not a dict or list", "raw_response": response_text}]
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            return [{"error": "JSONDecodeError", "raw_response": response_text}]

    def get_entity_details(self, script_text: str, entity_list: list[str], entity_type: str, mentions_dict: dict = None) -> list[dict]:
        """
        Gets detailed analysis for a list of characters or locations using RAG with per-entity context retrieval.
        """
        if not entity_list:
            print(f"No {entity_type}s provided for detail extraction.")
            return []
            
        # Create a trace for the entire entity details process
        trace = None
        if self.langfuse:
            trace = self.langfuse.trace(
                name=f"get_{entity_type}_details",
                metadata={
                    "entity_type": entity_type,
                    "entity_count": len(entity_list),
                    "script_length": len(script_text)
                }
            )

        if self.vector_store is None:
            self._build_vector_store(script_text, entity_list)

        print(f"Collecting context for {len(entity_list)} {entity_type}(s)...")
        context_dict = {}
            
        for entity in tqdm(entity_list, desc="Gathering context"):
            if entity_type == 'character':
                query = f"Detailed information about the character {entity}: their personality, background, appearance, memorable dialogues, and actions"
                k = config.RAG_CONTEXT_K_DETAILS_PER_CHARACTER
            elif entity_type == 'location':
                query = f"Detailed information about the location {entity}: its appearance, significance, history, and events that occur there"
                k = config.RAG_CONTEXT_K_LOCATIONS
            else:
                raise ValueError("entity_type must be 'character' or 'location'")

            context = self._get_relevant_context(query, k=k)
            context_dict[entity] = context

        # Construct a batch prompt
        print("Constructing batch prompt...")
        prompt_parts = []
        for idx, (entity, context) in enumerate(context_dict.items(), start=1):
            single_prompt = config.get_rag_single_character_details_user_prompt(entity, context)
            prompt_parts.append(f"### Character {idx} of {len(context_dict)} ###\n{single_prompt}")

        joined_prompt = "\n\n".join(prompt_parts)

        system_prompt = config.get_rag_single_character_details_system_prompt()
        user_prompt = f"""You will be analyzing **{len(entity_list)} characters** from a script. 
        Each one has been extracted with their own context. For each, you must return a JSON object 
        with the detailed analysis as specified.

        Your output MUST be a JSON **array** of {len(entity_list)} objects. Each object corresponds to one character.

        Follow this format exactly. Return no extra commentary — only valid JSON.

    {joined_prompt}
        """

        # Send one OpenAI call
        model = config.OPENAI_CHAT_MODEL_DETAILS
        temperature = config.RAG_DETAILS_TEMPERATURE
        response_text = self._call_openai_chat(system_prompt, user_prompt, model, temperature)

        if not response_text:
            print("Failed to get batch details from OpenAI.")
            return [{"normalized_name": name, "error": "OpenAI call failed"} for name in entity_list]

        # Parse the JSON array
        try:
            response_data = self._parse_json_response(response_text)
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            return [{"normalized_name": name, "error": "Parse error"} for name in entity_list]

        # Enrich each item with metadata
        enriched_results = []
        for item in response_data:
            name = item.get("character_name", None)
            if not name:
                item["error"] = "Missing character_name"
                enriched_results.append(item)
                continue

            item["character"] = name
            item["normalized_name"] = name
            # Normalize name before lookup
            normalized = name.lower()
            if mentions_dict:
                item["mentions"] = mentions_dict.get(name, mentions_dict.get(normalized, 0))
            else:
                item["mentions"] = 0

            enriched_results.append(item)

        print(f"Completed analysis for {len(enriched_results)} {entity_type}(s).")
        return enriched_results