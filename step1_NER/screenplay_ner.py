import os
import re
import argparse
import pandas as pd
from tqdm import tqdm

# Import GLiNER
from gliner import GLiNER

def parse_args():
    parser = argparse.ArgumentParser(description="Extract characters and locations from screenplay using GLiNER")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input screenplay text file")
    parser.add_argument("--output_dir", type=str, default="extracted_entities", help="Directory to save extracted entities")
    parser.add_argument("--model", type=str, default="urchade/gliner_medium-v2.1", help="GLiNER model to use")
    return parser.parse_args()

def read_screenplay(file_path):
    """Read the screenplay text file."""
    print(f"Reading screenplay from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def preprocess_screenplay(text):
    """Preprocess the screenplay by splitting it into manageable chunks."""
    # Split the screenplay into scenes or paragraphs
    scenes = re.split(r'\n\s*\n', text)
    return scenes

def normalize_entity_name(name):
    """Normalize entity names to handle case sensitivity and remove unwanted characters."""
    # Convert to title case for consistency
    normalized = name.strip().title()
    return normalized

def is_pronoun(text):
    """Check if the text is a pronoun that should be excluded."""
    pronouns = ["he", "she", "her", "him", "his", "hers", "they", "them", "their", "theirs","you","we","your","me","i","my","herself","everyone"]
    return text.lower() in pronouns

def is_likely_location(text, locations_list):
    """Check if a text is likely a location rather than a character."""
    location_indicators = ["room", "hall", "building", "street", "road", "avenue", "catacombs", 
                         "city", "town", "village", "house", "apartment", "wall", "castle", 
                         "palace", "temple", "church", "school", "hospital", "office", "store", 
                         "shop", "cafe", "restaurant", "bar", "pub", "club", "park", "garden", 
                         "forest", "woods", "lake", "river", "ocean", "sea", "beach", "mountain", 
                         "hill", "valley", "cave", "tunnel", "bridge", "station", "airport", 
                         "harbor", "port", "market", "square", "plaza", "alley", "basement", 
                         "attic", "roof", "tower", "dungeon", "prison", "jail", "cell", "corridor"]
    
    # Check if text contains common location words
    text_lower = text.lower()
    for indicator in location_indicators:
        if indicator in text_lower:
            return True
    
    # Check if already identified as a location elsewhere
    for location in locations_list:
        if text.lower() == location.lower():
            return True
            
    return False

def extract_entities_with_gliner(model, scenes):
    """Extract characters and locations from each scene using GLiNER."""
    characters = {}  # Changed to dict for case-insensitive tracking
    locations = {}   # Changed to dict for case-insensitive tracking
    character_mentions = {}
    location_mentions = {}
    scene_entities = []
    known_locations = set()  # Track known locations to help with disambiguation
    
    # Define entity types to extract
    entity_types = ["person", "character", "location", "place"]
    
    print("Extracting entities from screenplay using GLiNER...")
    
    # First pass: identify and collect locations to help with disambiguation later
    for scene in tqdm(scenes, desc="First pass - identifying locations"):
        if not scene.strip():
            continue
        
        # Look for scene headings (INT./EXT. LOCATION) to collect location names
        scene_headers = re.findall(r'(INT\.|EXT\.)\s+(.*?)(?:\s+-\s+|\s+|$)', scene)
        for header in scene_headers:
            location = header[1].strip()
            if location:
                # Clean up location names
                location = re.sub(r'\s+-\s+.*$', '', location)  # Remove anything after a dash
                location = location.strip()
                
                if location and len(location) > 1:
                    normalized_location = normalize_entity_name(location)
                    known_locations.add(normalized_location)
    
    # Second pass: extract all entities with improved disambiguation
    for i, scene in enumerate(tqdm(scenes, desc="Second pass - extracting entities")):
        if not scene.strip():
            continue
        
        # Get GLiNER predictions
        entities = model.predict_entities(scene, entity_types, threshold=0.5)
        
        scene_characters = []
        scene_locations = []
        
        for entity in entities:
            entity_text = entity["text"].strip()
            entity_label = entity["label"]
            
            # Skip pronouns
            if is_pronoun(entity_text):
                continue
                
            # Skip entities that are too short
            if len(entity_text) <= 1:
                continue
            
            normalized_entity = normalize_entity_name(entity_text)
            
            # Handle location vs character disambiguation
            if entity_label.lower() in ["location", "place"] or normalized_entity in known_locations or is_likely_location(entity_text, known_locations):
                if normalized_entity not in locations:
                    locations[normalized_entity] = entity_text  # Store original form
                scene_locations.append(normalized_entity)
                location_mentions[normalized_entity] = location_mentions.get(normalized_entity, 0) + 1
            
            elif entity_label.lower() in ["person", "character"] and normalized_entity not in known_locations and not is_likely_location(entity_text, known_locations):
                if normalized_entity not in characters:
                    characters[normalized_entity] = entity_text  # Store original form
                scene_characters.append(normalized_entity)
                character_mentions[normalized_entity] = character_mentions.get(normalized_entity, 0) + 1
        
        # Additionally, look for character names in ALL CAPS (common in screenplays)
        caps_names = re.findall(r'\b([A-Z][A-Z]+)\b', scene)
        for name in caps_names:
            if len(name) <= 1 or name in ["INT", "EXT", "FADE", "CUT", "TO", "DISSOLVE", "OS", "VO", "V.O.", "O.S."]:
                continue
                
            if is_pronoun(name):
                continue
                
            normalized_name = normalize_entity_name(name)
            
            # Skip if already identified as a location
            if normalized_name in known_locations or is_likely_location(normalized_name, known_locations):
                if normalized_name not in locations:
                    locations[normalized_name] = name
                scene_locations.append(normalized_name)
                location_mentions[normalized_name] = location_mentions.get(normalized_name, 0) + 1
            else:
                if normalized_name not in characters:
                    characters[normalized_name] = name
                scene_characters.append(normalized_name)
                character_mentions[normalized_name] = character_mentions.get(normalized_name, 0) + 1
        
        # Also look for scene headings (INT./EXT. LOCATION)
        scene_headers = re.findall(r'(INT\.|EXT\.)\s+(.*?)(?:\s+-\s+|\s+|$)', scene)
        for header in scene_headers:
            location = header[1].strip()
            if location:
                # Clean up location names
                location = re.sub(r'\s+-\s+.*$', '', location)  # Remove anything after a dash
                location = location.strip()
                
                if location and len(location) > 1:
                    normalized_location = normalize_entity_name(location)
                    if normalized_location not in locations:
                        locations[normalized_location] = location
                    scene_locations.append(normalized_location)
                    location_mentions[normalized_location] = location_mentions.get(normalized_location, 0) + 1
        
        # Add scene and its entities
        scene_entities.append({
            "scene_number": i + 1,
            "scene_text": scene[:100] + "..." if len(scene) > 100 else scene,
            "characters": list(set(scene_characters)),
            "locations": list(set(scene_locations))
        })
    
    # Post-processing: ensure entities are in the correct category
    characters_to_move = []
    locations_to_move = []
    
    for char in characters:
        if char in locations or is_likely_location(char, known_locations):
            characters_to_move.append(char)
    
    for loc in locations:
        if loc in characters and not is_likely_location(loc, known_locations):
            locations_to_move.append(loc)
    
    # Move misclassified entities
    for char in characters_to_move:
        if char not in locations:
            locations[char] = characters[char]
            location_mentions[char] = character_mentions.get(char, 0)
        else:
            location_mentions[char] = location_mentions.get(char, 0) + character_mentions.get(char, 0)
        del characters[char]
        del character_mentions[char]
    
    for loc in locations_to_move:
        if loc not in characters:
            characters[loc] = locations[loc]
            character_mentions[loc] = location_mentions.get(loc, 0)
        else:
            character_mentions[loc] = character_mentions.get(loc, 0) + location_mentions.get(loc, 0)
        del locations[loc]
        del location_mentions[loc]
    
    # Update scene entities to reflect corrections
    for scene in scene_entities:
        # Remove characters that are actually locations
        scene["characters"] = [char for char in scene["characters"] if char in characters]
        # Remove locations that are actually characters
        scene["locations"] = [loc for loc in scene["locations"] if loc in locations]
    
    return {
        "characters": characters,
        "locations": locations,
        "character_mentions": character_mentions,
        "location_mentions": location_mentions,
        "scene_entities": scene_entities
    }

def analyze_entities(entities_data):
    """Analyze extracted entities to identify main characters, locations, etc."""
    character_mentions = entities_data["character_mentions"]
    location_mentions = entities_data["location_mentions"]
    
    # Sort characters by number of mentions
    sorted_characters = sorted(character_mentions.items(), key=lambda x: x[1], reverse=True)
    
    # Identify main characters (top 3)
    main_characters = sorted_characters[:3]
    
    # Identify supporting characters (next 10)
    supporting_characters = sorted_characters[3:13]
    
    # Identify minor characters (the rest)
    minor_characters = sorted_characters[13:]
    
    # Sort locations by number of mentions
    sorted_locations = sorted(location_mentions.items(), key=lambda x: x[1], reverse=True)
    
    # Identify main locations (top 3)
    main_locations = sorted_locations[:3]
    
    # Identify secondary locations (the rest)
    secondary_locations = sorted_locations[3:]
    
    return {
        "main_characters": main_characters,
        "supporting_characters": supporting_characters,
        "minor_characters": minor_characters,
        "main_locations": main_locations,
        "secondary_locations": secondary_locations
    }

def analyze_character_relationships(entities_data):
    """Analyze relationships between characters based on co-occurrence in scenes."""
    scene_entities = entities_data["scene_entities"]
    characters = list(entities_data["characters"].keys())
    
    # Create a matrix to track co-occurrences
    co_occurrences = {}
    for char in characters:
        co_occurrences[char] = {}
        for other_char in characters:
            if char != other_char:
                co_occurrences[char][other_char] = 0
    
    # Count co-occurrences in scenes
    for scene in scene_entities:
        scene_chars = scene["characters"]
        for i, char in enumerate(scene_chars):
            for other_char in scene_chars[i+1:]:
                if char != other_char:
                    co_occurrences[char][other_char] = co_occurrences[char][other_char] + 1
                    co_occurrences[other_char][char] = co_occurrences[other_char][char] + 1
    
    # Find the strongest relationships
    relationships = []
    for char in characters:
        for other_char, count in co_occurrences[char].items():
            if count > 0:
                relationships.append((char, other_char, count))
    
    # Sort relationships by co-occurrence count
    relationships.sort(key=lambda x: x[2], reverse=True)
    
    return relationships[:50]  # Return top 50 relationships

def save_results(entities_data, analysis_data, relationships, output_dir):
    """Save extracted entities and analysis results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all characters
    characters_df = pd.DataFrame([
        {"character": entities_data["characters"][char], "normalized_name": char, "mentions": entities_data["character_mentions"][char]}
        for char in entities_data["characters"]
    ])
    characters_df = characters_df.sort_values("mentions", ascending=False)
    characters_df.to_csv(os.path.join(output_dir, "characters.csv"), index=False)
    
    # Save all locations
    locations_df = pd.DataFrame([
        {"location": entities_data["locations"][loc], "normalized_name": loc, "mentions": entities_data["location_mentions"][loc]}
        for loc in entities_data["locations"]
    ])
    locations_df = locations_df.sort_values("mentions", ascending=False)
    locations_df.to_csv(os.path.join(output_dir, "locations.csv"), index=False)
    
    # Save scene entities
    scene_entities_df = pd.DataFrame(entities_data["scene_entities"])
    scene_entities_df["characters"] = scene_entities_df["characters"].apply(lambda x: ", ".join([entities_data["characters"].get(char, char) for char in x]))
    scene_entities_df["locations"] = scene_entities_df["locations"].apply(lambda x: ", ".join([entities_data["locations"].get(loc, loc) for loc in x]))
    scene_entities_df.to_csv(os.path.join(output_dir, "scene_entities.csv"), index=False)
    
    # Save character analysis
    character_analysis_df = pd.DataFrame(
        [("Main", entities_data["characters"][char], mentions) for char, mentions in analysis_data["main_characters"]] +
        [("Supporting", entities_data["characters"][char], mentions) for char, mentions in analysis_data["supporting_characters"]] +
        [("Minor", entities_data["characters"][char], mentions) for char, mentions in analysis_data["minor_characters"]],
        columns=["role", "character", "mentions"]
    )
    character_analysis_df.to_csv(os.path.join(output_dir, "character_analysis.csv"), index=False)
    
    # Save location analysis
    location_analysis_df = pd.DataFrame(
        [("Main", entities_data["locations"][loc], mentions) for loc, mentions in analysis_data["main_locations"]] +
        [("Secondary", entities_data["locations"][loc], mentions) for loc, mentions in analysis_data["secondary_locations"]],
        columns=["importance", "location", "mentions"]
    )
    location_analysis_df.to_csv(os.path.join(output_dir, "location_analysis.csv"), index=False)
    
    # Save character relationships
    relationships_df = pd.DataFrame([
        (entities_data["characters"][char1], entities_data["characters"][char2], co_occurrences) 
        for char1, char2, co_occurrences in relationships
    ], columns=["character1", "character2", "co_occurrences"])
    relationships_df.to_csv(os.path.join(output_dir, "character_relationships.csv"), index=False)
    
    print(f"Results saved to {output_dir}")

def main():
    args = parse_args()
    
    # Load GLiNER model
    print(f"Loading GLiNER model: {args.model}")
    model = GLiNER.from_pretrained(args.model)
    
    # Read screenplay
    text = read_screenplay(args.input_file)
    
    # Preprocess screenplay
    scenes = preprocess_screenplay(text)
    
    # Extract entities
    entities_data = extract_entities_with_gliner(model, scenes)
    
    # Analyze entities
    analysis_data = analyze_entities(entities_data)
    
    # Analyze character relationships
    relationships = analyze_character_relationships(entities_data)
    
    # Save results
    save_results(entities_data, analysis_data, relationships, args.output_dir)
    
    # Print summary
    print("\n--- SCREENPLAY ANALYSIS SUMMARY ---")
    print(f"Found {len(entities_data['characters'])} characters and {len(entities_data['locations'])} locations.")
    print("\nMain Characters:")
    for char, mentions in analysis_data["main_characters"]:
        print(f"- {entities_data['characters'][char]} ({mentions} mentions)")
    
    print("\nMain Locations:")
    for loc, mentions in analysis_data["main_locations"]:
        print(f"- {entities_data['locations'][loc]} ({mentions} mentions)")
    
    print("\nTop Character Relationships:")
    for char1, char2, count in relationships[:5]:
        print(f"- {entities_data['characters'][char1]} & {entities_data['characters'][char2]} appear together in {count} scenes")

if __name__ == "__main__":
    main()