import pandas as pd
import json
# from rag2 import RagProcessor
from langfuserag import RagProcessor

def main():
    with open("film_script2.txt", "r", encoding="utf-8") as f:
        script_text = f.read()

    df = pd.read_csv("filtered_characters1.csv")
    character_list = df["normalized_name"].dropna().unique().tolist()
    mentions_dict = dict(zip(df["normalized_name"], df["mentions"]))

    rag = RagProcessor()
    # Extract character details 
    character_details = rag.get_entity_details(
        script_text=script_text,
        entity_list=character_list,
        entity_type="character",
        mentions_dict=mentions_dict,


    )

    
    with open("character_details_outputfinal1.json", "w", encoding="utf-8") as f:
        json.dump(character_details, f, indent=2, ensure_ascii=False)

    print(f"Saved character details to character_details_output8.json")

if __name__ == "__main__":
    main()