# main.py

from embedding import EmbeddingEngine
from similarity_engine import SimilarityEngine
from insight_generator import InsightGenerator
from utils import (
    load_case_database,
    validate_case_input,
    format_output,
    log
)


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

def main():

    log("Loading case database...")
    case_database = load_case_database(r"D:\chiselon\Week 0\Week_0_Prep_Week_Ssample Data_clinic_cases.csv")

    log("Initializing embedding engine...")
    embedding_engine = EmbeddingEngine(embedding_dim=128)

    log("Generating embeddings for case database...")
    case_embeddings = {}

    for case_id, case_data in case_database.items():
        embedding = embedding_engine.generate_embedding(case_data)
        case_embeddings[case_id] = embedding

    log("Initializing similarity engine...")
    similarity_engine = SimilarityEngine(case_embeddings)

    log("Initializing insight generator...")
    insight_generator = InsightGenerator(case_database)

    # ---------------------------------------------------
    # Example New Patient (Replace Later with API Input)
    # ---------------------------------------------------

    new_case = {
        "case_id": "NEW001",
        "symptoms": ["chest pain", "shortness of breath"],
        "diagnosis": "",
        "notes": "Patient reports fatigue and mild dizziness."
    }

    # Validate Input
    validate_case_input(new_case)

    log("Generating query embedding...")
    query_embedding = embedding_engine.generate_embedding(new_case)

    log("Retrieving top similar cases...")
    top_matches = similarity_engine.retrieve_top_k(
        query_embedding,
        top_k=3
    )

    log("Generating clinical insight...")
    insight = insight_generator.generate_insight(top_matches)

    # Format Output
    final_output = format_output(
        query_case_id=new_case["case_id"],
        top_matches=top_matches,
        insight=insight
    )

    log("Pipeline completed successfully.\n")

    print(final_output)


# ---------------------------------------------------
# Entry Point
# ---------------------------------------------------

if __name__ == "__main__":
    main()