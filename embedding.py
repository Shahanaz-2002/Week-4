import numpy as np
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class BioBERTEmbedding:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embedding = self.mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

class EmbeddingEngine:
    def __init__(self, embedding_dim: int = 768):  # Changed to 768 (BioBERT dim)
        self.embedding_dim = embedding_dim
        self.embedding_model = BioBERTEmbedding()  # Real BioBERT model!
    
   
    # Public Method
    
    def generate_embedding(self, case_data: Dict[str, Any]) -> np.ndarray:
        processed_text = self._preprocess_case(case_data)
        embedding_vector = self.embedding_model.get_embedding(processed_text)[:self.embedding_dim]
        return embedding_vector
    
    
    # Private Methods
   
    def _preprocess_case(self, case_data: Dict[str, Any]) -> str:
        symptoms = case_data.get("symptoms", [])
        diagnosis = case_data.get("diagnosis", "")
        notes = case_data.get("notes", "")
        combined_text = " ".join(symptoms) + " " + diagnosis + " " + notes
        return combined_text.lower().strip()
