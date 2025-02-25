# embeddings.py
from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL
)
import numpy as np

class EmbeddingGenerator:
    EXPECTED_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072
    }

    def __init__(self):
        print(f"Debug - Initializing EmbeddingGenerator")
        print(f"Debug - Configured model in config.py: {OPENAI_EMBEDDING_MODEL}")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        if OPENAI_EMBEDDING_MODEL not in self.EXPECTED_DIMENSIONS:
            raise ValueError(
                f"Unsupported embedding model: {OPENAI_EMBEDDING_MODEL}. "
                f"Supported models are: {list(self.EXPECTED_DIMENSIONS.keys())}"
            )
        
        self.expected_dim = self.EXPECTED_DIMENSIONS[OPENAI_EMBEDDING_MODEL]
        print(f"Debug - Expected dimensions for {OPENAI_EMBEDDING_MODEL}: {self.expected_dim}")

    async def generate_embedding(self, text):
        try:
            print(f"\nDebug - Generating embedding for text (first 50 chars): {text[:50]}...")
            print(f"Debug - Using model: {OPENAI_EMBEDDING_MODEL}")
            
            # Generate embedding
            response = self.client.embeddings.create(
                input=text,
                model=OPENAI_EMBEDDING_MODEL,
                encoding_format="float"
            )
            
            # Extract embedding and check dimensions
            embedding = response.data[0].embedding
            actual_dim = len(embedding)
            
            print(f"Debug - Generated embedding:")
            print(f"Debug - Dimensions: {actual_dim}")
            print(f"Debug - Expected dimensions: {self.expected_dim}")
            print(f"Debug - First 5 values: {embedding[:5]}")
            print(f"Debug - Value range: {min(embedding)} to {max(embedding)}")
            
            # Check if dimensions match expected
            if actual_dim != self.expected_dim:
                error_msg = (
                    f"Dimension mismatch error!\n"
                    f"Expected dimensions: {self.expected_dim}\n"
                    f"Actual dimensions: {actual_dim}\n"
                    f"Model configured: {OPENAI_EMBEDDING_MODEL}\n"
                    f"Response model: {response.model}"
                )
                print(f"Error - {error_msg}")
                raise ValueError(error_msg)
            
            # Additional validation
            if not all(isinstance(x, float) for x in embedding):
                raise ValueError("Invalid embedding format - not all values are floats")
            
            print(f"Debug - Embedding validation successful")
            return embedding

        except Exception as e:
            error_msg = f"Error generating embedding: {str(e)}"
            print(f"Error - {error_msg}")
            print(f"Debug - Full error details: {e}")
            raise

    def _validate_embedding(self, embedding):
        """Additional validation checks for embeddings"""
        if not isinstance(embedding, list):
            raise ValueError(f"Embedding must be a list, got {type(embedding)}")
            
        if len(embedding) != self.expected_dim:
            raise ValueError(
                f"Invalid embedding dimensions. Expected {self.expected_dim}, got {len(embedding)}"
            )
            
        if not all(isinstance(x, float) for x in embedding):
            raise ValueError("All embedding values must be floats")