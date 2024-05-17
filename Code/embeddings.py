# embeddings.py
import os
import numpy as np
from pickle import dump

class EmbeddingMatrixBuilder:
    def __init__(self, embedding_file='glove.6B.50d.txt', embedding_dim=50):
        # Navigate to the Data directory to access the embedding file
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))
        self.embedding_file = os.path.join(base_dir, embedding_file)
        self.embedding_dim = embedding_dim
        self.embeddings_index = self._load_embeddings()

    def _load_embeddings(self):
        embeddings_index = dict()
        with open(self.embedding_file, 'r', encoding="utf8") as fid:
            for line in fid:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def build_embedding_matrix(self, word_index):
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))
        for word, idx in word_index.items():
            embed_vector = self.embeddings_index.get(word)
            if embed_vector is not None:
                embedding_matrix[idx] = embed_vector
        return embedding_matrix

    def save_embedding_matrix(self, embedding_matrix, output_file='../Generated_files/embedding_matrix.pkl'):
        # Resolve the absolute path for the output file
        output_file = os.path.abspath(output_file)
        output_dir = os.path.dirname(output_file)

        # Debugging print statements
        print(f"Resolved output directory: {output_dir}")
        print(f"Saving embedding matrix to: {output_file}")

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the embedding matrix
        try:
            with open(output_file, 'wb') as fid:
                dump(embedding_matrix, fid)
            print(f"Embedding matrix saved successfully to: {output_file}")
        except Exception as e:
            print(f"Error saving embedding matrix: {e}")
