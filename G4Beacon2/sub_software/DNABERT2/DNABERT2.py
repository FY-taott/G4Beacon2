# DNABERT2
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def dna_to_embedding(dna_sequence, tokenizer, model):
    inputs = tokenizer(dna_sequence, return_tensors='pt')['input_ids']
    hidden_states = model(inputs)[0]  # [1, sequence_length, 768]
    embedding_mean = torch.mean(hidden_states[0], dim=0)
    return embedding_mean

def load_dna_sequences_from_txt(file_path):
    with open(file_path, 'r') as file:
        dna_sequences = file.readlines()
    dna_sequences = [seq.strip() for seq in dna_sequences]
    return dna_sequences

def save_embedding_to_csv(file_path, embedding):
    with open(file_path, 'a') as file:
        # Convert the tensor to a NumPy array without gradients
        embedding_np = embedding.detach().numpy()
        # Convert the array to a comma-separated string
        embedding_str = ','.join(str(val) for val in embedding_np)
        # Write the string to the file followed by a new line
        file.write(embedding_str + '\n')

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
    model = AutoModel.from_pretrained("./model", trust_remote_code=True)

    txt_file_path = sys.argv[1]
    dna_sequences = load_dna_sequences_from_txt(txt_file_path)
    output_csv_file_path = "DNABERT2_embedding.csv"

    for dna_sequence in tqdm(dna_sequences, desc="processing"):
        embedding = dna_to_embedding(dna_sequence, tokenizer, model)
        save_embedding_to_csv(output_csv_file_path, embedding)