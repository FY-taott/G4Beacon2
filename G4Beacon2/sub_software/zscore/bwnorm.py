#bwnorm.py
import numpy as np
import sys

def read_bedgraph(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                data.append(float(parts[3]))
    return data

def z_score_normalize(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = [(x - mean) / std_dev for x in data]
    return normalized_data

if len(sys.argv) != 2:
    print("Usage: python bwnorm.py <input_bedgraph_file>")
    sys.exit(1)

input_file = sys.argv[1]
data = read_bedgraph(input_file)
output_prefix = input_file.split(".")[0]
z_score_normalized_data = z_score_normalize(data)

with open(f"{output_prefix}_z_score_normalized.bedgraph", "w") as f:
    for i, value in enumerate(z_score_normalized_data):
        f.write(f"{data[i]}\t{value}\n")