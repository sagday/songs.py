import csv
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read song data from the CSV file
songs = []
with open('SONGS2.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        songs.append(row)

# Encode the descriptions and generate BERT embeddings
embeddings = []
for row in songs:
    description = row['DESCRIPTION']
    # Tokenize the description
    tokens = tokenizer.encode(description, add_special_tokens=True)
    # Convert tokens to tensors
    input_ids = torch.tensor(tokens).unsqueeze(0)
    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

# Reshape the embeddings
embeddings = torch.tensor(embeddings)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Now, suppose a user likes "Sina". We can recommend another song based on cosine similarity.
liked_song = "Sina"
liked_song_index = next(index for index, song in enumerate(songs) if song['TITLE'] == liked_song)

# Find the most similar songs
similar_song_indices = similarity_matrix[liked_song_index].argsort()[::-1][1:4]  # Exclude the liked song itself
recommended_songs = [songs[index]['TITLE'] for index in similar_song_indices]

print("Because you liked " + liked_song + ", we recommend: " + ", ".join(recommended_songs))

