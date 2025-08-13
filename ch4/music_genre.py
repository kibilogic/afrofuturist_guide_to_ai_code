# =========== put this code block in a new cell ===========
!pip install --upgrade pip

# Uninstall ALL potentially conflicting libraries
!pip uninstall -y torch torchvision torchaudio transformers datasets accelerate fsspec gcsfs torchcodec 

# May take a moment to install
!pip install torch torchvision torchaudio

# Install/Upgrade libraries
!pip install --upgrade transformers datasets accelerate librosa soundfile fsspec gcsfs huggingface_hub

# --- Verify installations (verification does not fail, Good ) ---
print("\n--- Verifying Installations ---")
try:
    import torch
    import torchvision
    import transformers
    import datasets
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"transformers version: {transformers.__version__}")
    print(f"datasets version: {datasets.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (from torch): {torch.version.cuda}")
except Exception as e:
    print(f"Verification failed: {e}")

# =========== put this code block in a new cell ===========
from google.colab import userdata
import os
from huggingface_hub import login

# Get Hugging Face token and log in
hf_token = userdata.get("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token
login(token=hf_token)

# put this code block in a new cell
!pip install --upgrade torchcodec

# =========== put this code block in a new cell ===========
# =========== Restart runtime in colab Runtime -> Restart Session ===========

from datasets import load_dataset, Audio
import torch
import torchaudio
import librosa
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# =========== may take a moment to run ===========

# Define sampling rate 
TARGET_SAMPLING_RATE = 16000

# Load dataset
dataset = load_dataset("lewtun/music_genres_small")

# Load model and processor
model_id = "openai/whisper-tiny"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Look for GPU otherwise using CPU  
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Define the embedding 
def extract_embedding(batch):
    audio_array = batch["audio"]["array"]
    sampling_rate = batch["audio"]["sampling_rate"]

    # Resample, if necessary 
    if sampling_rate != TARGET_SAMPLING_RATE:
        audio_tensor = torch.from_numpy(audio_array).float()
        audio_tensor = torchaudio.functional.resample(
            audio_tensor,
            orig_freq=sampling_rate,
            new_freq=TARGET_SAMPLING_RATE
        )
        audio_array = audio_tensor.numpy()

    # Process audio 
    input_features = processor(
        audio_array,
        sampling_rate=TARGET_SAMPLING_RATE,
        return_tensors="pt"
    ).input_features.to(device)

    # Extract embeddings
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_features)
        embedding = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    batch["embedding"] = embedding
    return batch

# Apply extract_embedding function to the train split
dataset['train'] = dataset['train'].map(extract_embedding, batched=False)

# Set to numpy so embedding is returned as a NumPy array
dataset['train'].set_format(type="numpy", columns=['audio', 'song_id', 'genre_id', 'genre', 'embedding'])
print(dataset)
print(dataset['train'][0]["embedding"].shape)

# =========== put this code block in a new cell ===========
import numpy as np
import pandas as pd 

# Extract Embeddings and Labels 
embeddings = np.array(dataset['train']['embedding'])
genres = dataset['train']['genre'] 

print(f"Shape of original embeddings: {embeddings.shape}") 

# =========== put this code block in a new cell ===========
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Extract Embeddings and Labels 
embeddings = np.array(dataset['train']['embedding'])
genres = dataset['train']['genre'] 

print(f"Shape of original embeddings: {embeddings.shape}") 

# Dimensionality Reduction (t-SNE) 
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)

print(f"Shape of 2D embeddings: {embeddings_2d.shape}")

# Setup how to plot 
df_plot = pd.DataFrame({
    'TSNE-1': embeddings_2d[:, 0],
    'TSNE-2': embeddings_2d[:, 1],
    'Genre': genres
})

# KMeans clustering  
kmeans = KMeans(n_clusters=10, random_state=42)
df_plot['Cluster'] = kmeans.fit_predict(embeddings)

# Map each cluster to the most common genre 
cluster_to_genre = (
    df_plot.groupby('Cluster')['Genre']
    .agg(lambda x: x.value_counts().idxmax())  
    .to_dict()
)

df_plot['ClusterLabel'] = df_plot['Cluster'].map(cluster_to_genre)

# Plot using genre label for clusters 
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='TSNE-1', y='TSNE-2',
    hue='ClusterLabel',
    palette='tab10',
    data=df_plot,
    legend="full"
)
plt.title('2D t-SNE Visualization of Music Clusters by Dominant Genre')
plt.xlabel('TSNE-1')
plt.ylabel('TSNE-2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
