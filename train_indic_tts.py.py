import os
import json
import torch
import re
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from fish_speech.model import FishSpeech  # Ensure FishSpeech is imported
import sys
import os

sys.path.append(os.path.abspath("fish-speech"))
from fish_speech.model import FishSpeech  # Now this should wor

# Step 1: Convert Indic-TTS Dataset into Required Format
DATA_DIR = "data/train"
TRANSCRIPT_FILE = "data/transcripts.json"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Convert text files into JSON format
transcriptions = {}

for root, _, files in os.walk("indic_tts/Indic-TTS-master"):
    for filename in files:
        if filename.endswith(".wav"):
            text_file = os.path.join(root, filename.replace(".wav", ".txt"))
            if os.path.exists(text_file):
                with open(text_file, "r", encoding="utf-8") as f:
                    transcriptions[filename] = f.read().strip()
                os.rename(os.path.join(root, filename), os.path.join(DATA_DIR, filename))

# Save transcripts as JSON
with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
    json.dump(transcriptions, f, ensure_ascii=False, indent=4)

print(f"✅ Dataset prepared with {len(transcriptions)} samples.")

# Step 2: Update Tokenizer for Indic Languages
INDIC_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u0900-\u097F\u0B80-\u0BFF "

class IndicTokenizer:
    def __init__(self):
        self.chars = sorted(set(INDIC_CHARACTERS))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices):
        return "".join([self.idx_to_char[i] for i in indices])

tokenizer = IndicTokenizer()

# Step 3: Create Custom Indic-TTS Dataset
class IndicSpeechDataset(Dataset):
    def __init__(self, audio_dir, transcript_file):
        self.audio_dir = audio_dir
        with open(transcript_file, "r", encoding="utf-8") as f:
            self.transcriptions = json.load(f)
        self.audio_files = list(self.transcriptions.keys())

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        text = self.transcriptions[self.audio_files[idx]]
        text_encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        return waveform, text_encoded

dataset = IndicSpeechDataset(DATA_DIR, TRANSCRIPT_FILE)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"✅ Dataset loaded with {len(dataset)} samples.")

# Step 4: Train Fish-Speech on Indic Languages
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishSpeech().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 10

for epoch in range(EPOCHS):
    total_loss = 0
    for waveform, text_encoded in dataloader:
        waveform, text_encoded = waveform.to(device), text_encoded.to(device)
        optimizer.zero_grad()
        output = model(waveform)
        loss = loss_fn(output, text_encoded)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "fish_speech_indic.pth")
print("✅ Model trained and saved as `fish_speech_indic.pth`")

# Step 5: Inference (Test the trained model)
def infer(text):
    model.eval()
    input_text = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    output_audio = model.synthesize(input_text)
    return output_audio

sample_text = "नमस्ते दुनिया"  # Example in Hindi
output_audio = infer(sample_text)
torchaudio.save("output.wav", output_audio.cpu(), 16000)
print("✅ Inference complete. Check `output.wav`")
