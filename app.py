import json
import torch
import torch.nn as nn
import tkinter as tk
from collections import Counter
from torch.nn.functional import one_hot

# Load data
with open("data.json", "r") as f:
    phrases = json.load(f)

# Build vocab
tokens = [word for phrase in phrases for word in phrase.split()]
vocab = {word: i for i, word in enumerate(set(tokens))}
reverse_vocab = {i: word for word, i in vocab.items()}
vocab_size = len(vocab)

# Encode data
def encode(phrase):
    return [vocab[word] for word in phrase.split() if word in vocab]

encoded_data = [encode(p) for p in phrases]

# Model
class TinyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        return self.fc(out)

model = TinyModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Train
for epoch in range(100):
    for seq in encoded_data:
        inputs = torch.tensor(seq[:-1]).unsqueeze(0)
        targets = torch.tensor(seq[1:])
        outputs = model(inputs)[0]
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Generate
def generate(prompt, max_len=10):
    tokens = encode(prompt)
    for _ in range(max_len):
        input_tensor = torch.tensor(tokens).unsqueeze(0)
        output = model(input_tensor)[0][-1]
        next_token = output.argmax().item()
        tokens.append(next_token)
    return ' '.join([reverse_vocab[t] for t in tokens])

# UI
def on_generate():
    prompt = entry.get()
    result = generate(prompt)
    output_label.config(text=result)

root = tk.Tk()
root.title("Your Local AI")

entry = tk.Entry(root, width=50)
entry.pack()

tk.Button(root, text="Generate", command=on_generate).pack()
output_label = tk.Label(root, text="", wraplength=400)
output_label.pack()

root.mainloop()
