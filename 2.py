import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader

# 1. Dataset nhỏ (toy data)
data = [
    ("i love you", "je t aime"),
    ("he is tall", "il est grand"),
    ("she is smart", "elle est intelligente"),
    ("good morning", "bonjour"),
    ("thank you", "merci"),
]

# 2. Tạo từ điển (vocab)
src_vocab = {"<pad>":0, "<sos>":1, "<eos>":2}
tgt_vocab = {"<pad>":0, "<sos>":1, "<eos>":2}

for src, tgt in data:
    for w in src.split():
        if w not in src_vocab:
            src_vocab[w] = len(src_vocab)
    for w in tgt.split():
        if w not in tgt_vocab:
            tgt_vocab[w] = len(tgt_vocab)

inv_tgt_vocab = {v:k for k,v in tgt_vocab.items()}

def encode(sentence, vocab):
    return [vocab["<sos>"]] + [vocab[w] for w in sentence.split()] + [vocab["<eos>"]]

# 3. Dataset + Dataloader
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_tensor = torch.tensor(encode(src, src_vocab))
        tgt_tensor = torch.tensor(encode(tgt, tgt_vocab))
        return src_tensor, tgt_tensor

def collate_fn(batch):
    src, tgt = batch[0]
    src = src.unsqueeze(0)  
    tgt = tgt.unsqueeze(0)
    return src, tgt

dataset = TranslationDataset(data)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# 4. Mô hình Transformer
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        src_emb = self.src_emb(src) * (self.d_model ** 0.5)
        tgt_emb = self.tgt_emb(tgt) * (self.d_model ** 0.5)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output)

# 5. Huấn luyện
model = TransformerModel(len(src_vocab), len(tgt_vocab))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(30):
    total_loss = 0
    for src, tgt in loader:
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1]) 
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 6. Dịch thử
def translate(sentence):
    model.eval()
    src = torch.tensor(encode(sentence, src_vocab)).unsqueeze(0)
    tgt = torch.tensor([[tgt_vocab["<sos>"]]])

    for _ in range(10):
        output = model(src, tgt)
        next_word = output.argmax(2)[:, -1].item()
        tgt = torch.cat([tgt, torch.tensor([[next_word]])], dim=1)
        if next_word == tgt_vocab["<eos>"]:
            break

    return " ".join([inv_tgt_vocab[i.item()] for i in tgt[0][1:-1]])

print("\n Input: 'i love you'")
print(" Output:", translate("i love you"))
