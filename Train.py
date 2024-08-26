import os
import json
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from tqdm import tqdm

# Vari link e nomi usati
img_dir = r'C:\Users\Giulio\anaconda3\envs\pythorch_env\src\ImageDescription\flickr30k_images'  # Directory contenente le immagini
model_file = 'cnn_rnn_model.pth'  # Nome del file per salvare il modello addestrato
log_file = 'training_log.txt'  # Nome del file per il log dell'addestramento
csv_file = r'C:\Users\Giulio\anaconda3\envs\pythorch_env\src\ImageDescription\flickr30k_images\results.csv'  # File CSV con le annotazioni delle immagini
vocab_file = 'vocab.json'  # Nome del file per salvare il vocabolario

# Carica il vocabolario
with open(vocab_file, 'r') as f:
    vocab = json.load(f)

# Funzione per creare un vocabolario
def build_vocab(captions, threshold=1):
    counter = defaultdict(int)
    for caption in captions:
        if isinstance(caption, str):
            for word in caption.split(' '):
                counter[word] += 1
    vocab = {word: idx+4 for idx, (word, count) in enumerate(counter.items()) if count >= threshold}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab['<START>'] = 2
    vocab['<END>'] = 3
    return vocab

def save_vocab(csv_file, vocab_file):
    df = pd.read_csv(csv_file, sep='|')
    vocab = build_vocab(df.iloc[:, 2])
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)

# Definizione delle trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Ridimensiona l'immagine
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Creazione del dataset
class FlickrDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, vocab=None):
        self.annotations = pd.read_csv(csv_file, sep='|')
        self.img_dir = img_dir
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0].strip())
        image = Image.open(img_name).convert('RGB')
        caption = self.annotations.iloc[idx, 2]

        if isinstance(caption, str):
            caption = caption.strip()
        else:
            caption = ""

        caption_indices = caption_to_indices(caption, self.vocab)
        caption_tensor = torch.tensor(caption_indices)

        if self.transform:
            image = self.transform(image)

        return image, caption_tensor

def caption_to_indices(caption, vocab):
    indices = [vocab.get('<START>', 2)]
    indices.extend([vocab.get(word, vocab['<UNK>']) for word in caption.split(' ')])
    indices.append(vocab.get('<END>', 3))
    return indices

def collate_fn(vocab):
    def collate_fn_inner(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        captions = pad_sequence(captions, batch_first=True, padding_value=vocab['<PAD>'])
        return images, captions
    return collate_fn_inner

# Definizione dell'encoder CNN con EfficientNet-B3
class CNN(nn.Module):
    def __init__(self, embed_size):
        super(CNN, self).__init__()
        efficient_net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Congela tutti i parametri dell'EfficientNet-B3
        for param in efficient_net.parameters():
            param.requires_grad = False

        # Rimuovi l'ultimo livello di classificazione
        modules = list(efficient_net.children())[:-1]
        self.efficient_net = nn.Sequential(*modules)

        # Aggiungi un nuovo livello di embedding
        self.embed = nn.Linear(1536, embed_size)

        # Sblocca solo il nuovo livello
        for param in self.embed.parameters():
            param.requires_grad = True

    def forward(self, image):
        features = self.efficient_net(image)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

# Definizione del decoder LSTM
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)
        lstm_out, self.hidden = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

# Parametri di addestramento
embed_size = 256
hidden_size = 512
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Inizializza dataset e dataloaders
dataset = FlickrDataset(csv_file=csv_file, img_dir=img_dir, transform=transform, vocab=vocab)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn(vocab))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn(vocab))

# Inizializza i modelli
encoder = CNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size=len(vocab))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Funzione di perdita e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.embed.parameters()), lr=learning_rate)

# Ciclo di addestramento con fase di validazione
print("Start Training")
with open(log_file, 'w') as f:
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0

        # Training
        for images, captions in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions)
            loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_dataloader)
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}\n")
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, captions in val_dataloader:
                images = images.to(device)
                captions = captions.to(device)

                features = encoder(images)
                outputs = decoder(features, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}\n")
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # Save model checkpoints
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'val_loss': avg_val_loss
        }, model_file)

print('Training complete!')
