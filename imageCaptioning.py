import os
import json
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

# File e directory
model_file = 'cnn_rnn_model.pth'
vocab_file = 'vocab.json'
img_dir = r'C:\Users\Giulio\anaconda3\envs\pythorch_env\src\ImageDescription\flickr30k_images'

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Ridimensiona le immagini a 300x300 per adattarsi a EfficientNet-B3
    transforms.ToTensor(),          # Converte l'immagine in un tensore
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Media dei pixel dell'immagine per la normalizzazione
        std=[0.229, 0.224, 0.225]    # Deviazione standard dei pixel dell'immagine per la normalizzazione
    )
])

# Funzione per caricare il vocabolario
def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)  # Carica il vocabolario dal file JSON
    inv_vocab = {v: k for k, v in vocab.items()}  # Inverte il dizionario per ottenere mappature da indice a parola
    return vocab, inv_vocab

# Carica il vocabolario
with open(vocab_file, 'r') as f:
    vocab = json.load(f)
inv_vocab = {v: k for k, v in vocab.items()}

# Encoder CNN
class CNN(nn.Module):
    def __init__(self, embed_size):
        super(CNN, self).__init__()
        efficient_net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)  # Inizializza EfficientNet-B3 con pesi pre-addestrati
        for param in efficient_net.parameters():
            param.requires_grad = False  # Congela i parametri della rete pre-addestrata per evitare modifiche durante l'addestramento
        modules = list(efficient_net.children())[:-1]  # Escludi l'ultimo layer di  classificazione
        self.efficient_net = nn.Sequential(*modules)  # Usa i moduli rimanenti come estrattore di caratteristiche
        self.embed = nn.Linear(1536, embed_size)  # Layer lineare per mappare le caratteristiche a una dimensione di embedding specificata

    def forward(self, image):
        features = self.efficient_net(image)  # Estrai le caratteristiche dall'immagine
        features = features.view(features.size(0), -1)  # Appiattisci le caratteristiche per il layer lineare
        features = self.embed(features)  # Applica il layer lineare per ottenere l'embedding dell'immagine
        return features

# Decoder LSTM
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)  # Layer di embedding per le parole
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM per generare sequenze
        self.linear = nn.Linear(hidden_size, vocab_size)  # Layer lineare per predire la parola successiva
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))  # Stato iniziale nascosto e cella per LSTM

    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:, :-1])  # Embedding delle parole nella didascalia
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)  # Combina le caratteristiche dell'immagine con l'embedding delle parole
        lstm_out, self.hidden = self.lstm(embeddings)  # Passa attraverso LSTM
        outputs = self.linear(lstm_out)  # Ottieni le probabilità per ogni parola del vocabolario
        return outputs

# Carica e preprocessa l'immagine
def preprocess_image(file_path):
    image = Image.open(file_path).convert('RGB')  # Carica l'immagine e convertila in RGB
    image = transform(image).unsqueeze(0)  # Applica le trasformazioni e aggiungi una dimensione batch
    return image

# Funzione per generare la didascalia
def generate_caption(image_features, decoder, vocab, inv_vocab, max_length=70):
    caption_indices = [vocab['<START>']]  # Inizia la didascalia con il token <START>
    caption_tensor = torch.tensor(caption_indices).unsqueeze(0).to(device)  # Converti gli indici in tensore e spostalo sulla GPU

    with torch.no_grad():
        for _ in range(max_length):  # Limita la lunghezza della didascalia
            outputs = decoder(image_features, caption_tensor)  # Ottieni le probabilità delle parole per la didascalia corrente
            _, predicted = outputs[:, -1, :].max(1)  # Trova la parola con la probabilità più alta
            predicted_word = predicted.item()  # Ottieni l'indice della parola predetta

            if predicted_word == vocab['<END>']:  # Se il token <END> è previsto interrompo la generazione
                break

            caption_indices.append(predicted_word)  # Aggiungi la parola predetta
            caption_tensor = torch.tensor(caption_indices).unsqueeze(0).to(device)  # Aggiorna il tensore della didascalia

    decoded_caption = [inv_vocab.get(idx, '<UNK>') for idx in caption_indices[1:-1]]  # Decodifica gli indici in parole
    unique_caption = remove_duplicates(decoded_caption)  # Rimuovi duplicati dalla didascalia
    return ' '.join(unique_caption)  # Restituisci la didascalia come stringa

# Funzione per rimuovere duplicati consecutivi dalla didascalia
def remove_duplicates(values):
    if not values:
        return values

    unique_values = [values[0]]  # Inizia con la prima parola

    for i in range(1, len(values)):
        if values[i] != values[i - 1]:  # Confronta con la parola precedente
            unique_values.append(values[i])  # Aggiungi la parola solo se non è uguale alla precedente

    return unique_values

# Funzione per selezionare un'immagine e generare la didascalia
def select_image():
    # Apri la finestra di dialogo per selezionare un'immagine
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")],  # Filtra solo i file immagine
        title="Select an Image"  # Titolo della finestra di dialogo
    )

    if file_path:
        # Carica e preprocessa l'immagine
        image = preprocess_image(file_path)
        image_pil = Image.open(file_path).resize((300, 300))  # Carica l'immagine e ridimensionala a 300x300 per la visualizzazione

        # Genera la didascalia
        with torch.no_grad():
            image_features = encoder(image.to(device))  # Ottieni le caratteristiche dell'immagine dall'encoder
            caption = generate_caption(image_features, decoder, vocab, inv_vocab)  # Genera la didascalia

        # Mostra l'immagine e la didascalia
        img_tk = ImageTk.PhotoImage(image_pil)  # Converte l'immagine PIL in un formato compatibile con Tkinter
        image_label.config(image=img_tk)  # Imposta l'immagine sulla label
        image_label.image = img_tk  # Mantieni un riferimento all'immagine per evitare che venga rimossa dal garbage collector
        caption_label.config(text=caption)  # Imposta la didascalia sulla label

# Configura la finestra principale dell'interfaccia grafica
root = tk.Tk()
root.title("Image Caption Generator")  # Titolo della finestra principale

# Aggiungi un pulsante per selezionare un'immagine
button = tk.Button(root, text="Select an Image", command=select_image)  # Pulsante per selezionare un'immagine
button.pack(pady=10)  # Aggiungi il pulsante alla finestra con un margine verticale

# Aggiungi una label per visualizzare l'immagine
image_label = tk.Label(root)  # Label per visualizzare l'immagine
image_label.pack(pady=10)  # Aggiungi la label alla finestra con un margine verticale

# Aggiungi una label per visualizzare la didascalia
caption_label = tk.Label(root, text="", wraplength=400, justify="left")  # Label per visualizzare la didascalia
caption_label.pack(pady=10)  # Aggiungi la label alla finestra con un margine verticale

# Configura il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa la GPU se disponibile altrimenti usa la CPU

# Inizializza e carica i pesi dei modelli
encoder = CNN(embed_size=256).to(device)  # Crea l'encoder CNN e spostalo sul dispositivo
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab)).to(device)  # Crea il decoder RNN e spostalo sul dispositivo

checkpoint = torch.load(model_file, map_location=device)  # Carica lo stato del modello dal file
encoder.load_state_dict(checkpoint['encoder_state_dict'])  # Carica i pesi dell'encoder
decoder.load_state_dict(checkpoint['decoder_state_dict'])  # Carica i pesi del decoder

encoder.eval()  # Imposta l'encoder in modalità di valutazione
decoder.eval()  # Imposta il decoder in modalità di valutazione

# Avvia la GUI
root.mainloop()  # Avvia il loop principale dell'interfaccia grafica
