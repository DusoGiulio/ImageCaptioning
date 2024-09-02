import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

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
        print(features.shape)
        features = features.view(features.size(0), -1)
        print(features.shape)
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
