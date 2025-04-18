import pandas as pd
import torch

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import ColBERTModel, ColBERTTokenizer

import random

# Load Pre-trained ColBERTv2 model and tokenizer from Hugging Face
tokenizer = ColBERTTokenizer.from_pretrained("bert-base-uncased")
model = ColBERTModel.from_pretrained("bert-base-uncased")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


class ColBERTDataset(Dataset):
        def __init__(self, triplets_file, tokenizer, max_length=128):
            self.data = pd.read_csv(triplets_file, sep='\t', header=None, names=['query', 'positive', 'negative'])
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            query = self.data.iloc[idx]['query']
            positive_doc = self.data.iloc[idx]['positive']
            negative_doc = self.data.iloc[idx]['negative']

            # Tokenize the query and both the positive and negative documents
            query_encodings = self.tokenizer(query, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
            pos_encodings = self.tokenizer(positive_doc, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
            neg_encodings = self.tokenizer(negative_doc, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
            return {
                'query': query_encodings,
                'positive': pos_encodings,
                'negative': neg_encodings
            }

# Create the dataset and dataloader
triplets_file = "triples.tsv"
dataset = ColBERTDataset(triplets_file, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Loss function for fine-tuning (Triplet Loss)
class TripletLoss(nn.Module):
        def __init__(self, margin=1.0):
            super(TripletLoss, self).__init__()
            self.margin = margin
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

        def forward(self, query_embeds, pos_embeds, neg_embeds):
            return self.loss_fn(query_embeds, pos_embeds, neg_embeds)

# Initialize model and optimizer
model = ColBERTModel.from_pretrained("bert-base-uncased").to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)


def train():
        # Triplet loss criterion
        triplet_loss = TripletLoss().to(device)

        # Training loop
        epochs = 3
        for epoch in range(epochs):
            model.train()
            total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

        # Tokenize input and move to device
        query_input_ids = batch['query']['input_ids'].squeeze(1).to(device)
        positive_input_ids = batch['positive']['input_ids'].squeeze(1).to(device)
        negative_input_ids = batch['negative']['input_ids'].squeeze(1).to(device)

        # Get embeddings for queries, positive, and negative documents
        query_embeds = model(query_input_ids).pooler_output
        pos_embeds = model(positive_input_ids).pooler_output
        neg_embeds = model(negative_input_ids).pooler_output

        # Compute triplet loss
        loss = triplet_loss(query_embeds, pos_embeds, neg_embeds)

        # Backpropagate and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")



def test():
        # Example evaluation code (retrieving relevant documents)
        model.eval()

        # Test on a single query (after training)
        query = "Who has the highest goals in world football?"
        query_input_ids = tokenizer(query, return_tensors='pt', truncation=True, padding='max_length', max_length=128).input_ids.to(device)
        query_embed = model(query_input_ids).pooler_output

        # Now retrieve the top k documents using cosine similarity (or another retrieval method)
        # Ensure you have embeddings for your corpus to compare with the query embeddings
        model.save_pretrained("fine_tuned_colbert_v2")
        tokenizer.save_pretrained("fine_tuned_colbert_v2")


def __main__():
     train()
     test()