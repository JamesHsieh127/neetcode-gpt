import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            ix=torch.randint(len(data)-context_length, (batch_size, ))
            x_list=[]
            y_list=[]
            for i in ix:
                idx=i.item()
                x_chunkList=[]
                y_chunkList=[]
                for j in range(0, context_length, 1):
                    x_chunkList.append(data[idx+j])
                    y_chunkList.append(data[idx+j+1])
                x_list.append(torch.stack(x_chunkList))
                y_list.append(torch.stack(y_chunkList))
            x=torch.stack(x_list)
            y=torch.stack(y_list)
            logits=model(x)
            B, T, C=logits.shape
            loss=F.cross_entropy(logits.view(B*T, C), y.view(B*T))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return round(loss.item(), 4)
        pass
