import torch.nn as nn
import torch
from src.model import model
from src.dataset import dataloader

EPOCHS = 50
LR = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# train loop
for epoch in range(EPOCHS):
    total_loss = 0

    for input_seq, target in dataloader:
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')

    #save the model
    torch.save(model.state_dict(), 'models/next_word_model.pth')