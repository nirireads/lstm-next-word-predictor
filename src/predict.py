import torch
from src.dataset import vocab, text_to_indices, max_len
from src.model import model
from nltk.tokenize import word_tokenize


def predict_top_k(input, model=model, vocab=vocab, k=5):
    model.load_state_dict(torch.load('models/next_word_model.pth'))
    # preprocess input
    tokens = word_tokenize(input.lower())
    indices = text_to_indices(tokens, vocab)

    #truncate if input is longer than max_len
    if len(indices) > max_len:
        indices = indices[-max_len:]
    
    #pad if input is shorter than max_len
    padded = [0] * (max_len - len(indices)) + indices
    input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        topk = torch.topk(probs, k) #gives 3 highest probabilities

    
    inv_vocab = {new_value: new_key for new_key, new_value in vocab.items()}
    topk_words = [inv_vocab[idx.item()] for idx in topk.indices[0]]

    return topk_words



# dummy_input = "A faint music played"
# top_words = predict_top_k(dummy_input)
# print("Top predicted words:", top_words)