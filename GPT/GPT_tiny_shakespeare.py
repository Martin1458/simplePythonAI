import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

#
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

print(device)

# Read our shakespeare dataset
with open(r"GPT/datasets/tinyshakespeare.txt", "r", encoding="UTF-8") as f:
    text = f.read()


# Print list of all the chars and symbols, that are in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create tokenization functions to convert all the characters and symbols from the dataset into something that GPT can process

# Make a character to integer and integer to character dictionary
char_to_int = {char: index for index, char in enumerate(chars)}
int_to_char = {index: char for index, char in enumerate(chars)}

# Function to convert a string to a list of integers
def encoder(s):
    return [char_to_int[c] for c in s]

# Function to convert a list of integers to a string
def decoder(l):
    return ''.join([int_to_char[i] for i in l])

# Encode the whole dataset, so that the model can read it

encoded_text = encoder(text)

# Storing the encoded text in a torch.tensor object

data = torch.tensor(encoded_text, dtype=torch.long, device=device)


# Split the data into training and testing sets
test_size = int(0.1*len(data))

train_data = data[:test_size]
test_data = data[test_size:]

batch_size = 4 
block_size = 8


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #pdb.set_trace()
            logits, loss = self(idx)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)
xb = xb.to(device)
yb = yb.to(device)
logits, loss = m(xb, yb)
print("\nNew prediction from our model if the user input is a new line character:", end="")
print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()), end="\n\n")
# Lets optimize and train the model

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# This codeblock of training the model can be executed multiple times to train the model more

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nNew prediction from our model if the user input is a new line character:", end="")
print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))


