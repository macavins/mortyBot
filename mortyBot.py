import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch

# Define the ConversationDataset class
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.examples = tokenizer(data, return_tensors="pt", padding=True, truncation=True)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.examples.items()}

# Load the CSV file into a DataFrame
data = pd.read_csv("testMorty.csv")

# Filter the dataset to include only Morty's lines
morty_lines = data[data["name"] == "Morty"]

# Initialize the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Create the ConversationDataset using only Morty's lines
dataset = ConversationDataset(tokenizer, morty_lines["line"].values.tolist())

# Define your training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(3):  # Example: Train for 3 epochs
    for batch in train_loader:
        inputs = batch["input_ids"].to(device)
        labels = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(inputs, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# Save the trained model
model.save_pretrained("path_to_save_trained_model")

# You can load the trained model later using:
# model = GPT2LMHeadModel.from_pretrained("path_to_save_trained_model")
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the trained model
model = GPT2LMHeadModel.from_pretrained("path_to_save_trained_model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate text
input_text = "Rick: Hey Morty, "
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate a response
output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)
while True:
    user_input = input("You: ")
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Bot:", response)
import discord
from discord.ext import commands

bot = commands.Bot(command_prefix='!')

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.command()
async def talk(ctx, *, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    await ctx.send(response)

bot.run('MTIwNjcyOTU0NDI1ODgyMjE1NA.GGHA2G.o-_QyXN_tZkYlLaKtPB1-_2dd5llmTfjLwQmmQ')
