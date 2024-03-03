import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the bag_of_words function
def bag_of_words(sentence, words):
    # Tokenize the sentence
    sentence_words = word_tokenize(sentence)
    # Stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Load intents from JSON file
with open('.\chatbot\intents.json', 'r') as file:
    intents = json.load(file)

# Tokenize patterns
nltk.download('punkt')
all_words = []
tags = []
patterns = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words
        w = word_tokenize(pattern)
        all_words.extend(w)
        patterns.append((w, intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# Preprocess words
stemmer = PorterStemmer()
ignore_words = ['?', '.', '!']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Define X_train (input) and Y_train (output) data
X_train = []
Y_train = []
for (pattern_sentence, tag) in patterns:
    bag = bag_of_words(' '.join(pattern_sentence), all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Define the PyTorch model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Define training parameters
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# Initialize the model, loss function, and optimizer
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    inputs = torch.from_numpy(X_train).float()
    targets = torch.from_numpy(Y_train)

    # Forward pass
    outputs = model(inputs)
    targets = torch.from_numpy(Y_train).long()

# Use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'chatbot_model.pth')


# Load the trained model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('chatbot_model.pth'))
model.eval()

def process_user_input(user_input_1):
# Initialize the chatbot
    print("Chatbot is ready to chat. Type 'quit' to exit.")
    while True:
        # Get user input
        user_input = user_input_1
        if user_input.lower() == 'quit':
            return "Bye bye....."
            break

        # Tokenize and preprocess user input
        input_bag = bag_of_words(user_input, all_words)
        input_bag = torch.from_numpy(input_bag).unsqueeze(0).float()

        # Forward pass through the model
        output = model(input_bag)

        # Get the predicted tag
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        # Select a response from intents based on the predicted tag
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])