#StockTwits RNN


import json
import nltk
import os
import random
import re
import torch
import numpy as np
import StockTwitsFileMaker as stfm
from torch import nn, optim
import torch.nn.functional as F


with open(os.path.join('..', '..', 'data', 'project_6_stocktwits', 'twits.json'), 'r') as f:
    twits = json.load(f)

print(twits['data'][:10])

"""print out the number of twits"""

# TODO Implement 
twit_count = 0
for twit in twits['data']:
    twit_count += 1

print(twit_count)


messages = [twit['message_body'] for twit in twits['data']]
# Since the sentiment scores are discrete, we'll scale the sentiments to 0 to 4 for use in our network
sentiments = [twit['sentiment'] + 2 for twit in twits['data']]
# ### Pre-Processing
nltk.download('wordnet')
from nltk import WordNetLemmatizer

def preprocess(message):
    """
    This function takes a string as input, then performs these operations: 
        - lowercase
        - remove URLs
        - remove ticker symbols 
        - removes punctuation
        - tokenize by splitting the string on whitespace 
        - removes any single character tokens
    
    Parameters
    ----------
        message : The text message to be preprocessed.
        
    Returns
    -------
        tokens: The preprocessed text into tokens.
    """ 
    #TODO: Implement 
    
    # Lowercase the twit message
    text = message.lower()
    
    # Replace URLs with a space in the message
    text = re.sub(r'(http|https)://\S+', ' ', text)
    
    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub(r'\$\w*', ' ', text)
    
    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    text = re.sub(r'\@\w*', ' ', text)

    # Replace everything not a letter with a space
    text = re.sub(r'\W', ' ', text)
    
    # Tokenize by splitting the string on whitespace into a list of words
    tokens = text.split()
    # Lemmatize words using the WordNetLemmatizer. You can ignore any word that is not longer than one character.
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if len(w) > 1]
    
    return tokens


# ### Preprocess All the Twits 

from tqdm import tqdm

tokenize = []
for m in tqdm(messages):
    tokenize.append(preprocess(m))
    
print(tokenize[:10])


# ### Bag of Words

from collections import Counter

"""
Create a vocabulary by using Bag of words
"""

# TODO: Implement 

bow = Counter()
for tokens in tqdm(tokenize):
    bow.update(tokens)


# ### Frequency of Words Appearing in Message

"""
Set the following variables:
    freqs
    low_cutoff
    high_cutoff
    K_most_common
"""

# Dictionary that contains the Frequency of words appearing in messages.
# The key is the token and the value is the frequency of that word in the corpus.
freqs = {k:v/len(tokenize) for k, v in bow.items()}
# Float that is the frequency cutoff. Drop words with a frequency that is lower or equal to this number.
low_cutoff = .000008

# Integer that is the cut off for most common words. Drop words that are the `high_cutoff` most common words.
high_cutoff = 12

# The k most common words in the corpus. Use `high_cutoff` as the k.
K_most_common = bow.most_common(high_cutoff)


filtered_words = [word for word in freqs if (freqs[word] > low_cutoff and word not in K_most_common)]
print(K_most_common)
len(filtered_words)

# A dictionary for the `filtered_words`. The key is the word and value is an id that represents the word.
counts = Counter(filtered_words)
vocab_to_int = sorted(counts, key=counts.get, reverse = True)
vocab = {word: ii for ii, word in enumerate(vocab_to_int, 1)}
# Reverse of the `vocab` dictionary. The key is word id and value is the word. 
id2vocab = sorted(vocab, key=vocab.get, reverse = True)
# tokenized with the words not in `filtered_words` removed.
filtered = [[word for word in messages if word in vocab] for messages in tqdm(tokenize)]


# ### Balancing the classes

balanced = {'messages': [], 'sentiments':[]}

n_neutral = sum(1 for each in sentiments if each == 2)
N_examples = len(sentiments)
keep_prob = (N_examples - n_neutral)/4/n_neutral

for idx, sentiment in enumerate(sentiments):
    message = filtered[idx]
    if len(message) == 0:
        # skip this message because it has length zero
        continue
    elif sentiment != 2 or random.random() < keep_prob:
        balanced['messages'].append(message)
        balanced['sentiments'].append(sentiment) 


n_neutral = sum(1 for each in balanced['sentiments'] if each == 2)
N_examples = len(balanced['sentiments'])
n_neutral/N_examples


# Finally let's convert our tokens into integer ids which we can pass to the network.

token_ids = [[vocab[word] for word in message] for message in balanced['messages']]
sentiments = balanced['sentiments']


# ## Neural Network
# Now we have our vocabulary which means we can transform our tokens into ids, which are then passed to our network. So, let's define the network now!
# 
# Here is a nice diagram showing the network we'd like to build: 
# 
# #### Embed -> RNN -> Dense -> Softmax
# ### Implement the text classifier

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        """
        Initialize the model by setting up the layers.
        
        Parameters
        ----------
            vocab_size : The vocabulary size.
            embed_size : The embedding layer size.
            lstm_size : The LSTM layer size.
            output_size : The output size.
            lstm_layers : The number of LSTM layers.
            dropout : The dropout probability.
        """
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        
        # TODO Implement

        # Setup embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Setup additional layers
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout)
        self.fc = nn.Linear(lstm_size, output_size)
        self.soft = nn.LogSoftmax(dim=0)


    def init_hidden(self, batch_size):
        """ 
        Initializes hidden state
        
        Parameters
        ----------
            batch_size : The size of batches.
        
        Returns
        -------
            hidden_state
            
        """
        
        # TODO Implement 
        
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
        
        return hidden


    def forward(self, nn_input, hidden_state):
        """
        Perform a forward pass of our model on nn_input.
        
        Parameters
        ----------
            nn_input : The batch of input to the NN.
            hidden_state : The LSTM hidden state.

        Returns
        -------
            logps: log softmax output
            hidden_state: The new hidden state.

        """
        
        # TODO Implement 
        batch_size = nn_input.size(1)
        
        #embeddings and LSTM OUT
        embeds = self.embedding(nn_input)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        
        #stack up LSTM Outputs
        lstm_out = lstm_out[-1,:,:]
        
        #dropout and fully connect layer
        out = self.fc(lstm_out)
        
        #softmax function
        logps = self.soft(out)

        return logps, hidden_state

model = TextClassifier(len(vocab), 10, 6, 5, dropout=0.1, lstm_layers=2)
model.embedding.weight.data.uniform_(-1, 1)
input = torch.randint(0, 1000, (5, 4), dtype=torch.int64)
hidden = model.init_hidden(4)

logps, _ = model.forward(input, hidden)
print(logps)


# Training
## DataLoaders and Batching


def dataloader(messages, labels, sequence_length=30, batch_size=32, shuffle=False):
    """ 
    Build a dataloader.
    """
    if shuffle:
        indices = list(range(len(messages)))
        random.shuffle(indices)
        messages = [messages[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    total_sequences = len(messages)

    for ii in range(0, total_sequences, batch_size):
        batch_messages = messages[ii: ii+batch_size]
        
        # First initialize a tensor of all zeros
        batch = torch.zeros((sequence_length, len(batch_messages)), dtype=torch.int64)
        for batch_num, tokens in enumerate(batch_messages):
            token_tensor = torch.tensor(tokens)
            # Left pad!
            start_idx = max(sequence_length - len(token_tensor), 0)
            batch[start_idx:, batch_num] = token_tensor[:sequence_length]
        
        label_tensor = torch.tensor(labels[ii: ii+len(batch_messages)])
        
        yield batch, label_tensor


# ### Training and  Validation
# With our data in nice shape, we'll split it into training and validation sets.

"""
Split data into training and validation datasets. Use an appropriate split size.
The features are the `token_ids` and the labels are the `sentiments`.
"""   

split = int(len(token_ids)*0.8)
train_features = token_ids[:split]
valid_features = token_ids[split:]
train_labels = sentiments[:split]
valid_labels = sentiments[split:]


text_batch, labels = next(iter(dataloader(train_features, train_labels, sequence_length=20, batch_size=64)))
model = TextClassifier(len(vocab)+1, 200, 128, 5, dropout=0.)
hidden = model.init_hidden(64)
logps, hidden = model.forward(text_batch, hidden)


# ### Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextClassifier(len(vocab)+1, 1024, 512, 5, lstm_layers=2, dropout=0.2)
model.embedding.weight.data.uniform_(-1, 1)
model.to(device)

"""
Train your model with dropout. Make sure to clip your gradients.
Print the training loss, validation loss, and validation accuracy for every 100 steps.
"""

epochs = 4
batch_size = 128
learning_rate = .003

print_every = 100
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

for epoch in range(epochs):
    print('Starting epoch {}'.format(epoch + 1))
    
    steps = 0
    for text_batch, labels in dataloader(
            train_features, train_labels, batch_size=batch_size, sequence_length=20, shuffle=True):
        steps += 1
        hidden = model.init_hidden(labels.shape[0])
       
        # Set Device
        text_batch, labels = text_batch.to(device), labels.to(device)
        for each in hidden:
            each.to(device)
       
        # TODO Implement: Train Model
        model.zero_grad()
        output, hidden = model(text_batch, hidden)
       
        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
       
       
        if steps % print_every == 0:
            model.eval()
       
            val_accuracy = 0
           
            for text_batch, labels in dataloader(valid_features, valid_labels, batch_size=batch_size, sequence_length=20, shuffle=True):
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
               
               
                # Get validation loss
                val_h = model.init_hidden(labels.shape[0])
                val_losses = []
               
               
                val_h = tuple([each.data for each in val_h])
 
                text_batch, labels = text_batch.to(device), labels.to(device)
                for each in val_h:
                    each.to(device)
               
                output, val_h = model(text_batch, val_h)
                val_loss = criterion(output, labels)
 
                val_losses.append(val_loss.item())
               
                topval, topclass = torch.exp(output).topk(1)
                val_accuracy += (torch.sum(topclass.squeeze() == labels))
                               
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Step: {}...".format(steps),
                  "Loss: {:.4f}...".format(loss.item()),
                  "Val Loss: {:.4f}".format(np.mean(val_losses)),
                  "Val Accuracy: {:.4f}".format(val_accuracy.item() / len(valid_labels)))
           
            model.train()


# ## Making Predictions
# ### Prediction 
# Okay, now that you have a trained model, try it on some new twits and see if it works appropriately. Remember that for any new text, you'll need to preprocess it first before passing it to the network.


def predict(text, model, vocab):
    """ 
    Make a prediction on a single sentence.

    Parameters
    ----------
        text : The string to make a prediction on.
        model : The model to use for making the prediction.
        vocab : Dictionary for word to word ids. The key is the word and the value is the word id.

    Returns
    -------
        pred : Prediction vector
    """    
    
    # TODO Implement
    
    tokens = preprocess(text)
    
    # Filter non-vocab words
    tokens = [i for i in tokens if i in vocab]
    # Convert words to ids
    tokens = [vocab[i] for i in tokens]
    # Adding a batch dimension
    text_input = torch.from_numpy(np.asarray(tokens)).view(-1,1)
    # Get the NN output
    hidden = model.init_hidden(1)
    logps, _ = model.forward(text_input, hidden)
    # Take the exponent of the NN output to get a range of 0 to 1 for each label.
    pred = torch.exp(logps)
    
    return pred


# In[97]:


text = "Google is working on self driving cars, I'm bullish on $goog"
model.eval()
model.to("cpu")
predict(text, model, vocab)
# ## Testing
# ### Load the Data 

# In[98]:


with open(os.path.join('..', '..', 'data', 'project_6_stocktwits', 'test_twits.json'), 'r') as f:
    test_data = json.load(f)


# ### Twit Stream
#To be updated for an actual stream
def twit_stream():
    for twit in test_data['data']:
        yield twit

next(twit_stream())


# Using the `prediction` function, let's apply it to a stream of twits.

def score_twits(stream, model, vocab, universe):
    """ 
    Given a stream of twits and a universe of tickers, return sentiment scores for tickers in the universe.
    """
    for twit in stream:

        # Get the message text
        text = twit['message_body']
        symbols = re.findall('\$[A-Z]{2,4}', text)
        score = predict(text, model, vocab)

        for symbol in symbols:
            if symbol in universe:
                yield {'symbol': symbol, 'score': score, 'timestamp': twit['timestamp']}




universe = {'$BBRY', '$AAPL', '$AMZN', '$BABA', '$YHOO', '$LQMT', '$FB', '$GOOG', '$BBBY', '$JNUG', '$SBUX', '$MU'}
score_stream = score_twits(twit_stream(), model, vocab, universe)

next(score_stream)

