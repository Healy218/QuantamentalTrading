import pandas as pd
import numpy as np
import Twitter
import TweetFileMaker as tfm
import json
import nltk
import os
import random
import re
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F


with open(os.path.join('..', '..', 'data', 'QuantamentalTools', 'tweets.json'), 'r') as f:
    tweetts = json.load(f)

print(tweets['data'][:10])

"""print out the number of twits"""

# TODO Implement 
tweet_count = 0
for tweet in tweets['data']:
    tweet_count += 1

print(tweet_count)

