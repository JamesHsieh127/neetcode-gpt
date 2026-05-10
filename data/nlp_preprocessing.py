import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combine=positive+negative
        unique_words=set()
        for s in combine:
            words=s.split()
            for w in words:
                unique_words.add(w)
        vocabulary=sorted(list(unique_words))
        wordToId={}
        for i in range(0, len(vocabulary), 1):
            w=vocabulary[i]
            wordToId[w]=i+1
        encode=[]
        for s in combine:
            words=s.split()
            ids=[]
            for w in words:
                ids.append(wordToId[w])
            encode.append(torch.tensor(ids))
        return nn.utils.rnn.pad_sequence(encode, batch_first=True)
        pass
