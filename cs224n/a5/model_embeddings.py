#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab, char_embedding_size=50, drop_prob=0.3):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()
        self.word_embed_size = word_embed_size
        self.char_embedding_size = char_embedding_size
        self.vocab = vocab
        self.char_embedding = nn.Embedding(len(self.vocab.char2id), char_embedding_size)
        self.cnn = CNN(self.char_embedding_size, self.word_embed_size)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(drop_prob)


    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        # Map from x_padded to x_word_emb
        x_emb = self.char_embedding(input)
        x_reshaped = x_emb.permute(0, 1, 3, 2).contiguous()
        sentence_length, batch_size, character_emb_size, m_word = x_reshaped.size()
        x_reshaped = x_reshaped.contiguous().view(-1, character_emb_size, m_word)
        x_conv_out = self.cnn(x_reshaped)
        x_highway = self.highway(x_conv_out)
        x_word_emb = self.dropout(x_highway)
        return x_word_emb.contiguous().view(sentence_length, batch_size, self.word_embed_size)
