import json

import spacy
from spacy.lang.en import English


class Vocab:
    """Assigns a unique id to each word in the corpus of recipes"""
    
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'SOI', 3:'EOI', 4:'unk'}
        self.n_words = 5  # Count SOS and EOS
        self.lang = English()
        self.tokenizer = self.lang.Defaults.create_tokenizer(self.lang)
    
    
    def add_instruction(self, instruction):
        """Takes a single instruction and adds it to the vocab"""
        self._tokenize(instruction)
            
            
    def add_ingredient(self, ingredient):
        """Takes a single ingredient and adds it to the vocab"""
        self._tokenize(ingredient)
        
        
    def add_title(self, title):
        """Takes a title of a recipe and adds it to the vocab"""
        self._tokenize(title)
            

    def _tokenize(self, string):
        """Separates a string into tokens and adds each token to the vocab"""
        tokens = self.tokenizer(string)
        for token in tokens:
            self._add_word(token.text.lower())
    

    def _add_word(self, word):
        """Processes a word and inserts into the vocab"""
        if  word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            
        else:
            self.word2count[word] += 1

            
    def dump(self):
        """Dumps the vocab to disk"""
        with open('word2index.json', 'w') as f:
            json.dump(self.word2index, f)

        with open('index2word.json', 'w') as f:
            json.dump(self.index2word, f)
            
        with open('word2count.json', 'w') as f:
            json.dump(self.word2count, f)
        

def create_vocab():
    """Creates a vocab based on the recipe train data"""
    vocab = Vocab()
    for j in range(9):
        file_path = f'../data/train{j}.json'
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            print('Loaded File: ', file_path)
        
        for i, recipe in enumerate(data):
            print('Recipe Count: ', str(i), '/', str(len(data)), end='\r')
            vocab.add_title(recipe["title"])
            for ingredient in recipe["ingredients"]:
                vocab.add_ingredient(ingredient)
            for instruction in recipe["instructions"]:
                vocab.add_instruction(instruction)
        
        print('Done: ', j)
        vocab.dump()
    print("Finished Vocab")
        

def load_vocab():
    """Loads vocab from dump data"""
    vocab = Vocab()
    self.n_words = len(word2index)
    
    with open('word2index.json', 'r') as f:
        vocab.word2index = json.load(f)

    with open('index2word.json', 'r') as f:
        vocab.index2word = json.load(f)

    with open('word2count.json', 'r') as f:
        vocab.word2count = json.load(f)
    
    return vocab
    


if __name__ == "__main__":
    create_vocab()
