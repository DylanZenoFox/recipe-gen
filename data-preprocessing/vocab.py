import spacy
import json

nlp = spacy.load("en_core_web_sm")

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'SOI', 3:'EOI', 4:'unk'}
        self.n_words = 5  # Count SOS and EOS
    
    def add_instruction(self, instruction):
        """Takes a single instruction and adds it to the vocab"""
        self.tokenize(instruction)
            
    def add_ingredient(self, ingredient):
        """Takes a single ingredient and adds it to the vocab"""
        self.tokenize(ingredient)
        
    def add_title(self, title):
        self.tokenize(title)
            
    def tokenize(self, list_of_words):
        tokens = nlp(list_of_words)
        for token in tokens:
            self.add_word(token.text.lower())
    
    def add_word(self, word):
        if  word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            print('Words: ', self.n_words, end='\r')
        else:
            self.word2count[word] += 1
            
    def save_dictionaries(self):
        with open('word2index.json', 'w') as f:
            json.dump(self.word2index, f)

        with open('index2word.json', 'w') as f:
            json.dump(self.index2word, f)
            
        with open('word2count.json', 'w') as f:
            json.dump(self.word2count, f)

        print('DONE')

def create_vocab():
    vocab = Vocab()
    for j in range(10):
        file = f'../data/train{j}.json'
        
        with open(file, 'r') as f:
            file = json.load(f)
            print('Loaded File: ', file)
        
        for recipe in file:
            vocab.add_title(recipe["title"])
            for ingredient in recipe["ingredients"]:
                vocab.add_ingredient(ingredient)
            for instruction in recipe["instructions"]:
                vocab.add_instruction(instruction)
            
        vocab.save_dictionaries()