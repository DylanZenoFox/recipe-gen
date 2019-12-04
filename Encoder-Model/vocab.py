import json
import spacy
from spacy.lang.en import English


class Vocab:
    """Assigns a unique id to each word in the corpus of recipes"""
    
    def __init__(self, path_to_vocab_files):
        self.path_to_vocab_files = path_to_vocab_files
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: '<S>', 1: '</S>', 2: '<PAD>', 3: '<I>', 4:'</I>', 5:'<UNK>'}
        self.n_words = 6  # Count SOS and EOS
        self.lang = English()
        self.tokenizer = self.lang.Defaults.create_tokenizer(self.lang)

    def get_tokenizer(self):
        return self.tokenizer
    
    
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
        with open(self.path_to_vocab_files + '/word2index.json', 'w') as f:
            json.dump(self.word2index, f)

        with open(self.path_to_vocab_files + '/index2word.json', 'w') as f:
            json.dump(self.index2word, f)
            
        with open(self.path_to_vocab_files + '/word2count.json', 'w') as f:
            json.dump(self.word2count, f)
        

def create_vocab(vocab_file_loc):
    """Creates a vocab based on the recipe train data"""
    vocab = Vocab(vocab_file_loc)
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
        

def load_vocab(vocab_file_loc):
    """Loads vocab from dump data"""
    vocab = Vocab(vocab_file_loc)
    self.n_words = len(word2index)
    
    with open(vocab_file_loc + '/word2index.json', 'r') as f:
        vocab.word2index = json.load(f)

    with open(vocab_file_loc + '/index2word.json', 'r') as f:
        vocab.index2word = json.load(f)

    with open(vocab_file_loc + '/word2count.json', 'r') as f:
        vocab.word2count = json.load(f)
    
    return vocab
    


if __name__ == "__main__":


    vocab_file_loc = './vocab_files'


    create_vocab(vocab_file_loc)
