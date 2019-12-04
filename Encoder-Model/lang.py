import json
import spacy
from vocab import Vocab
from spacy.lang.en import English

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:

	def __init__(self, path_to_vocab_files):

		self.tokenizer = English().Defaults.create_tokenizer(English())

		self.word2index = {}
		self.index2word = {}
		self.word2count = {}

		with open(path_to_vocab_files + 'word2index.json') as f:
			self.word2index = json.load(f)

		with open(path_to_vocab_files + 'index2word.json') as f:
			self.index2word = json.load(f)

		with open(path_to_vocab_files + 'word2count.json') as f:
			self.word2count = json.load(f)


	def get_vocab_size(self):
		return len(self.index2word)

	# Takes a string, tokenizes it, and makes it lowercase
	def process_string(self, string):

		tokens =  self.tokenizer(string)
		return  [token.text.lower() for token in tokens]


	# Takes a list of tokens and converts it to a list of indices
	def tokenlist2indexlist(self,tokens):

		index_list = []

		for token in tokens:

			if(token not in self.word2index):
				index_list.append(5)
			else:
				index_list.append(self.word2index[token])

		return index_list

	# Takes a title string and returns a list of tokens
	def get_title_indices(self,title):

		title = [0] + self.tokenlist2indexlist(self.process_string(title)) + [1]
		return torch.unsqueeze(torch.tensor(title, device = device),0)


	# Takes a list of ingredient strings and returns a list of token tensors
	def get_ingredient_indices(self,ingredient_list):

		ingr = []

		for i in ingredient_list:

			ingr.append(torch.tensor([0] + self.tokenlist2indexlist(self.process_string(i)) + [1], device = device))

		return ingr

	# Takes a list of instruction strings and returns a list of token tensors
	def get_instruction_indices(self, instruction_list):

		instr = []

		for i in instruction_list:

			instr.append(torch.tensor([0] + self.tokenlist2indexlist(self.process_string(i)) + [1], device = device))

		return instr



	# Takes a list of tokens and returns a string of decoded tokens
	def indices2string(self, index_list):

		return " ".join([self.index2word[str(index)] for index in index_list])






