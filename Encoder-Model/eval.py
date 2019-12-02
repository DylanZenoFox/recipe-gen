from models.title_encoder import TitleEncoder
from models.ingredients_encoder import IngredientsEncoder
from models.instructions_decoder import InstructionsDecoder, EndInstructionsClassifier
from vocab import Vocab

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import json
import spacy
from spacy.lang.en import English

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Solver():

	def __init__(self, load_from_path = None, save_to_path = None, save_frequency = 100):


		self.load_from_path = load_from_path

		# TOKENIZER 

		self.tokenizer = English().Defaults.create_tokenizer(English())

		# VOCABULARY DICTS

		self.word2index = {}
		self.index2word = {}
		self.word2count = {}

		with open('word2index.json') as f:
			self.word2index = json.load(f)

		with open('index2word.json') as f:
			self.index2word = json.load(f)

		with open('word2count.json') as f:
			self.word2count = json.load(f)

		# HYPERPARAMETERS 

		self.vocab_size = len(self.index2word)

		self.word_embedding_dim = 100

		self.title_hidden_dim = 200

		self.ingredients_hidden_dim = 200
		self.single_ingr_dim = 150
		self.ingredients_bidirectional = False

		self.instructions_hidden_dim = self.title_hidden_dim + self.ingredients_hidden_dim
		self.single_instruction_dim = 200
		self.max_instr_length = 25

		self.max_num_instructions = 10

		self.teacher_forcing_ratio = 1.0

		self.binary_MLP_hidden_dim = 50

		self.learning_rate = 0.01

		# MODELS

		#self.instr_hidden2input = nn.Linear(self.instructions_hidden_dim, self.single_instruction_dim).to(device)

		self.title_encoder = TitleEncoder(embedding_dim= self.word_embedding_dim, hidden_dim = self.title_hidden_dim, vocab_size = self.vocab_size).to(device)

		self.ingredients_encoder = IngredientsEncoder(ingr_embed_dim = self.single_ingr_dim, word_embed_dim = self.word_embedding_dim, 
			hidden_dim = self.ingredients_hidden_dim, vocab_size = self.vocab_size, outer_bidirectional = self.ingredients_bidirectional).to(device)

		self.instructions_decoder = InstructionsDecoder(instr_hidden_dim = self.single_instruction_dim, word_embedding_dim = self.word_embedding_dim,
			rec_hidden_dim = self.instructions_hidden_dim, vocab_size = self.vocab_size, max_instr_length = self.max_instr_length, 
			teacher_forcing_ratio = self.teacher_forcing_ratio).to(device)

		self.end_instructions_classifier = EndInstructionsClassifier(instr_embed_dim = self.instructions_hidden_dim, hidden_dim = self.binary_MLP_hidden_dim).to(device)


		# OPTIMIZERS

		self.title_encoder_optimizer = optim.Adam(self.title_encoder.parameters())
		self.ingredients_encoder_optimizer = optim.Adam(self.ingredients_encoder.parameters())

		self.instructions_decoder_optimizer = optim.Adam(self.instructions_decoder.parameters())
		self.end_instructions_classifier_optimizer = optim.Adam(self.end_instructions_classifier.parameters())
		#self.instr_hidden2input_optimizer = optim.Adam(self.instr_hidden2input.parameters())

		# LOSS FUNCTIONS

		self.decoder_criterion = nn.NLLLoss()
		self.end_instr_criterion = nn.NLLLoss()

		# LOAD PARAMETERS IF POSSIBLE

		if(self.load_from_path is not None):

			self.load_model(self.load_from_path)


	def evaluate_example(self,title, ingredients, target_instructions, length):

		# Encode title and ingredients
		title_outputs, encoded_title = self.title_encoder(title)
		ingr_outputs, encoded_ingr = self.ingredients_encoder(ingredients)

		# Concatenate to get first hidden layer of decoder
		decoder_hidden = torch.cat([encoded_title,encoded_ingr], dim = 2)

		#decoder_input = self.instr_hidden2input(decoder_hidden)
		decoder_input = torch.zeros(self.single_instruction_dim, device = device)
		decoder_input = torch.unsqueeze(decoder_input,0)
		decoder_input = torch.unsqueeze(decoder_input,0)

		instructions = []

		for i in range(length):

			decoder_output, decoder_hidden, decoded_instruction, loss = self.instructions_decoder(decoder_input, decoder_hidden, 
				self.decoder_criterion, targets = None)

			instructions.append(self.indices2string(decoded_instruction))

			end_instructions = self.end_instructions_classifier(decoder_hidden[0])

			# Print the decoded instruction next to the actual instruction
			print(str(i) + " Decoded: " + self.indices2string(decoded_instruction))
			print(str(i) + " Actual: " + self.indices2string(target_instructions[i].tolist()))

			decoder_input = decoder_output.detach()

		return instructions



	def evalIters(self):

		iters = 0


		for i in range(5):

			with open('../data/test' + str(i) + ".json") as f:

				recipe_data = json.load(f)

				for recipe in recipe_data:

					print("Recipe 10:")
					
					title = recipe['title']

					ingredients = recipe['ingredients']

					instructions = recipe['instructions']

					print("Recipe " + iters + " (" + title + "): ")


					title_input = self.get_title_indices(title)
					ingredients_input = self.get_ingredient_indices(ingredients)
					instructions_input = self.get_instruction_indices(instructions)

					decoded_instructions = self.evaluate_example(title_input, ingredients_input, instructions_input)

					print("Decoded Instructions: ")

					for i in range(len(decoded_instructions)):
						print(str(i) + ": " + decoded_instructions[i])

					print("Ground Truth Instructions: ")

					for i in range(len(instructions)):
						print(str(i) + ": " + instructions[i])

					print("------------------------------------------------")




					iters+=1



	def process_string(self, string):

		tokens =  self.tokenizer(string)

		return  [token.text.lower() for token in tokens]

	def tokenlist2indexlist(self,tokens):

		index_list = []

		for token in tokens:

			if(token not in self.word2index):
				index_list.append(4)
			else:
				index_list.append(self.word2index[token])


		return index_list

	def get_title_indices(self,title):

		title = [0] + self.tokenlist2indexlist(self.process_string(title)) + [1]
		return torch.unsqueeze(torch.tensor(title, device = device),0)

	def get_ingredient_indices(self,ingredient_list):

		ingr = []

		for i in ingredient_list:

			ingr.append(torch.tensor([0] + self.tokenlist2indexlist(self.process_string(i)) + [1], device = device))

		return ingr

	def get_instruction_indices(self, instruction_list):

		instr = []

		for i in instruction_list:

			instr.append(torch.tensor([0] + self.tokenlist2indexlist(self.process_string(i)) + [1], device = device))

		return instr


	def indices2string(self, index_list):

		return " ".join([self.index2word[str(index)] for index in index_list])


	def load_model(self,model_params_path):

		checkpoint = torch.load(model_params_path,map_location=device)

		self.title_encoder.load_state_dict(checkpoint['title_encoder_state_dict'])
		self.ingredients_encoder.load_state_dict(checkpoint['ingredients_encoder_state_dict'])
		self.instructions_decoder.load_state_dict(checkpoint['instructions_decoder_state_dict'])
		self.end_instructions_classifier.load_state_dict(checkpoint['end_instructions_classifier_state_dict'])



if(__name__ == '__main__'):

	test = Solver(load_from_path = './model_params/train_checkpoint2')

	#loss_per_instr = test.train_example(test_title, test_ingredients, test_targets)
	#print(loss_per_instr)

	test.evalIters()

	#print(test.get_title_indices("World famous chicken"))
