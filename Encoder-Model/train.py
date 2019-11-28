from models.title_encoder import TitleEncoder
from models.ingredients_encoder import IngredientsEncoder
from models.instructions_decoder import InstructionsDecoder, EndInstructionsClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


class Solver():

	def __init__(self):

		# HYPERPARAMETERS 

		self.vocab_size = 10

		self.word_embedding_dim = 5

		self.title_hidden_dim = 20

		self.ingredients_hidden_dim = 20
		self.single_ingr_dim = 10
		self.ingredients_bidirectional = False

		self.instructions_hidden_dim = self.title_hidden_dim + self.ingredients_hidden_dim
		self.single_instruction_dim = 20
		self.max_instr_length = 20

		self.max_num_instructions = 10

		self.teacher_forcing_ratio = 0.5

		self.binary_MLP_hidden_dim = 20

		self.learning_rate = 0.01

		# MODELS

		self.instr_hidden2input = nn.Linear(self.instructions_hidden_dim, self.single_instruction_dim)

		self.title_encoder = TitleEncoder(embedding_dim= self.word_embedding_dim, hidden_dim = self.title_hidden_dim, vocab_size = self.vocab_size)

		self.ingredients_encoder = IngredientsEncoder(ingr_embed_dim = self.single_ingr_dim, word_embed_dim = self.word_embedding_dim, 
			hidden_dim = self.ingredients_hidden_dim, vocab_size = self.vocab_size, outer_bidirectional = self.ingredients_bidirectional)

		self.instructions_decoder = InstructionsDecoder(instr_hidden_dim = self.single_instruction_dim, word_embedding_dim = self.word_embedding_dim,
			rec_hidden_dim = self.instructions_hidden_dim, vocab_size = self.vocab_size, max_instr_length = self.max_instr_length, 
			teacher_forcing_ratio = self.teacher_forcing_ratio)

		self.end_instructions_classifier = EndInstructionsClassifier(instr_embed_dim = self.instructions_hidden_dim, hidden_dim = self.binary_MLP_hidden_dim)

		# OPTIMIZERS

		self.title_encoder_optimizer = optim.SGD(self.title_encoder.parameters(), lr=self.learning_rate)
		self.ingredients_encoder_optimizer = optim.SGD(self.ingredients_encoder.parameters(), lr=self.learning_rate)

		self.instructions_decoder_optimizer = optim.SGD(self.instructions_decoder.parameters(), lr=self.learning_rate)
		self.end_instructions_classifier_optimizer = optim.SGD(self.end_instructions_classifier.parameters(), lr=self.learning_rate)

		# LOSS FUNCTIONS

		self.decoder_criterion = nn.NLLLoss()
		self.end_instr_criterion = nn.NLLLoss()


	#  
	# Train a Single Example
	#
	# Parameters:
	#				
	# Input:
	#		title: tensor of shape (1, seq_len) representing the title of the recipe
	#		ingredients: list of ingredient tensors of shape (num_ingredients, num_words)
	#		target_instructions: list of instruction tensors of shape (num_instructions,num_words)

	def train_example(self, title, ingredients, target_instructions):

		self.title_encoder_optimizer.zero_grad()
		self.ingredients_encoder_optimizer.zero_grad()
		self.instructions_decoder_optimizer.zero_grad()
		self.end_instructions_classifier_optimizer.zero_grad()

		# Encode title and ingredients
		title_outputs, encoded_title = self.title_encoder(title)
		ingr_outputs, encoded_ingr = self.ingredients_encoder(ingredients)


		# Concatenate to get first hidden layer of decoder
		decoder_hidden = torch.cat([encoded_title,encoded_ingr], dim = 2)

		decoder_input = self.instr_hidden2input(decoder_hidden).detach()

		total_loss = 0

		for i in range(len(target_instructions)):

			decoder_output, decoder_hidden, decoded_instruction, loss = self.instructions_decoder(decoder_input, decoder_hidden, 
				self.decoder_criterion, targets = target_instructions[i])

			#print(loss)
			total_loss += loss

			end_instructions = self.end_instructions_classifier(decoder_hidden[0])

			# 1 means that instructions should end, 0 means that instructions should continue
			if(i == len(target_instructions)-1):
				total_loss += self.end_instr_criterion(end_instructions, torch.tensor([1]))
			else:
				total_loss += self.end_instr_criterion(end_instructions, torch.tensor([0]))

			decoder_input = decoder_output.detach()

		total_loss.backward()

		self.title_encoder_optimizer.step()
		self.ingredients_encoder_optimizer.step()
		self.instructions_decoder_optimizer.step()
		self.end_instructions_classifier_optimizer.step()

		return total_loss.item() / len(target_instructions)




	def trainIters(self, n_iters):

		pass





test = Solver()

test_title = torch.tensor([[0,3,4,6,7,5,1]])

test_ingredients = [
						torch.tensor([0,3,5,4,5,7,1]),
						torch.tensor([0,7,2,4,1]),
						torch.tensor([0,4,2,2,2,2,2,2,1])
					]

test_targets = [

				torch.tensor([0,2,3,4,5,6,7,1]),
				torch.tensor([0,6,7,8,8,1])

				]


loss_per_instr = test.train_example(test_title, test_ingredients, test_targets)
print(loss_per_instr)






