from models.encoder_decoder import EncoderDecoder
from lang import Lang

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Solver():

	def __init__(self, load_from_path = None, save_to_path = None, save_frequency = 100):


		self.load_from_path = load_from_path
		self.save_to_path = save_to_path
		self.save_frequency = save_frequency

		# LANG CLASS

		self.lang = Lang(path_to_vocab_files= './vocab_files/')

		# HYPERPARAMETERS 

		self.vocab_size = self.lang.get_vocab_size()

		self.word_embedding_dim = 128

		self.title_hidden_dim = 256
		self.title_bidirectional = True

		self.ingr_list_hidden_dim = 256
		self.single_ingr_hidden_dim = 192

		self.ingredients_outer_bidirectional = True
		self.ingredients_inner_bidirectional = True

		self.single_instr_hidden_dim = 256
		self.max_instr_length = 25

		self.max_num_instructions = 15

		self.single_instr_tf_ratio = 0.5
		self.instr_list_tf_ratio = 0.5

		self.end_instr_hidden_dim = 128

		self.learning_rate = 0.01

		# MODELS

		self.encoder_decoder = EncoderDecoder(vocab_size = self.vocab_size, word_embedding_dim = self.word_embedding_dim, title_hidden_dim = self.title_hidden_dim, ingr_list_hidden_dim = self.ingr_list_hidden_dim,
			single_ingr_hidden_dim = self.single_ingr_hidden_dim, single_instr_hidden_dim = self.single_instr_hidden_dim, end_instr_hidden_dim = self.end_instr_hidden_dim, max_num_instr = self.max_num_instructions,
			max_instr_length = self.max_instr_length, single_instr_tf_ratio = self.single_instr_tf_ratio, instr_list_tf_ratio = self.instr_list_tf_ratio, title_bidirectional = self.title_bidirectional, 
			ingr_outer_bidirectional = self.ingredients_inner_bidirectional, ingr_inner_bidirectional = self.ingredients_inner_bidirectional).to(device)

		# OPTIMIZER

		self.optimizer = optim.Adam(self.encoder_decoder.parameters())

		# LOSS FUNCTIONS

		self.word_criterion = nn.NLLLoss()
		self.end_instr_criterion = nn.NLLLoss()

		# LOAD PARAMETERS IF POSSIBLE

		if(self.load_from_path is not None):
			self.load_model(self.load_from_path)


		print("====================INITIALIZING MODEL====================")
		print("\n")
		print("=====================HYPERPARAMETERS======================")
		print("")
		print("---------------------Model Statistics---------------------")
		print("")

		print("Number of Model Parameters: " + str(sum(p.numel() for p in self.encoder_decoder.parameters())))
		print("Vocab Size: " + str(self.vocab_size))
		print("")
		print("----------------------Embedding Sizes---------------------")
		print("")
		print("Word Embedding Dimension: " + str(self.word_embedding_dim))
		print("Title Embedding Dimension: " + str(self.title_hidden_dim))
		print("Single Ingredient Embedding Dimension: " + str(self.single_ingr_hidden_dim))
		print("Ingredients List Embedding Dimension: " +str(self.ingr_list_hidden_dim))
		print("Instructions List Embedding Dimension: " + str(self.ingr_list_hidden_dim + self.title_hidden_dim))
		print("Single Instruction Embedding Dimension: "  + str(self.single_instr_hidden_dim))
		print("")
		print("-----------------------Bidirectional----------------------")
		print("")
		print("Title Bidirectional Encoder: " + str(self.title_bidirectional))
		print("Single Ingredient Bidirectional Encoder: " + str(self.ingredients_inner_bidirectional))
		print("Ingredients List Bidirectional Encoder: " + str(self.ingredients_outer_bidirectional))
		print("")
		print("----------------------Teacher Forcing---------------------")
		print("")
		print("Instruction List Teacher Forcing Ratio: " + str(self.instr_list_tf_ratio))
		print("Single Instruction Teacher Forcing Ratio: " + str(self.single_instr_tf_ratio))
		print("")
		print("------------------Generation Limitations------------------")
		print("")
		print("Max Instruction Length: " + str(self.max_instr_length))
		print("Max Number of Instructions: " + str(self.max_num_instructions))
		print("")
		print("------------------------Attention-------------------------")
		print("")
		print("TODO")
		print("")
		print("------------------------Load/Save-------------------------")
		print("")
		print("Loading Checkpoint From: " + str(self.load_from_path))
		print("Saving Checkpoints To: " + str(self.save_to_path))
		print("Save Frequency: " + str(self.save_frequency))
		print("")
		print("---------------------Miscellaneous------------------------")
		print("")
		print("Hidden Dimension of EoI Classifier: " + str(self.end_instr_hidden_dim))
		print("")
		print("==========================================================")
		print("")







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

		self.optimizer.zero_grad()

		instructions, word_loss, end_instr_loss = self.encoder_decoder(title, ingredients, self.word_criterion, self.end_instr_criterion, target_instructions)

		print("Total Word Loss: " + str(word_loss))
		print("Total End_Instr Loss: " + str(end_instr_loss))

		total_loss = word_loss + end_instr_loss

		print("Forward Pass Complete")

		total_loss.backward()

		print("Computed Gradients")

		self.optimizer.step()

		print("Updated Gradients")

		return (total_loss.item() / len(target_instructions)) , instructions




	def trainIters(self, print_every = 20, num_epochs = 1, num_train_files = 10):

		iters = 0
		total_loss = 0


		for i in range(num_train_files):

			with open('../data/train' + str(i) + ".json") as f:

				recipe_data = json.load(f)

				for recipe in recipe_data:
					
					title = recipe['title']

					ingredients = recipe['ingredients']

					instructions = recipe['instructions']

					print("Recipe : " + str(iters) + " (" + title + ") ")


					title_input = self.lang.get_title_indices(title)
					ingredients_input = self.lang.get_ingredient_indices(ingredients)
					instructions_input = self.lang.get_instruction_indices(instructions)

					loss, decoded_instructions = self.train_example(title_input, ingredients_input, instructions_input)

					total_loss += loss


					iters += 1
					if(iters % print_every == 0):

						print("ACTUAL INSTRUCTIONS:")

						for i in range(len(instructions)):
							print(str(i) + ": " + instructions[i])

						print("DECODED INSTRUCTIONS:")

						for i in range(len(decoded_instructions)):
							print(str(i) + ": " + self.lang.indices2string(decoded_instructions[i]))

						print(" ")

						print("Average per Instruction Loss for the Past " + str(print_every) + " Recipes: " + str(total_loss/print_every))
						total_loss = 0 

					print("==============================================================================")


					if((self.save_to_path is not None) and (iters % self.save_frequency == 0)):

						self.save_model(self.save_to_path)


	def save_model(self, model_params_path):

		torch.save({

					'model_state_dict':
					self.encoder_decoder.state_dict(),
			}, model_params_path)




	def load_model(self,model_params_path):

		checkpoint = torch.load(model_params_path)

		self.encoder_decoder.load_state_dict(checkpoint['model_state_dict'])



if(__name__ == '__main__'):

	test = Solver(load_from_path = None, save_to_path = './model_params/updated_train_checkpoint2', save_frequency = 1000)


	#loss_per_instr = test.train_example(test_title, test_ingredients, test_targets)
	#print(loss_per_instr)

	test.trainIters()

	#print(test.get_title_indices("World famous chicken"))






