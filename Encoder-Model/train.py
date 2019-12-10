from models.encoder_decoder import EncoderDecoder
from lang import Lang

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import shuffle
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_Token = 0
EOS_Token = 1
PAD_Token = 2

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

		self.ingr_instr_attention = True

		self.learning_rate = 0.01

		self.batch_size = 40

		# MODELS

		self.encoder_decoder = EncoderDecoder(vocab_size = self.vocab_size, word_embedding_dim = self.word_embedding_dim, title_hidden_dim = self.title_hidden_dim, ingr_list_hidden_dim = self.ingr_list_hidden_dim,
			single_ingr_hidden_dim = self.single_ingr_hidden_dim, single_instr_hidden_dim = self.single_instr_hidden_dim, end_instr_hidden_dim = self.end_instr_hidden_dim, max_num_instr = self.max_num_instructions,
			max_instr_length = self.max_instr_length, single_instr_tf_ratio = self.single_instr_tf_ratio, instr_list_tf_ratio = self.instr_list_tf_ratio, title_bidirectional = self.title_bidirectional,
			ingr_outer_bidirectional = self.ingredients_outer_bidirectional, ingr_inner_bidirectional = self.ingredients_inner_bidirectional, ingr_instr_attention = self.ingr_instr_attention).to(device)

		# OPTIMIZER

		self.optimizer = optim.Adam(self.encoder_decoder.parameters())

		# LOSS FUNCTIONS

		self.word_criterion = nn.NLLLoss(ignore_index = PAD_Token)
		self.end_instr_criterion = nn.NLLLoss(ignore_index = PAD_Token)

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
		print("Ingredient-Instruction Attention: " + str(self.ingr_instr_attention))
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
		print("Number of GPUs: " + str(torch.cuda.device_count()))
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


	def train_batch(self, batch):

		print("Starting Batch Train")
		print("Memory Used: " + str(torch.cuda.memory_allocated(device=device)))


		self.optimizer.zero_grad()

		title = batch[0]
		ingredients = batch[1]
		target_instructions = batch[2]

		print(title.shape)
		print(ingredients[0].shape)
		print(len(ingredients))
		print(target_instructions[0].shape)
		print(len(target_instructions))

		#print("Target Instructions Length: " + str(len(target_instructions)))

		instructions, word_loss, end_instr_loss = self.encoder_decoder(title, ingredients, self.word_criterion, self.end_instr_criterion, target_instructions)

		#print("Total Word Loss: " + str(word_loss))
		#print("Total End_Instr Loss: " + str(end_instr_loss))

		total_loss = word_loss + end_instr_loss

		#print("Forward Pass Complete")

		total_loss.backward()

		#print("Computed Gradients")

		self.optimizer.step()

		#print("Updated Gradients")

		print("Memory Used: " + str(torch.cuda.memory_allocated(device=device)))


		print("Ending Batch Train")

		return (total_loss.detach().item() / len(target_instructions)) , instructions


	def trainIters(self, print_every = 50, num_epochs = 5, num_train_files = 10):

		iters = 0
		total_loss = 0


		for i in range(num_train_files):

			with open('../data/train' + str(i) + ".json") as f:

				recipe_data = json.load(f)

				print('batching train' + str(i) + '.json ...')
				total_batches = ceil(len(recipe_data) / self.batch_size)
				print('total batches =', total_batches)
				batches = self.batchify(recipe_data, total_batches)
				print('randomizing batch order ...')
				shuffle(batches)
				print('ready to start training!')

				for batch in batches:

					# DYLAN!! This is where the code fails atm

					#loss, decoded_instructions = self.train_example(title_input, ingredients_input, instructions_input)
					loss, decoded_instructions = self.train_batch(batch)

					total_loss += loss

					iters += 1
					if(iters % print_every == 0):

						print("ITER: " + str(iters*self.batch_size))

						print("RECIPE: " + self.lang.indices2string(batch[0][0].tolist()))

						print("ACTUAL INSTRUCTIONS:")

						instructions = batch[2]

						for i in range(len(instructions)):
							print(str(i) + ": " + self.lang.indices2string(instructions[i][0].tolist()))

						print("DECODED INSTRUCTIONS:")

						for i in range(len(decoded_instructions)):
							print(str(i) + ": " + self.lang.indices2string(decoded_instructions[i][0]))

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

	def pad_batch(self, unpadded_batch):
		max_title_len = 0
		max_ingr_len = 0
		max_single_ingr_len = 0
		max_instr_len = 0
		max_single_instr_len = 0
		for title, ingredients, instructions in unpadded_batch:
			if len(title[0]) > max_title_len:
				max_title_len = len(title[0])
			if len(ingredients) > max_ingr_len:
				max_ingr_len = len(ingredients)
			if len(instructions) > max_instr_len:
				max_instr_len = len(instructions)
			for ingr in ingredients:
				if len(ingr) > max_single_ingr_len:
					max_single_ingr_len = len(ingr)
			for instr in instructions:
				if len(instr) > max_single_instr_len:
					max_single_instr_len = len(instr)


		title_batch = torch.zeros(self.batch_size, max_title_len).fill_(PAD_Token).type(torch.LongTensor).to(device)
		ingr_batch_list = [torch.zeros(self.batch_size, max_single_ingr_len).fill_(PAD_Token).type(torch.LongTensor).to(device)for x in range(max_ingr_len)]
		instr_batch_list = [torch.zeros(self.batch_size, max_single_instr_len).fill_(PAD_Token).type(torch.LongTensor).to(device) for x in range(max_instr_len)]

		#ingr_batch_list = [torch.zeros(self.batch_size, max_single_ingr_len).fill_(PAD_Token).type(torch.LongTensor).to(device)] * max_ingr_len
		#instr_batch_list = [torch.zeros(self.batch_size, max_single_instr_len).fill_(PAD_Token).type(torch.LongTensor).to(device)] * max_instr_len

		for i, (title, ingredients, instructions) in enumerate(unpadded_batch):
			for j in range(len(title[0])):
				title_batch[i][j] = title[0][j]
			for j in range(len(ingredients)):
				for k in range(len(ingredients[j])):
					ingr_batch_list[j][i][k] = ingredients[j][k]
			for j in range(len(instructions)):
				for k in range(len(instructions[j])):
					instr_batch_list[j][i][k] = instructions[j][k]

		# uncomment to see internal structure of each batch
		# print()
		# print('max_title_len', max_title_len, '\nmax_ingr_len', max_ingr_len, '\nmax_single_ingr_len', max_single_ingr_len,
		# 	  '\nmax_instr_len', max_instr_len, '\nmax_single_instr_len', max_single_instr_len)
		# print()
		# print('title_batch\t\t', title_batch.shape)
		# print('len(ingr_batch_list)\t', len(ingr_batch_list))
		# print('ingr_batch_list[0]\t', ingr_batch_list[0].shape)
		# print('len(instr_batch_list)\t', len(instr_batch_list))
		# print('instr_batch_list[0]\t', instr_batch_list[0].shape)
		# print('\n-------------')


		return [title_batch, ingr_batch_list, instr_batch_list]



	def batchify(self, recipe_list, total_batches):

		# sort by instruction length, then ingredient lenth,
		# then by the max length of the words in the instructions,
		# and finally by the max length of the words in the ingredients
		recipe_list.sort(key=lambda x: (len(x['instructions']), len(x['ingredients']),
										max([len(i.split()) for i in x['instructions']]),
										max([len(i.split()) for i in x['ingredients']])))

		batches = []
		unpadded_batch = []
		for i, recipe in enumerate(recipe_list):
			if i % self.batch_size == 0 and i>0:
				# unpadded_batch is full, now pad and add to batches
				padded_batch = self.pad_batch(unpadded_batch)
				batches.append(padded_batch)
				unpadded_batch = []
				if 10*len(batches) % total_batches == 0:
					print(len(batches), 'batches\t', round((len(batches)/total_batches)*100), '%')
					print("Memory Used: " + str(torch.cuda.memory_allocated(device=device)))

			title = recipe['title']
			ingredients = recipe['ingredients']
			instructions = recipe['instructions']
			title_input = self.lang.get_title_indices(title)
			ingredients_input = self.lang.get_ingredient_indices(ingredients)
			instructions_input = self.lang.get_instruction_indices(instructions)
			unpadded_batch.append([title_input, ingredients_input, instructions_input])

			if(i == 1000):
				break

		return batches


if(__name__ == '__main__'):

	test = Solver(load_from_path = None, save_to_path = './model_params/updated_train_checkpoint2', save_frequency = 500)


	#loss_per_instr = test.train_example(test_title, test_ingredients, test_targets)
	#print(loss_per_instr)

	test.trainIters()

	#print(test.get_title_indices("World famous chicken"))
