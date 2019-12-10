from models.title_encoder import TitleEncoder
from models.ingredients_encoder import IngredientsEncoder
from models.instructions_decoder import InstructionsDecoder, EndInstructionsClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import json
import spacy
from spacy.lang.en import English
from lang import Lang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lang = Lang(path_to_vocab_files= 'vocab_files/')

SOS_Token = 0
EOS_Token = 1
PAD_Token = 2

class EncoderDecoder(torch.nn.Module):


	def __init__(self, vocab_size, word_embedding_dim, title_hidden_dim, ingr_list_hidden_dim, single_ingr_hidden_dim, single_instr_hidden_dim, end_instr_hidden_dim,
		max_num_instr, max_instr_length, single_instr_tf_ratio, instr_list_tf_ratio, title_bidirectional, ingr_outer_bidirectional, ingr_inner_bidirectional, ingr_instr_attention):

		super(EncoderDecoder, self).__init__()

		self.vocab_size = vocab_size

		self.word_embedding_dim = word_embedding_dim

		self.title_hidden_dim = title_hidden_dim

		self.ingr_list_hidden_dim = ingr_list_hidden_dim
		self.single_ingr_hidden_dim = single_ingr_hidden_dim

		self.ingr_list_output_dim = ingr_list_hidden_dim
		if(ingr_outer_bidirectional):
			self.ingr_list_output_dim *= 2

		self.instr_list_hidden_dim = self.ingr_list_hidden_dim + self.title_hidden_dim
		self.single_instr_hidden_dim = single_instr_hidden_dim

		self.end_instr_hidden_dim = end_instr_hidden_dim

		self.max_num_instr = max_num_instr
		self.max_instr_length = max_instr_length

		self.single_instr_tf_ratio = single_instr_tf_ratio
		self.instr_list_tf_ratio = instr_list_tf_ratio

		self.title_bidirectional = title_bidirectional
		self.ingr_outer_bidirectional = ingr_outer_bidirectional
		self.ingr_inner_bidirectional = ingr_inner_bidirectional

		self.ingr_instr_attention = ingr_instr_attention

		# MODELS 

		# Shared set of word embeddings
		self.shared_word_embeddings = nn.Embedding(vocab_size, word_embedding_dim).to(device)

		# Title Encoder
		self.title_encoder = TitleEncoder(shared_embeddings= self.shared_word_embeddings, embedding_dim = self.word_embedding_dim, 
			hidden_dim = self.title_hidden_dim, vocab_size = self.vocab_size, bidirectional= self.title_bidirectional).to(device)


		# Ingredients Encoder
		self.ingredients_encoder = IngredientsEncoder(shared_embeddings = self.shared_word_embeddings,word_embed_dim = self.word_embedding_dim, ingr_embed_dim = self.single_ingr_hidden_dim,
			hidden_dim = self.ingr_list_hidden_dim, vocab_size = self.vocab_size, outer_bidirectional = self.ingr_outer_bidirectional, inner_bidirectional = self.ingr_inner_bidirectional).to(device)

		# Instructions Decoder
		self.instructions_decoder = InstructionsDecoder(shared_embeddings = self.shared_word_embeddings, word_embedding_dim = self.word_embedding_dim, single_instr_hidden_dim = self.single_instr_hidden_dim,
			instr_list_hidden_dim = self.instr_list_hidden_dim, vocab_size = self.vocab_size, ingredients_output_dim = self.ingr_list_output_dim, max_instr_length = self.max_instr_length, teacher_forcing_ratio = self.single_instr_tf_ratio).to(device)

		self.end_instructions_classifier = EndInstructionsClassifier(instr_embed_dim = self.single_instr_hidden_dim, hidden_dim = self.end_instr_hidden_dim).to(device)


	# Forward pass of the Encoder Decoder
	# Target Instructions needed for teacher forcing
	#
	# Parameters:
	#	
	# Input: 
	#		title: tensor of shape (batch_size, seq_len) representing the title of the recipe
	#		ingredients: list of ingredient tensors of shape (batch_size, num_words) of length num_ingredients
	#		target_instructions: list of instruction tensors of shape (batch_size,num_words) or length num_ingredients, None if evaluating
	#		word_loss: criterion for the word level loss, None if evaluating
	#		end_instr_los: criterion for end instructions loss for the binary classifier. None if evaluating
	#
	# Output:
	#		instructions: list of tensors representing the decoded recipe
	#		word_loss: Loss contribution from words in the instructions
	#		end_instr_loss: Loss contribution from end of instructions loss


	def forward(self, title, ingredients, word_criterion = None, end_instr_criterion = None, target_instructions = None):
		
		batch_size = title.size(0)

		#Losses are 0 to start
		word_loss = 0 
		end_instr_loss = 0

		instructions = []

		# Encode title and ingredients
		title_outputs, encoded_title = self.title_encoder(title)
		ingr_outputs, encoded_ingr, single_ingr_outputs = self.ingredients_encoder(ingredients)


		# Concatenate to get first hidden layer of decoder
		decoder_hidden = torch.cat([encoded_title,encoded_ingr], dim = 2)

		# Create first hidden vector of the instruction decoder
		decoder_input = torch.zeros((1,batch_size, self.single_instr_hidden_dim), device = device)

		# Decide if teacher forcing will be used for this instruction batch
		use_teacher_forcing = True if random.random() < self.instr_list_tf_ratio else False


		# IF EVALUATING
		if(target_instructions is None):

			for i in range(self.max_num_instr):

				if(self.ingr_instr_attention):
					decoder_output, decoder_hidden, decoded_instruction, loss = self.instructions_decoder(decoder_input, decoder_hidden, 
						word_criterion, targets = None, ingr_outputs = ingr_outputs)
				else:
					decoder_output, decoder_hidden, decoded_instruction, loss = self.instructions_decoder(decoder_input, decoder_hidden, 
						word_criterion, targets = None, ingr_outputs = None)


				instructions.append(decoded_instruction)

				end_instructions_pred = self.end_instructions_classifier(decoder_output[0])

				end = end_instructions_pred.topk(1)[1]

				if(end == 1):
					break

				decoder_input = decoder_output.detach()

			return instructions, word_loss, end_instr_loss


		# IF TRAINING

		# Using teacher forcing 
		if(use_teacher_forcing):

			#print("Using instruction level teacher forcing")

			for i in range(len(target_instructions)):

				if(self.ingr_instr_attention):
					decoder_output, decoder_hidden, decoded_instruction, loss = self.instructions_decoder(decoder_input, decoder_hidden, 
						word_criterion, targets = target_instructions[i], ingr_outputs = ingr_outputs)
				else:
					decoder_output, decoder_hidden, decoded_instruction, loss = self.instructions_decoder(decoder_input, decoder_hidden, 
						word_criterion, targets = target_instructions[i], ingr_outputs = None)

				instructions.append(decoded_instruction)

				#print(len(instructions))

				#print("Decoded Instruction: " + str(lang.indices2string(decoded_instruction)))
				#print("Actual Instruction: " + str(lang.indices2string(target_instructions[i].tolist())))

				word_loss += loss

				#print(decoder_output.shape)
				#print(decoder_output[0].shape)

				end_instructions_pred = self.end_instructions_classifier(decoder_output[0])

				#print(end_instructions_pred.topk(1)[1])

				#end = end_instructions_pred.topk(1)[1].item()

				#print(len(target_instructions))

				end_truth = []
				for j in range(target_instructions[i].size(0)):

					if(target_instructions[i][j][0] == PAD_Token):
						end_truth.append(PAD_Token)
					elif(i == len(target_instructions) - 1):
						end_truth.append(1)
					else:
						end_truth.append(1) if (target_instructions[i+1][j][0] == PAD_Token) else end_truth.append(0)

				end_truth = torch.tensor(end_truth, device = device)

				#print(end_truth)


				#if(i == len(target_instructions) - 1):

				single_instr_cl_loss = end_instr_criterion(end_instructions_pred, end_truth)

					#print("Should End [1]. Values: " + str(end_instructions_pred) + " Loss:" + str(single_instr_cl_loss.item()))

				end_instr_loss += single_instr_cl_loss

				#else:
					#single_instr_cl_loss = end_instr_criterion(end_instructions_pred, end_truth)

					#print("Should not end [0]. Values: " + str(end_instructions_pred) + " Loss:" + str(single_instr_cl_loss.item()))

					#end_instr_loss += single_instr_cl_loss

				decoder_input = decoder_output.detach()

		# Not using teacher forcing
		else:

			#print("Not using instruction level teacher forcing")

			for i in range(len(target_instructions)):

				if(self.ingr_instr_attention):
					decoder_output, decoder_hidden, decoded_instruction, loss = self.instructions_decoder(decoder_input, decoder_hidden, 
						word_criterion, targets = target_instructions[i], ingr_outputs = ingr_outputs)
				else:
					decoder_output, decoder_hidden, decoded_instruction, loss = self.instructions_decoder(decoder_input, decoder_hidden, 
						word_criterion, targets = target_instructions[i], ingr_outputs = None)

				instructions.append(decoded_instruction)

				word_loss += loss

				end_instructions_pred = self.end_instructions_classifier(decoder_output[0])

				#print(end_instructions_pred.topk(1)[1])



				end_truth = []
				for j in range(target_instructions[i].size(0)):

					if(target_instructions[i][j][0] == PAD_Token):
						end_truth.append(PAD_Token)
					elif(i == len(target_instructions) - 1):
						end_truth.append(1)
					else:
						end_truth.append(1) if (target_instructions[i+1][j][0] == PAD_Token) else end_truth.append(0)


				end_truth = torch.tensor(end_truth, device = device)

				single_instr_cl_loss = end_instr_criterion(end_instructions_pred, end_truth)

				end_instr_loss += single_instr_cl_loss


				decoder_input = decoder_output.detach()


		return instructions, word_loss, end_instr_loss