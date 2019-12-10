from .attention_mechanisms import IngrToInstrAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

SOS_Token = 0
EOS_Token = 1
PAD_Token = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InstructionsDecoder(torch.nn.Module):

	# Initialize Total Instructions Decoder
	#
	# Parameters:
	#	
	# Input: 
	#		shared_embeddings: shared words embeddings between title, ingredients, and instructions
	# 		vocab_size: size of the vocabulary
	#		single_instr_hidden_dim: dimension of the hidden state for the single instruction decoder GRU
	#		word_embedding_dim: dimension of the word embeddings
	#		instr_list_hidden_dim: dimension of the hidden layer for all the recipe instructions
	#		ingredients_output_dim: dimension of the ingredients hidden layer, used for attention
	# 		max_instr_length: max length of instructions
	#		teacher_forcing_ratio: what proportion of innerGRU decoding sessions will be teacher forced

	def __init__(self, shared_embeddings, word_embedding_dim, single_instr_hidden_dim, instr_list_hidden_dim, vocab_size, ingredients_output_dim, max_instr_length = 20, teacher_forcing_ratio = 0.5):

		super(InstructionsDecoder,self).__init__()

		self.single_instr_hidden_dim = single_instr_hidden_dim
		self.word_embedding_dim = word_embedding_dim
		self.instr_list_hidden_dim = instr_list_hidden_dim
		self.vocab_size = vocab_size
		self.max_instr_length = max_instr_length
		self.teacher_forcing_ratio = teacher_forcing_ratio

		self.outerGRU = nn.GRU(single_instr_hidden_dim, instr_list_hidden_dim)

		self.outer2inner = nn.Linear(instr_list_hidden_dim, single_instr_hidden_dim)

		self.ingr2instr_attention = IngrToInstrAttention(decoder_hidden_dim = instr_list_hidden_dim, encoder_output_dim = ingredients_output_dim, attention_output_dim = single_instr_hidden_dim)

		self.innerGRU = SingleInstructionDecoder(shared_embeddings = shared_embeddings, embedding_dim= word_embedding_dim, hidden_dim = single_instr_hidden_dim, vocab_size = vocab_size, ingredients_output_dim = ingredients_output_dim)

		#self.end_instructions_classifier = EndInstructionsClassifier(instr_embed_dim = single_instr_hidden_dim, hidden_dim= end_instr_hidden_dim)


	# Forward pass of the Instruction Decoder
	#
	# Parameters:
	#	
	# Input: 
	#		input: tensor of shape (1, batch_size , instr_hidden_dim) the last hidden state of the previous inner GRU
	#		hidden: tensor of shape (1, batch_size, rec_hidden_dim) representing the hidden state for the previous timestep of the outer GRU
	#		word_loss: loss criterion for the word level
	#		targets: tensor of shape (batch_size, num_words) containing target indices for ground truth instructions.  None if evaluating.
	#		ingr_outputs: tensor of shape (seq_len, batch_size, encoder_output_size) representing the outputs of the ingredient encoder. Used for attention
	#
	# Output:
	#		output: tensor of shape (batch_size, single_instr_hidden_dim) representing the last hidden state of the inner GRU for this timestep
	#		hidden: tensor of shape (1, batch_size, instr_list_hidden_dim) representing the hidden state of the current timestep
	#		decoded_instruction: tensor of shape (num_words) containing indices for words in instructions
	# 		loss: total loss contributed to by this stage of GRU

	def forward(self, input, hidden, word_loss, targets = None, ingr_outputs = None):

		if(targets is not None):
			targets = torch.transpose(targets,0,1)

		batch_size = hidden.size(1)

		#Get the next state from the outer GRU using the previous hidden state and the input from the last timestep
		single_instr, hidden = self.outerGRU(input, hidden)

		# Using ingredient to instruction attention
		if(ingr_outputs is not None):
			single_instr = self.ingr2instr_attention(ingr_outputs, single_instr)
		# If no attention, use a linear layer to convert the hidden state of the outer GRU to the first hidden state of the inner GRU
		else:
			single_instr = self.outer2inner(single_instr)


		# Introduce skip connection to allow new instruction to gain information about old instruction
		single_instr = single_instr.add(input)

		single_instr = F.relu(single_instr)

		#Always create first input as SOS token
		instr_input = torch.tensor([[SOS_Token] for x in range(batch_size)],device = device)

		# Values to return
		decoded_instruction = [[SOS_Token] for x in range(batch_size)]
		loss = 0


		#If evaluating then just generate until EOS token
		if(targets is None):

			# Iterate until you hit max length or an EOS token.
			# Assume first token is always SOS_Token
			for i in range(1, self.max_instr_length):

				instr_output, instr_hidden = self.innerGRU(instr_input, single_instr, ingr_outputs)

				# Get max index and value of the output
				topv, topi = instr_output.topk(1)

				# Append this word value to the decoded instructions list
				for j in range(len(decoded_instruction)):
					decoded_instruction[j].append(topi[j].item())

				# Set the input to the next stage to be this input value
				instr_input = topi.detach()


				# I guess will always run until max length


				# If the EOS_Token is generated, end generation
				#if(instr_input.item() == EOS_Token):
				#	break


		# If training, then we have ground truth values
		else:

			# Decided if teacher forcing will be used for this instruction
			use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

			# Unsqueeze to get into better format for passing into GRU
			#targets = torch.unsqueeze(targets,1)

			# If using teacher forcing
			if(use_teacher_forcing):

				#print("Using word level teacher forcing")

				# Iterate until the end of the target 
				for i in range(1, targets.size(0)):

					instr_output, instr_hidden = self.innerGRU(instr_input, single_instr, ingr_outputs)

					# Calculate loss on this word
					in_loss = word_loss(instr_output, targets[i])

					loss += in_loss

					#print("inner loss for word " + str(targets[i].item()) + ": " + str(in_loss.item()))

					# Get the top value and index of this output
					topv, topi = instr_output.topk(1)

					# Append this to the decoded instructions

					for j in range(len(decoded_instruction)):
						decoded_instruction[j].append(topi[j].item())

					# Set the next input to be the target value at this timestep
					instr_input = torch.unsqueeze(targets[i],1)


			else:

				#print("Not using word level Teacher Forcing")

				# Iterate until end of target instruction
				for i in range(1, targets.size(0)):

					instr_output, instr_hidden = self.innerGRU(instr_input, single_instr, ingr_outputs)

					# Get the top value and index of this output					
					topv, topi = instr_output.topk(1)

					for j in range(len(decoded_instruction)):
						decoded_instruction[j].append(topi[j].item())

					# Calculate loss for this word
					in_loss = word_loss(instr_output, targets[i])

					loss += in_loss
					
					#print("inner loss for word " + str(targets[i].item()) + ": " + str(in_loss.item()))

					# Set the next input to be predicted word
					instr_input = topi.detach()

					#If end of sentence is found, stop generating
					#if(instr_input.item() == EOS_Token):
					#	break


		#end_instr = self.end_instructions_classifier(instr_hidden)

		# Set output to be the last hidden state of the inner GRU for this timestep
		output = instr_hidden

		instruction = decoded_instruction

		return output, hidden, instruction, loss




################################################
#
# 	SINGLE INSTRUCTION DECODER
#
################################################


class SingleInstructionDecoder(torch.nn.Module):

	# Initialize Single Instruction Decoder
	#
	# Parameters:
	#	
	# Input: 
	# 		vocab_size: size of the vocabulary
	#		embedding_dim: dimension of the word embedding
	#		hidden_dim: dimension of the hidden state 

	def __init__(self, shared_embeddings, embedding_dim, hidden_dim, vocab_size, ingredients_output_dim):
		super(SingleInstructionDecoder,self).__init__()

		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim

		self.embedding = shared_embeddings
		self.gru = nn.GRU(embedding_dim,hidden_dim)
		self.out = nn.Linear(hidden_dim,vocab_size)
		self.softmax = nn.LogSoftmax(dim=1)

		self.inner_attention = IngrToInstrAttention(decoder_hidden_dim = hidden_dim, encoder_output_dim = ingredients_output_dim, attention_output_dim = hidden_dim)


	# Forward pass of the Single Instruction Decoder
	#
	# Parameters:
	#	
	# Input: 
	#		input: tensor of shape (batch_size, 1) representing vocab indices to extract for embedding 
	#		hidden: tensor of shape (1, batch_size, hidden_dim) representing the hidden state for the previous timestep
	#
	# Output:
	#		output: tensor of shape (batch_size, vocab_size) representing softmax distribution over words in vocabulary
	#		hidden: tensor of shape (1, batch_size, hidden_dim) representing the hidden state of the current timestep

	def forward(self,input,hidden, ingr_outputs = None):

		batch_size = hidden.size(1)

		output = self.embedding(input)
		output = torch.transpose(output,0,1)

		output = F.relu(output)
		output, hidden =  self.gru(output,hidden)

		if(ingr_outputs is not None):
			output = self.inner_attention(ingr_outputs, output)

		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self,batch_size):
		return torch.zeros(1,batch_size, self.hidden_dim, device = device)






################################################
#
# 	END OF INSTRUCTIONS BINARY CLASSIFIER
#
################################################

class EndInstructionsClassifier(torch.nn.Module):

	# Initialize Binary Classification MLP
	#
	# Parameters:
	#	
	# Input: 
	# 		instr_embed_dim: dimension of the hidden layer of outer GRU
	#		hidden_dim: dimension of the hidden state 


	def __init__(self, instr_embed_dim, hidden_dim):
		super(EndInstructionsClassifier, self).__init__()

		self.instr_embed_dim = instr_embed_dim
		self.hidden_dim = hidden_dim

		self.l1 = nn.Linear(instr_embed_dim,hidden_dim)
		self.l2 = nn.Linear(hidden_dim, 2)
		self.softmax = nn.LogSoftmax(dim = 1)


	# Forward Pass of Binary Classification MLP
	#
	# Parameters:
	#	
	# Input: 
	# 		hidden_state: Tensor of shape (batch size, hidden_dim) representing the hidden state of the outer GRU 
	#
	# Output: 
	#		LogSoftmax: Tensor of shape (batch_size, 2) representing log softmax values for the binary classifier

	def forward(self, hidden_state):

		return self.softmax(self.l2(F.relu(self.l1(hidden_state))))




#TESTING

if(__name__ == '__main__'):


	classifier = EndInstructionsClassifier(10,5)

	test = torch.tensor([
							[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0],
							[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0],
							[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0],
						])

	print(classifier(test))

	# embeddings = nn.Embedding(10,3)
	# decoder = InstructionsDecoder( shared_embeddings= embeddings, word_embedding_dim = 3, single_instr_hidden_dim = 5, instr_list_hidden_dim  = 10, 
	# 	vocab_size = 10, ingredients_output_dim = 10, teacher_forcing_ratio = 0.0)
	# crit = nn.NLLLoss(ignore_index = 3)

	# test = torch.tensor([
	# 						[[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0],
	# 						[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0]]
	# 					])

	# input = torch.tensor([
	# 						[[3.0,4.0,5.0,6.0,7.0],
	# 						[3.0,4.0,5.0,6.0,7.0]]
	# 					])

	# targets = torch.tensor([
	# 							[0,8,8,8,8,8,1,3,3,3],
	# 							[0,8,8,8,8,8,5,5,1,3]
	# 						])

	# output, hidden, instructions, loss = decoder( input= input,  hidden = test, word_loss = crit, targets = None)

	# print(output)
	# print(hidden)
	# print(loss)
	# print(instructions)







