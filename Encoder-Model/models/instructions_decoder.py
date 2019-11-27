import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

SOS_Token = 0
EOS_Token = 1

class InstructionsDecoder(torch.nn.Module):

	# Initialize Total Instructions Decoder
	#
	# Parameters:
	#	
	# Input: 
	# 		vocab_size: size of the vocabulary
	#		instr_hidden_dim: dimension of the hidden state for the single instruction decoder GRU
	#		word_embedding_dim: dimension of the word embeddings
	#		rec_hidden_dim: dimension of the hidden layer for all the recipe instructions
	#		binary_MLP_hidden_dim: dimension of the hidden layer of the End of Instructions binary classifier
	# 		max_instr_length: max length of instructions

	def __init__(self, instr_hidden_dim, word_embedding_dim, rec_hidden_dim, vocab_size, max_instr_length = 20, teacher_forcing_ratio = 0.5):

		super(InstructionsDecoder,self).__init__()

		self.instr_hidden_dim = instr_hidden_dim
		self.word_embedding_dim = word_embedding_dim
		self.rec_hidden_dim = rec_hidden_dim
		self.vocab_size = vocab_size
		self.max_instr_length = max_instr_length
		self.teacher_forcing_ratio = teacher_forcing_ratio


		self.outerGRU = nn.GRU(instr_hidden_dim, rec_hidden_dim)

		self.outer2inner = nn.Linear(rec_hidden_dim, instr_hidden_dim)

		self.innerGRU = SingleInstructionDecoder(word_embedding_dim, instr_hidden_dim, vocab_size)


	# Forward pass of the Instruction Decoder
	#
	# Parameters:
	#	
	# Input: 
	#		input: tensor of shape (1, batch_size , instr_hidden_dim) the last hidden state of the previous inner GRU
	#		hidden: tensor of shape (1, batch_size, rec_hidden_dim) representing the hidden state for the previous timestep of the outer GRU
	#		targets: tensor of shape (num_words) containing target indices for ground truth instructions.  None if evaluating.
	#
	# Output:
	#		output: tensor of shape (batch_size, instr_hidden_dim) representing the last hidden state of the inner GRU for this timestep
	#		hidden: tensor of shape (1, batch_size, rec_hidden_dim) representing the hidden state of the current timestep
	#		decoded_instruction: tensor of shape (num_words) containing indices for words in instructions
	# 		loss: total loss contributed to by this stage of GRU

	def forward(self, input, hidden, inner_loss, targets = None):

		#Get the next state from the outer GRU using the previous hidden state and the input from the last timestep
		single_instr , hidden = self.outerGRU(input, hidden)

		# Use a linear layer to convert the hidden state of the outer GRU to theh first hidden state of the inner GRU
		single_instr = self.outer2inner(single_instr)
		single_instr = F.relu(single_instr)

		#Always create first input as SOS token
		instr_input = torch.tensor([[SOS_Token]])

		# Values to return
		decoded_instruction = [SOS_Token]
		loss = 0

		#If evaluating then just generate until EOS token
		if(targets is None):


			# Iterate until you hit max length or an EOS token.
			# Assume first token is always SOS_Token
			for i in range(1, self.max_instr_length):

				instr_output, instr_hidden = self.innerGRU(instr_input, single_instr)

				# Get max index and value of the output
				topv, topi = instr_output.topk(1)

				# Append this word value to the decoded instructions list
				decoded_instruction.append(topi.item())

				# Set the input to the next stage to be this input value
				instr_input = topi.detach()

				# If the EOS_Token is generated, end generation
				if(instr_input.item() == EOS_Token):
					break


		# If training, then we have ground truth values
		else:

			# Decided if teacher forcing will be used for this instruction
			use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

			# Unsqueeze to get into better format for passing into GRU
			targets = torch.unsqueeze(targets,1)

			# If using teacher forcing
			if(use_teacher_forcing):

				print("Using teacher forcing")

				# Iterate until the end of the target 
				for i in range(1, targets.size(0)):

					instr_output, instr_hidden = self.innerGRU(instr_input, single_instr)

					# Calculate loss on this word
					loss += inner_loss(instr_output, targets[i])

					# Get the top value and index of this output
					topv, topi = instr_output.topk(1)

					# Append this to the decoded instructions
					decoded_instruction.append(topi.item())

					# Set the next input to be the target value at this timestep
					instr_input = torch.unsqueeze(targets[i],0)


			else:
				print("Not using teacher forcing")

				# Iterate until end of target instruction
				for i in range(1, targets.size(0)):

					instr_output, instr_hidden = self.innerGRU(instr_input, single_instr)

					# Get the top value and index of this output					
					topv, topi = instr_output.topk(1)
					decoded_instruction.append(topi.item())

					# Calculate loss for this word
					loss += inner_loss(instr_output, targets[i])

					# Set the next input to be predicted word
					instr_input = topi.detach()

					#If end of sentence is found, stop generating
					if(instr_input.item() == EOS_Token):
						break

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

	def __init__(self, embedding_dim, hidden_dim, vocab_size):
		super(SingleInstructionDecoder,self).__init__()

		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim

		self.embedding = nn.Embedding(vocab_size,embedding_dim)
		self.gru = nn.GRU(embedding_dim,hidden_dim)
		self.out = nn.Linear(hidden_dim,vocab_size)
		self.softmax = nn.LogSoftmax(dim=1)


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

	def forward(self,input,hidden):

		batch_size = hidden.size(1)

		output = self.embedding(input)
		output = torch.transpose(output,0,1)

		output = F.relu(output)
		output, hidden =  self.gru(output,hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self,batch_size):
		return torch.zeros(1,batch_size, self.hidden_dim)






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

decoder = InstructionsDecoder(instr_hidden_dim = 5, word_embedding_dim = 3, rec_hidden_dim  = 10, vocab_size = 10)
crit = nn.NLLLoss()

test = torch.tensor([
						[[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0]]
					])

output, hidden, instructions, loss = decoder( input= torch.tensor([[[3.0,4.0,5.0,6.0,7.0]]]),  hidden = test, inner_loss = crit, targets = torch.tensor([0,3,5,5,3,9,1]))

print(output)
print(hidden)
print(loss)
print(instructions)






