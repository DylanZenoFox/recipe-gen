import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class InstructionsDecoder(torch.nn.Module):


	def __init__():
		pass



	def forward():
		pass


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
		self.softmax = nn.Softmax(dim=1)


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

test = torch.tensor([[
						[1.0,2.0,3.0,4.0,5.0,6.0],
						[6.0,5.0,4.0,3.0,2.0,1.0],
						[6.0,5.0,4.0,3.0,2.0,1.0]
					]])

#print(test.shape)

innerGRU = SingleInstructionDecoder(embedding_dim = 5, hidden_dim = 6, vocab_size = 30)


out, hidden = innerGRU(torch.tensor([[9],[3],[4]]),test)
out, hidden = innerGRU(torch.tensor([[5],[6],[2]]),hidden)

print(out.shape)
print(out)
print(hidden.shape)
print(hidden)


