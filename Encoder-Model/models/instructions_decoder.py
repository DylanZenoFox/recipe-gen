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


	def __init__(self, hidden_dim, vocab_size):
		super(SingleInstructionDecoder,self).__init__()

		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size

		self.embedding = nn.Embedding(vocab_size,hidden_dim)
		self.gru = nn.GRU(hidden_dim,hidden_dim)
		self.out = nn.Linear(hidden_dim,vocab_size)
		self.softmax = nn.LogSoftmax(dim=1)


	def forward(self,input,hidden):

		output = self.embedding(input).view(1,1,-1)
		output = F.relu(output)
		output,hidden =  self.gru(output,hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self):
		return torch.zeros(1,1, self.hidden_dim)






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

	def forward(self, hidden_state):

		return self.softmax(self.l2(F.relu(self.l1(hidden_state))))




#TESTING

test = torch.tensor([
						[1.0,2.0,3.0,4.0,5.0,6.0],
						[6.0,5.0,4.0,3.0,2.0,1.0]
					])

print(test.shape)

endinstr = EndInstructionsClassifier(6,3)

out = endinstr(test)

print(out)
print(out.shape)