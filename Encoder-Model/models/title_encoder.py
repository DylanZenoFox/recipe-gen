import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TitleEncoder(torch.nn.Module):

	# Initialize Model
	#
	# Parameters:
	#
	# Input: 
	# 		embedding_dim: dimension of the embedding vector
	#		hidden_dim: dimension of the GRU hidden dimension
	#		vocab_size: size of the vocabulary
	#		bidirectional: Run as bidirectional RNN, Default = False

	def __init__(self, embedding_dim, hidden_dim, vocab_size):

		super(TitleEncoder,self).__init__()

		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.vocab_size = vocab_size

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True)


	# Forward Pass
	#
	# Parameters:
	#
	# Input: 
	# 		title: Tensor of shape (batch_size, seq_len) representing a batch of sentences
	#
	# Output: 
	#		output:  Tensor of shape (seq_len, batch_size, hidden_size) representing output for each timestep for each batch
	#		hidden:  Tensor of shape (num_layers, batch_size, hidden_size) representing the last output of the hidden state

	def forward(self, title):

		#Get batch size
		batch_size = title.size(0)

		#Initialize hidden state
		hidden = self.initHidden(batch_size)

		#Get embedding for each word
		embedded = self.embedding(title)
		embedded = torch.transpose(embedded,0,1)

		#Get outputs for each state and hidden state at the end
		output, hidden = self.gru(embedded, hidden)
		hidden_forward = hidden[:, 1] 
		hidden_backward = hidden[1, :] 
		hidden = (hidden_forward * hidden_backward).view(1, batch_size, -1) #may want to change: num_layers hard coded
		return output, hidden



	# Initialize Hidden State
	#
	# Parameters:
	#
	# Input: 
	# 		batch_size: Minibatch size
	#
	# Output: 
	#		Tensor of shape (num_layers = 1, batch_size, hidden_dim) representing initial hidden state

	def initHidden(self, batch_size):
		 return (torch.zeros(2, batch_size ,self.hidden_dim, device = device))


# TESTING CODE

if(__name__ == '__main__'):


	test = torch.tensor([

						[
							5,8,6,7,6
						],

						[
							4,8,6,3,6
						]
						])

	title_encoder = TitleEncoder(5,10,10)

	out, hidden = title_encoder(test)
	print('Out: ', out)
	print('Out.shape: ',out.shape)
	print('hidden: ', hidden)
	print('hidden.shape: ', hidden.shape)