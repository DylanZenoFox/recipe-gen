import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TitleEncoder(torch.nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, bidirectional = False):

		super(TitleEncoder,self).__init__()

		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.vocab_size = vocab_size

		self.bidirectional = bidirectional

		self.embedding = nn.Embedding(vocab_size,embedding_dim)
		self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional = self.bidirectional)



	# Forward Pass
	# Parameters:
	# Input: 
	# 		title: Tensor of shape (batch_size, seq_len) representing a batch of sentences
	# Output: 
	#		output:  Tensor of shape (seq_len, batch_size, hidden_size) representing output for each timestep for each batch
	#		hidden:  Tensor of shape (num_layers, batch_size, hidden_size) representing the last output of the hidden state

	def forward(self, title):
		print(title.shape)
		print(title)

		batch_size = title.size(0)
		print("batch_size:")
		print(batch_size)

		hidden = self.initHidden(batch_size)

		print("Hidden:")
		print(hidden)

		embedded = self.embedding(title)
		print(embedded)
		embedded = torch.transpose(embedded,0,1)

		print("Embedded:")
		print(embedded)
		print(embedded.shape)

		output, hidden = self.gru(embedded,hidden)

		return output, hidden




	def initHidden(self,batch_size):

		return torch.zeros(1, batch_size ,self.hidden_dim)


test = torch.tensor([

					[
						5,8,6,7,6
					],

					[
						4,8,6,3,6
					]
					])

title_encoder = TitleEncoder(5,10,10)
torch.manual_seed(0)

out, hidden = title_encoder(test)
print(out)
print(out.shape)
print(hidden)
print(hidden.shape)