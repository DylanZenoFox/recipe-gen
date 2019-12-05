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
	#		shared_embeddings: shared embedding layer between title, ingredients, and instructions
	# 		embedding_dim: dimension of the embedding vector
	#		hidden_dim: dimension of the GRU hidden dimension
	#		vocab_size: size of the vocabulary
	#		bidirectional: Run as bidirectional RNN, Default = False

	def __init__(self, shared_embeddings, embedding_dim, hidden_dim, vocab_size, bidirectional):

		super(TitleEncoder,self).__init__()

		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.vocab_size = vocab_size
		self.bidirectional = bidirectional

		self.embedding = shared_embeddings
		self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional = self.bidirectional)




	# Forward Pass
	#
	# Parameters:
	#
	# Input: 
	# 		title: Tensor of shape (batch_size, seq_len) representing a batch of sentences
	#
	# Output: 
	#		output:  Tensor of shape (seq_len, batch_size, hidden_size * num_directions) representing output for each timestep for each batch
	#		hidden:  Tensor of shape (num_layers * num_directions, batch_size, hidden_size) representing the last output of the hidden state

	def forward(self, title):

		#Get batch size
		batch_size = title.size(0)

		#Initialize hidden state
		hidden = self.initHidden(batch_size, self.bidirectional)

		#Get embedding for each word
		embedded = self.embedding(title)
		embedded = torch.transpose(embedded,0,1)

		#Get outputs for each state and hidden state at the end
		outputs, hidden = self.gru(embedded,hidden)

		if(self.bidirectional):

			hidden = torch.unsqueeze(hidden[0].add(hidden[1]),0)


		return outputs, hidden



	# Initialize Hidden State
	#
	# Parameters:
	#
	# Input: 
	# 		batch_size: Minibatch size
	#
	# Output: 
	#		Tensor of shape (num_layers = 1, batch_size, hidden_dim) representing initial hidden state

	def initHidden(self,batch_size, bidirectional):

		if(bidirectional):
			return torch.zeros(2, batch_size ,self.hidden_dim, device = device)
		else:
			return torch.zeros(1, batch_size ,self.hidden_dim, device = device)





# TESTING CODE

if(__name__ == '__main__'):


	test = torch.tensor([
						[
							4,8,6,3,6
						]
						])

	embeddings = nn.Embedding(10,5)

	title_encoder = TitleEncoder(embeddings,5,10,10, bidirectional = True)

	out, hidden = title_encoder(test)
	print(out)
	print(out.shape)
	print(hidden)
	print(hidden.shape)