import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SOS_Token = 0
EOS_Token = 1

################################################
#
# 			INGREDIENT LIST ENCODER
#
################################################


#TODO: MAKE IT WORK WITH INNER_BIDIRECTIONAL = TRUE

class IngredientsEncoder(torch.nn.Module):

	# Initialize Model
	#
	# Parameters:
	#
	# Input: 
	# 		ingr_embed_dim: Size of encoded vector for each ingredient, equal to the hidden state size of the single ingredient encoder
	#		word_embed_dim: Size of word embedding dimension for the single ingredient decoder
	#		hidden_dim: Size of the hidden dimension for the outer RNN 
	#		vocab_size: size of the vocabulary
	#		outer_bidirectional: Run outer RNN as bidirectional RNN, Default = True
	#		inner_bidirectional: Run inner RNN as bidirectional RNN, Default = False

	def __init__(self, ingr_embed_dim, word_embed_dim, hidden_dim, vocab_size, outer_bidirectional=True, inner_bidirectional = False):

		super(IngredientsEncoder, self).__init__()

		self.ingr_embed_dim = ingr_embed_dim
		self.word_embed_dim = word_embed_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.outer_bidirectional = outer_bidirectional
		self.inner_bidirectional = inner_bidirectional

		self.ingr_list_encoder = nn.GRU(ingr_embed_dim, hidden_dim, bidirectional = self.outer_bidirectional)

		self.single_ingr_encoder = SingleIngredientEncoder(word_embed_dim, ingr_embed_dim, vocab_size, bidirectional = inner_bidirectional)


	# Forward Pass of outer GRU
	#
	# Parameters:
	#
	# Input: 
	# 		ingredients: List of ingredient string tensors of shape (num_ingredients, num_words)
	#
	# Output: 
	#		output:  Tensor of shape (num_ingredients, batch_size, hidden_size * num_directions) representing output for each timestep for each batch
	#		hidden:  Tensor of shape (num_layers * num_directions, batch_size, hidden_size) representing the last output of the hidden state for the outer GRU

	def forward(self, ingredients):


		# Initialize hidden
		hidden = self.initHidden(1, self.outer_bidirectional)

		# Collect inputs for each ingredients string
		inputs = []

		# Iterate through each ingredient and pass it to the single ingredient encoder 
		for ingr in ingredients:

			ingr = torch.unsqueeze(ingr,0)

			# Retrieve final hidden state representing encoding for ingredient i
			_ , h = self.single_ingr_encoder(ingr)

			# Squeeze and append
			h = torch.squeeze(h)
			inputs.append(h)

		# Create tensor from list
		inputs = torch.stack(inputs)
		inputs = torch.unsqueeze(inputs,1)


		# Pass these values to the outer GRU and obtain the outputs and hidden values
		outputs , hidden = self.ingr_list_encoder(inputs,hidden)

		return outputs , hidden


	#TODO
	def initHidden(self,batch_size, bidirectional):

		if(bidirectional):
			return torch.zeros(2,batch_size,self.hidden_dim)
		else:	
			return torch.zeros(1,batch_size,self.hidden_dim)



################################################
#
# 			SINGLE INGREDIENT ENCODER
#
################################################


class SingleIngredientEncoder(torch.nn.Module):

	# Initialize Inner GRU Model
	#
	# Parameters:
	#
	# Input: 
	# 		embedding_dim: dimension of the embedding vector
	#		hidden_dim: dimension of the GRU hidden dimension
	#		vocab_size: size of the vocabulary
	#		bidirectional: Run as bidirectional RNN, Default = False

	def __init__(self, embedding_dim, hidden_dim, vocab_size, bidirectional = False):

		super(SingleIngredientEncoder,self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.bidirectional = bidirectional

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.gru = nn.GRU(embedding_dim,hidden_dim, bidirectional = self.bidirectional)



	# Forward Pass of Inner GRU
	#
	# Parameters:
	#
	# Input: 
	# 		ingr_string: Tensor of shape (batch_size, seq_len) representing a batch of ingredient strings
	#
	# Output: 
	#		output:  Tensor of shape (seq_len, batch_size, hidden_size) representing output for each timestep for each batch
	#		hidden:  Tensor of shape (num_layers * num_directions, batch_size, hidden_size) representing the last output of the hidden state

	def forward(self, ingr_string):

		#Get batch size
		batch_size = ingr_string.size(0)

		#Initialize hidden state
		hidden = self.initHidden(batch_size, self.bidirectional)

		#Get embedding for each word
		embedded = self.embedding(ingr_string)
		embedded = torch.transpose(embedded,0,1)

		#Get outputs for each state and hidden state at the end
		output, hidden = self.gru(embedded,hidden)

		return output, hidden


	#COMMENT HERE
	def initHidden(self,batch_size, bidirectional):

		if(bidirectional):
			return torch.zeros(2, batch_size ,self.hidden_dim)
		else: 
			return torch.zeros(1, batch_size ,self.hidden_dim)



if(__name__ == '__main__'):

	test = [ 

				torch.tensor([1,2,3,4,5]),
				torch.tensor([2,4,6,8]),
				torch.tensor([4,3])

			]
	
	ingr_encoder = IngredientsEncoder(10,5,20,10, outer_bidirectional = True, inner_bidirectional = False)
	
	outputs, hidden = ingr_encoder(test) 

	print(outputs)
	print(outputs.shape)

	print(hidden)
	print(hidden.shape)