import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
	# 		ingredients: Tensor of shape (batch_size, num_ingredients, seq_len) representing a batch of ingredient sets
	#
	# Output: 
	#		output:  Tensor of shape (num_ingredients, batch_size, hidden_size) representing output for each timestep for each batch
	#		hidden:  Tensor of shape (num_layers * num_directions, batch_size, hidden_size) representing the last output of the hidden state for the outer GRU

	def forward(self, ingredients):

		batch_size = ingredients.size(0)

		ingredients = torch.transpose(ingredients,0,1)

		hidden = self.initHidden(batch_size, self.outer_bidirectional)

		inputs = []

		for i in range(ingredients.size(0)):

			_ , h = self.single_ingr_encoder(ingredients[i])

			h = torch.squeeze(h)
			inputs.append(h)

		inputs = torch.stack(inputs)

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
		print("hola")

		#Get outputs for each state and hidden state at the end
		output, hidden = self.gru(embedded,hidden)
		print("macha")

		return output, hidden


	#COMMENT HERE
	def initHidden(self,batch_size, bidirectional):

		if(bidirectional):
			return torch.zeros(2, batch_size ,self.hidden_dim)
		else: 
			return torch.zeros(1, batch_size ,self.hidden_dim)


test = torch.tensor([
					#b1
						[
							#ingr list 1
							[1,2,3,4,5],
							[1,3,5,6,2],
						],
					#b2
						[
							#ingr list 2
							[1,2,3,4,5],
							[1,3,5,2,7]	
						]
					])

print(test.shape)

ingr_encoder = IngredientsEncoder(10,5,20,10, outer_bidirectional = True, inner_bidirectional = False)

outputs, hidden = ingr_encoder(test) 

print(hidden)
print(hidden.shape)