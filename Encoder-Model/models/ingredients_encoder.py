import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SOS_Token = 0
EOS_Token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

	def __init__(self, shared_embeddings, word_embed_dim, ingr_embed_dim, hidden_dim, vocab_size, outer_bidirectional, inner_bidirectional):

		super(IngredientsEncoder, self).__init__()

		self.ingr_embed_dim = ingr_embed_dim
		self.word_embed_dim = word_embed_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.outer_bidirectional = outer_bidirectional
		self.inner_bidirectional = inner_bidirectional

		self.ingr_list_encoder = nn.GRU(ingr_embed_dim, hidden_dim, bidirectional = self.outer_bidirectional)

		self.single_ingr_encoder = SingleIngredientEncoder(shared_embeddings = shared_embeddings, 
			embedding_dim = self.word_embed_dim, hidden_dim = self.ingr_embed_dim, vocab_size= self.vocab_size, inner_bidirectional = self.inner_bidirectional)


	# Forward Pass of outer GRU
	#
	# Parameters:
	#
	# Input: 
	# 		ingredients: List of ingredient string tensors of shape (num_batches,  num_words) of length num_ingredients
	#
	# Output: 
	#		output:  Tensor of shape (num_ingredients, batch_size, hidden_size * num_directions) representing output for each timestep for each batch
	#		hidden:  Tensor of shape (num_layers * num_directions, batch_size, hidden_size) representing the last output of the hidden state for the outer GRU

	def forward(self, ingredients):

		batch_size = ingredients[0].size(0)


		# Initialize hidden
		hidden = self.initHidden(batch_size, self.outer_bidirectional)

		# Collect inputs for each ingredients string
		inputs = []
		single_ingr_outputs = []

		# Iterate through each ingredient and pass it to the single ingredient encoder 
		for ingr in ingredients:

			#ingr = torch.unsqueeze(ingr,0)

			# Retrieve final hidden state representing encoding for ingredient i
			single_ingr_output , h = self.single_ingr_encoder(ingr)
			
			# Squeeze and append
			h = torch.squeeze(h,0)
			inputs.append(h)

			single_ingr_outputs.append(single_ingr_output)

		# Create tensor from list
		inputs = torch.stack(inputs)
		#inputs = torch.unsqueeze(inputs,1)


		# Pass these values to the outer GRU and obtain the outputs and hidden values
		outputs , hidden = self.ingr_list_encoder(inputs,hidden)

		if(self.outer_bidirectional):

			hidden = torch.unsqueeze(hidden[0].add(hidden[1]), 0)

		#print("Encoder Output Shape: " + str(outputs.shape ))
		#print("Bidirectional: " + str(self.outer_bidirectional))

		return outputs , hidden , single_ingr_outputs


	#TODO
	def initHidden(self,batch_size, bidirectional):

		if(bidirectional):
			return torch.zeros(2,batch_size,self.hidden_dim, device = device)
		else:	
			return torch.zeros(1,batch_size,self.hidden_dim, device = device)



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

	def __init__(self, shared_embeddings, embedding_dim, hidden_dim, vocab_size, inner_bidirectional):

		super(SingleIngredientEncoder,self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.inner_bidirectional = inner_bidirectional

		self.embedding = shared_embeddings
		self.gru = nn.GRU(embedding_dim,hidden_dim, bidirectional = self.inner_bidirectional)



	# Forward Pass of Inner GRU
	#
	# Parameters:
	#
	# Input: 
	# 		ingr_string: Tensor of shape (batch_size, seq_len) representing a batch of ingredient strings
	#
	# Output: 
	#		output:  Tensor of shape (seq_len, batch_size, hidden_size) representing output for each timestep for each batch
	#		hidden:  Tensor of shape (num_layers, batch_size, hidden_size) representing the last output of the hidden state

	def forward(self, ingr_string):

		#Get batch size
		batch_size = ingr_string.size(0)

		#Initialize hidden state
		hidden = self.initHidden(batch_size, self.inner_bidirectional)

		#Get embedding for each word
		embedded = self.embedding(ingr_string)
		embedded = torch.transpose(embedded,0,1)

		#Get outputs for each state and hidden state at the end
		output, hidden = self.gru(embedded,hidden)


		if(self.inner_bidirectional):

			hidden = torch.unsqueeze(hidden[0].add(hidden[1]), 0)


		return output, hidden



	def initHidden(self,batch_size, bidirectional):

		if(bidirectional):
			return torch.zeros(2, batch_size ,self.hidden_dim, device = device)
		else:
			return torch.zeros(1, batch_size ,self.hidden_dim, device = device)




if(__name__ == '__main__'):

	test = [ 
				#Ingr1
				torch.tensor([[1,2,3,4,5],
							[1,2,3,4,5]]),

				#Ingr2
				torch.tensor([[5,4,3,2,1],
							[5,4,3,2,1]]),

				#Ingr3
				torch.tensor([[1,1,1,1,1],
							[1,1,1,1,1]])


			]
	

	embeddings = nn.Embedding(10,5)

	ingr_encoder = IngredientsEncoder(embeddings, 5,10,20,10, outer_bidirectional = True, inner_bidirectional = True)
	
	outputs, hidden, single_ingr_outputs = ingr_encoder(test) 

	print(outputs)
	print(outputs.shape)

	print(hidden)
	print(hidden.shape)

	print(single_ingr_outputs)
	print(single_ingr_outputs[0].shape)