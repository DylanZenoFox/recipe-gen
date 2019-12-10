import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class IngrToInstrAttention(torch.nn.Module):


	# Instantiate Ingredient to Instruction Attention
	#
	# Parameters:
	#	
	# Input:  
	#		decoder_hidden_dim: hidden dimension of the decoder
	#		encoder_output_dim: output dimension of the encoder
	#		attention_output_dim: dimension of the output of the attention mechanism
	#


	def __init__(self, decoder_hidden_dim, encoder_output_dim, attention_output_dim):

		super(IngrToInstrAttention, self).__init__()
		
		self.decoder_hidden_dim = decoder_hidden_dim
		self.encoder_output_dim = encoder_output_dim
		self.attention_output_dim = encoder_output_dim

		self.projHidden2Encoder = nn.Linear(decoder_hidden_dim, encoder_output_dim)
		self.proj2Output = nn.Linear(encoder_output_dim + decoder_hidden_dim, attention_output_dim)



	# Forward pass of the Ingredient to Instruction Attention
	#
	# Parameters:
	#	
	# Input:  
	#		ingr_outputs: tensor of shape (seq_len, batch_size, encoder_output_size) representing the outputs of the ingredient encoder
	#		decoder_hidden: tensor of shape (1, batch_size, instr_list_hidden_dim) current hidden state of the decoder 
	#
	# Output:
	# 		output: tensor of shape (1, batch_size, attention_output) representing output of attention mechanism
	#		

	def forward(self, ingr_outputs, decoder_hidden):

		# Save decoder hidden for concat later
		ref = decoder_hidden
	
		# Project hidden state down to size of encoder hidden state
		decoder_hidden = self.projHidden2Encoder(decoder_hidden)

		# Reshape tensors for batch matrix multiplication
		decoder_hidden = torch.transpose(torch.transpose(decoder_hidden,0,1),1,2)
		ingr_outputs = torch.transpose(ingr_outputs,0,1)

		# Compute alignment scores
		align_scores = torch.bmm(ingr_outputs, decoder_hidden)
		align_scores = torch.transpose(align_scores,1,2)

		# Compute attention weights
		attn_weights = F.softmax(align_scores, dim = 2)

		#Compute context vecotor for this timestep
		context = torch.bmm(attn_weights, ingr_outputs)
		context = torch.transpose(context,0,1)


		# Concatenate context vector with decoder hidden state
		concat = torch.cat((context,ref), dim = 2)

		# Project to output space
		output = self.proj2Output(concat)

		return output


## TESTING

if(__name__ == '__main__'):

	testAttention = IngrToInstrAttention(4,5,8)

	ingr_outputs = torch.tensor([	#ingr1
									[

										#[1.0,1.0,1.0,1.0,1.0], #b1
										[4.0,4.0,4.0,4.0,4.0]  #b2
									],
									#ingr2
									[
										#[2.0,2.0,2.0,2.0,2.0], #b1
										[5.0,5.0,5.0,5.0,5.0]  #b2
									],
									#ingr3
									[
										#[3.0,3.0,3.0,3.0,3.0], #b1
										[6.0,6.0,6.0,6.0,3.0]  #b2
									]					
								])

	decoder_hidden = torch.tensor([[
										#[4.0,4.0,4.0,4.0],
										[5.0,5.0,5.0,5.0]
									]])

	#print(ingr_outputs.shape)
	#print(decoder_hidden.shape)

	output = testAttention(ingr_outputs, decoder_hidden)

	print(output)
	print(output.shape)




