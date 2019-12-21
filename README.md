# Hierarchical Seq2Seq Generation for Recipes 
Recipe Generation project for UMD CMSC723 - Computational Linguistics 1 

## Abstract

We propose a novel method for recipe generation given a summarizing title and set of ingredients. Inspired by Seq2Seq architectures, we implement a basic GRU encoder for the recipe title, a hierarchical GRU encoder for the set of ingredients, and a hierarchical GRU decoder for the list of instructions. Additionally, we experiment with attention between the encoder and decoder models at both the ingredient/instruction level and at the word level.  Although results look very promising, our main limitation was training time, as our model was unable to get through one epoch in a week (training set size: 700,000 recipes). We were able to slightly mitigate this problem through GPU resources, batching, and length restrictions on the training set, but believe longer training is the key to generating quality recipes.




