#!/usr/bin/env python3.7

import os
import sys
import json
import ijson
from spacy.lang.en import English
import random

tokenizer = English().Defaults.create_tokenizer(English())

max_title_length = 15
max_ingr_length = 15
max_num_ingr = 20
max_instr_length = 25
max_num_instr = 20


def shorten_string(string, max_length):

	tokens = tokenizer(string)
	newStr =  "".join(str(tokens[:max_length]))

	return newStr

	#print("Shorten:")
	#print(string)
	#print(newStr)




f = open('../data/layer1.json')

#test_files = []
#train_files = []

#for i in range(0,10):
#	test_files.append(open('../data/test' + str(i) + '.json', 'w+'))
#	train_files.append(open('../data/train' + str(i) + '.json', 'w+'))
#	print(i)


parser = ijson.parse(f)

newitem = {}

print("Creating train and test sets.")

trainCounter = 0
testCounter = 0

currentTrainFile = []
currentTestFile = []

for prefix, event,value in parser:

	#print(str(prefix) + " " + str(event) + " " + str(value))

	if(prefix == "item" and event == "start_map"):
		newItem = {}
		newItem['ingredients'] = []
		newItem['instructions'] = []
	elif(prefix.startswith('item.ingredients') and event == "string"):

		if(len(newItem['ingredients']) < max_num_ingr):

			newItem['ingredients'].append(shorten_string(value,max_ingr_length))


	elif(prefix == 'item.title' and event == 'string'):
		newItem['title'] = shorten_string(value, max_title_length)
	elif(prefix.startswith('item.instructions') and event == "string"):

		if(len(newItem['instructions']) < max_num_instr):
			newItem['instructions'].append(shorten_string(value,max_instr_length))

	elif(prefix == "item.id" and event=="string"):
		newItem['id'] = value

	elif(prefix == "item.partition" and event =="string"):
		newItem['partition'] = value

	elif(prefix == "item" and event == "end_map"):

 		if(newItem['partition'] == 'train'):

 			currentTrainFile.append(newItem)

 			if(len(currentTrainFile) == 70000):
 				print("Train: " + str(trainCounter))
 				trainf = open('../data/train' + str(trainCounter) + '.json', 'w+')
 				trainCounter += 1
 				json.dump(currentTrainFile, trainf)

 				trainf.close()

 				currentTrainFile = []

 		elif(newItem['partition'] == 'test'):

 			currentTestFile.append(newItem)

 			if(len(currentTestFile) == 30000):
 				print("Test: " + str(testCounter))
 				testf = open('../data/test' + str(testCounter) + '.json', 'w+')
 				testCounter += 1
 				json.dump(currentTestFile, testf)

 				testf.close()

 				currentTestFile = []


