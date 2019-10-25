#!/usr/bin/env python3.7

import os
import sys
import json
import ijson
import random

f = open('../data/layer1.json')

test_files = []
train_files = []

for i in range(0,10):
	test_files.append(open('../data/test' + str(i) + '.json', 'w+'))
	train_files.append(open('../data/train' + str(i) + '.json', 'w+'))
	print(i)


parser = ijson.parse(f)

newitem = {}

print("Creating train and test sets.")

index = 0

for prefix, event,value in parser:

	#print(str(prefix) + " " + str(event) + " " + str(value))

	if(prefix == "item" and event == "start_map"):
		newItem = {}
		newItem['ingredients'] = []
		newItem['instructions'] = []
	elif(prefix.startswith('item.ingredients') and event == "string"):
		newItem['ingredients'].append(value)
	elif(prefix == 'item.title' and event == 'string'):
		newItem['title'] = value
	elif(prefix.startswith('item.instructions') and event == "string"):
		newItem['instructions'].append(value)
	elif(prefix == "item.id" and event=="string"):
		newItem['id'] = value

	elif(prefix == "item.partition" and event =="string"):
		newItem['partition'] = value

	elif(prefix == "item" and event == "end_map"):

 		rand = random.randint(0,9)

 		if(newItem['partition'] == 'train'):
 			json.dump(newItem, train_files[rand])
 		elif(newItem['partition'] == 'test'):
 			json.dump(newItem, test_files[rand])

 		i += 1
 		if(i % 1000 == 0):
 			print(i)


for f in test_files:
	f.close()

for f in train_files:
	f.close()

print("Files Closed.")




