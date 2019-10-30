#!/usr/bin/env python3.7


import os
import sys
import json
import ijson
import random

f = open('../data/test0.json')

d = json.load(f)

print(d[0])



