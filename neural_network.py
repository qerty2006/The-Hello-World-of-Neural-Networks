# -*- coding: utf-8 -*-

#pip install numdifftools
import numpy as np
import math
import csv
import random as r
import numdifftools as nd
from matplotlib import pyplot as plt

"""Getting database (downloaded)"""

#Get the MNIST training set
trainingset = './mnist_train_small.csv'
#creating the brain
neurons = 2 # nodes per layer
layers = 4 # more layers => longer to train

input_size = 28**2 # based on dataset
output_size = 10 # based on dataset


labels = []
data = []



#Turns data set into usuable arrays from training
with open(trainingset, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
      unfiltered_data = row[0].split(",")
      unfiltered_data = [float(i) for i in unfiltered_data]
      label = [0 for i in range(10)]
      label[int(unfiltered_data.pop(0))] = 1
      labels.append(label)
      data.append(unfiltered_data)

#activation function should restrict domain, but also simple to calculate
relu = lambda x: x if x>0 else 0


#turns a 1d array into the format needed for the brain
def _1d_brain(x):
  global index
  index = -1
  #print(index)
  def next():
    global index
    index = index + 1
    #print(x[index])
    return x[index]

  brain = []
  brain.append([[[next() for i in range(input_size)],next()] for i in range(neurons)])
  #print("reach")
  for i in range(1,layers):
    brain.append([])
    for j in range(neurons):
      brain[i].append([[next() for i in range(neurons)],next()])
  #print("reach")
  brain.append([[[next() for i in range(neurons)],next()] for i in range(output_size)])
  return brain

#turns any multidimensional array into a 1d array
def flatten(data):
  flat_list = []
  for item in data:
    if isinstance(item, (list, tuple)):
      flat_list.extend(flatten(item))
    else:
      flat_list.append(item)
  return flat_list

#matrix multiplication for a single cell
def matmul(a,b):
  return sum([a[i]*b[i] for i in range(len(a))]) if len(a) == len(b) else None

#global count is to see how many times the function is run for training
count = 0
def runbrain(brain, input):
  global count
  count+=1
  if count%10000 == 0:
    print(count)
  isfirst = True
  for layer in range(len(brain)-1):
    if isfirst:
      #print(len(input))
      output = [relu(matmul(input,neuron[0]) + neuron[1]) for neuron in brain[layer]]
      isfirst = False
    else:
      output = [relu(matmul(output,neuron[0]) + neuron[1]) for neuron in brain[layer]]
  return [relu(matmul(output,neuron[0]) + neuron[1]) for neuron in brain[-1]]

def run1dbrain(brain, input):
  return runbrain(flatten(brain), input)

def get_original_number(output):
  max = None
  for i in range(len(output)):
    if max == None or output[i] > max:
      max = output[i]
      original_number = i
  return original_number

def error(output, expected):
  #print(output)
  #print(expected)
  output = [i/sum(output) for i in output]
  return sum([(output[i]-expected[i])**2 for i in range(len(output))])

def run1dbrainerror(brain,index):
  return error(runbrain(_1d_brain(brain.tolist()), data[index]),labels[index])

def show_digit(data):
  data = np.array(data)
  one_image = data.reshape(28, 28)
  plt.imshow(one_image, cmap=plt.cm.gray)
  plt.show()

brain =  _1d_brain([r.uniform(-1,1) for i in range((input_size+1)*neurons + neurons*(neurons+1)*(layers-1) + (neurons+1)*output_size)])
print(brain)


#learning processes
learning_rate = 10
for i in range(20):
  sum_grad = [0 for i in range(len(flatten(brain)))]
  for j in range(20):
    input = r.randint(0,len(data)-1)

    show_digit(data[input])
    print(get_original_number(runbrain(brain,data[input])))
    print(error(runbrain(brain,data[input]),labels[input]))
    print(i,j)
    def fun(brain):
      return run1dbrainerror(brain,input)
    grad2 = nd.Gradient(fun)(flatten(brain))
    sum_grad = [sum_grad[k] + grad2[k] for k in range(len(grad2))]
  sum_grad = [i/10 for i in sum_grad]
  flatbrain = flatten(brain)
  for i in range(len(flatbrain)):
    flatbrain[i] -= learning_rate*sum_grad[i]
  brain = _1d_brain(flatbrain)
  print(brain)
  learning_rate = learning_rate*0.7

print(brain)