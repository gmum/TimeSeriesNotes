# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:52:41 2016

@author: Maciek
"""

from pyneurgen.neuralnet import NeuralNet
from pyneurgen.recurrent import NARXRecurrent
import numpy as np

def Weierstrass_at_x(a, b, x, n):
    suma = 0
    a_i = 1
    b_ix = np.pi*x
    for i in xrange(n):
        suma = suma + a_i*np.cos(b_ix)
        a_i = a_i*a
        b_ix = b_ix*b
    return suma
    
def Weierstrass(a, b, arr, n = 100):
    return np.array(map(lambda x: Weierstrass_at_x(a, b, x, n), arr))
    
def simpleWeierstrassTimeSeries(arr):
    a = np.random.uniform(0.1, 1)
    b = np.random.randint(7, 1000)
    while(True):
        if (b%2 == 1 or a*b > 1 + 3/2*np.pi):
            break
    return Weierstrass(a, b, arr)


input_nodes = 1
hidden_nodes = 5
output_nodes = 1

output_order = 20
incoming_weight_from_output = .5
input_order = 20
incoming_weight_from_input = .5

net = NeuralNet()
net.init_layers(input_nodes, [hidden_nodes], output_nodes,
        NARXRecurrent(
            output_order,
            incoming_weight_from_output,
            input_order,
            incoming_weight_from_input))

net.randomize_network()

X = np.linspace(0, 10.0, num=10001)
Y = simpleWeierstrassTimeSeries(X)
Y = Y.reshape(-1, 1)

net.set_all_inputs(Y[:-1])
net.set_all_targets(Y[1:])

net.set_learn_range(0, 8000)
net.set_test_range(8000, 9999)

print net.test()
net.learn(epochs=5)
print net.test()