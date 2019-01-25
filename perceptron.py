#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:53:05 2018

@author: Gabriel Vieira
"""
import numpy as np
    
#AND
#data = np.array([[0,0], [0,1], [1,0], [1,1]])
#outputs = np.array([0, 0, 0, 1])

#OR
data = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([0, 1, 1, 1])

#XOR (não encontra o padrão com rede neural de uma camada / Linearmente não separável)
#data = np.array([[0,0], [0,1], [1,0], [1,1]])
#outputs = np.array([0, 1, 1, 0])

synapses = np.array([0.0, 0.0])
learning_rate = 0.1

def step_function(output_result):
    """
    Função de ativação - retorna 0 ou 1
    """
    if output_result >= 1:
        return 1
    else:
        return 0
    
def calculate_output(register):
    """
    Função de somatório
    """
    result = register.dot(synapses)
    return step_function(result)

def train():
    """
    Função de treinamento e ajuste de sinapses
    """
    print('Adjusting neural synapses ...')
    errors = 1
    
    while errors != 0:
        errors = 0
        
        for i in range(len(outputs)):
            print('Analyzing data {} ...'.format(i + 1))
            calculated_value = calculate_output(data[i])
            error = abs(outputs[i] - calculated_value)
            errors += error
            
            if error > 0:
                for j in range(len(synapses)):
                    synapses[j] = synapses[j] + (learning_rate * data[i][j] * error)
                    if j == 0:
                        print('Balancing synapses ...')
                        print('Synapse updated: ' + str(synapses[j]))
        
        if errors != 0:
            print('Error encountered, training perceptron ...')
        else:
            print('Errors not found.')
        
train()
print('Training completed!')

for i in range(len(outputs)):
    print('\n{}'.format(calculate_output(data[i])))
