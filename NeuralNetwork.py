#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:03:49 2018

@author: koushikahamed
"""
import numpy as np
import math
import random
class NN:
    def __init__(self):
        #connectivity among neuron forward feed
        #we can use random for being taking weight using random.random()    range 0-1
       self.connectivity_ffd=np.array([[0,0,.15,.25,0,0],
                              [0,0,.20,.30,0,0],
                              [0,0,0,0,.40,.50],
                              [0,0,0,0,.45,.55],
                              [0,0,0,0,0,0],
                              [0,0,0,0,0,0]]) 
       #connectivity among neuron backward propagation
       self.connectivity_bpg=self.connectivity_ffd.T
       self.bias=1
       self.bias_network={2:.35,3:.35,4:.60,5:.60} #bias weight of this neural network #you can use weight using random.random()
       self.learning_rate=.5
       
       #Every layer perceptron positional value  
       self.initial_layer={0:.05,1:.10}
       self.hidden_layer={2:0,3:0}
       self.output_layer={4:0,5:0}
       
       #net of hidden and output layer perceptron
       self.hidden_layer_net={2:0,3:0}
       self.output_layer_net={4:0,5:0}
       
       #output layer target value
       self.output_target={4:.01,5:.99}
       self.error={4:0,5:0}
       self.Error_Total=0
       
    #feed forward or forward propagation 
    def forward_propagation(self):
       #initial layer perceptron list
       self.initial_layer_perceptron=self.initial_layer.keys()
       #hidden layer perceptron list
       self.hidden_layer_perceptron=self.hidden_layer.keys()
       #output layer perceptron list
       self.output_layer_perceptron=self.output_layer.keys()
       #print self.initial_layer_perceptron,self.hidden_layer_perceptron,self.output_layer_perceptron
       
       #iterate hidden layer 
       for i in self.hidden_layer_perceptron:
           for j in self.initial_layer_perceptron:
               self.hidden_layer[i]+=self.connectivity_ffd[j][i]*self.initial_layer[j]
           self.hidden_layer[i]+=self.bias*self.bias_network[i]
           self.hidden_layer_net[i]=self.hidden_layer[i]
           #self.hidden_layer[i]=self.logistic(self.hidden_layer[i])
           self.hidden_layer.update({i:self.logistic(self.hidden_layer[i])})
           #print self.hidden_layer[i],
       print
       
       #iterate for output layer
       for i in self.output_layer_perceptron:
           for j in self.hidden_layer_perceptron:
               self.output_layer[i]+=self.connectivity_ffd[j][i]*self.hidden_layer[j]
           self.output_layer[i]+=self.bias*self.bias_network[i]
           self.output_layer_net[i]=self.output_layer[i]
           self.output_layer.update({i:self.logistic(self.output_layer[i])})
           #print self.output_layer[i],
       #self.Error_function() 
       #self.backward_propagation()
       
    #back propagation 
    def backward_propagation(self):
        #backward propagation for output layer
        for i in self.output_layer_perceptron:
            for j in self.hidden_layer_perceptron:
                #using chain rule we will pertial derivative total error acoording to different weight
               self.connectivity_bpg[i][j] =self.connectivity_bpg[i][j]-self.learning_rate*\
               ((self.output_layer[i]-self.output_target[i])*\
               (self.output_layer[i]*(1-self.output_layer[i]))*self.hidden_layer[j])
        
        #print self.connectivity_bpg    
        #backward propagation for hidden layer
        sum=0
        for i in self.hidden_layer_perceptron:
            for j in self.initial_layer_perceptron:
                for k in self.output_layer_perceptron:
                    sum+=(self.output_layer[k]-self.output_target[k])*(self.output_layer[k]*(1-self.output_layer[k]))*\
                    self.connectivity_bpg[k][i]
                self.connectivity_bpg[i][j]=self.connectivity_bpg[i][j]-self.learning_rate*\
                (sum*(self.hidden_layer[i]*(1-self.hidden_layer[i]))*self.initial_layer[j])
                sum=0
        #print self.connectivity_bpg
                
    #sigmoid function   
    def logistic(self,z):
        result=1/(1+math.exp(-z))
        return result
    
    #Error function
    def Error_function(self):
        for i in self.output_layer_perceptron:
            self.error[i]=.5*(self.output_target[i]-self.output_layer[i])**2
            self.Error_Total+=self.error[i]
            print self.output_layer[i],
        print '\nTotal Error:'+str(self.Error_Total)
        
if __name__=='__main__':
    NN=NN()
    for i in range(10000):    
        NN.forward_propagation()
        NN.backward_propagation()

    NN.Error_function()
    
    
    
