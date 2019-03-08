#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:55:52 2018

@author: koushikahamed
"""
import numpy as np

class DNN:
    def __init__(self):
        
        self.initial_layer=np.array([[.1,.2,.7]])
        self.output_target=np.array([[1.0,0.0,0.0]])
        self.Wij=np.random.rand(3,3)
        self.Wjk=np.random.rand(3,3)
        self.Wkl=np.random.rand(3,3)
        self.hidden1_net=np.array([[0,0,0]])
        self.hidden1_out=np.array([[0,0,0]])
        self.hidden2_net=np.array([[0,0,0]])
        self.hidden2_out=np.array([[0,0,0]])        
        self.output_net=np.array([[0,0,0]])
        self.output_out=np.array([[0,0,0]])
        self.biash1=np.array([[.20,.20,.20]])
        self.biash2=np.array([[.30,.30,.30]])
        self.biasOutput=np.array([[.50,.50,.50]])
        self.learning_rate=.01
        self.Error=np.array([[0.0,0.0,0.0]])

    #feed forward your network    
    def feed_forward(self):
        #initial to first hidden layer net
       self.hidden1_net=np.dot(self.initial_layer,self.Wij)+self.biash1
       #initial to first hidden layer out
       self.hidden1_out=np.array([max(0,a) for a in self.hidden1_net.tolist()])
       #hidden1 to hidden2 layer net
       self.hidden2_net=np.dot(self.hidden1_out,self.Wjk)+self.biash2
       #hidden1 to hidden2 layer out
       self.hidden2_out=np.array([(1/(1+np.exp(-z))) for z in self.hidden2_net])
       #hidden2 to output 1 net
       self.output_net=np.dot(self.hidden2_out,self.Wkl)+self.biasOutput
       #out of output 
       self.x=[np.exp(i) for i in self.output_net]
       self.output_out=np.array([ i/np.sum(self.x) for i in self.x])

    #cross entropy function   
    def cross_entropy(self):
        err=self.output_target*[np.log(i) for i in self.output_out]+(1-self.output_target)*[np.log(1-i) for i in self.output_out]
        print self.output_out
        print "Error:"+str(-np.sum(err))
        
    #backward propagation
    def back_propagation(self):
        '''
        output layer to hiddenlayer2 backward propagation
         δE1/δOout1 =[-1((Y1*1/Oout1)+(1-Y1)*(1/(1-Oout1)))]
        '''
        self.EbyOout=np.zeros((3,1),dtype=np.float128)

        for i in range(self.EbyOout.size):            
           self.EbyOout[i][0]=-1*((self.output_target[0][i]*(1/self.output_out[0][i]))+(1-self.output_target[0][i])*(1/(1-self.output_out[0][i])))    

        #partial derivative δOout/δOin
        
        self.OoutByOin=np.array([[np.exp(self.output_net[0,0])*(np.exp(self.output_net[0,1])+np.exp(self.output_net[0,2]))/(np.sum(self.x)**2)],
                                   [np.exp(self.output_net[0,1])*(np.exp(self.output_net[0,0])+np.exp(self.output_net[0,2]))/(np.sum(self.x)**2)],
                                   [np.exp(self.output_net[0,2])*(np.exp(self.output_net[0,0])+np.exp(self.output_net[0,1]))/(np.sum(self.x)**2)]],dtype=np.float128)

        #partial derivative δOin/δW
        self.OinByW=np.zeros((3,1),dtype=float)
        for i in range(self.OinByW.size):
            self.OinByW[i,0]=self.hidden2_out[0,i]
        #create δGWkl
        self.GWkl=np.zeros((3,3),dtype=float)
        for i in range(self.EbyOout.size):
            for j in range(self.EbyOout.size):
               self.GWkl[i,j]=  self.EbyOout[j][0]*self.OoutByOin[j,0]*self.OinByW[i,0]
               
        #update weight W'kl
        self.UWkl=np.zeros((3,3),dtype=float)
        for i in range(3):
            for j in range(3):
                self.Wkl[i,j]=self.Wkl[i,j]-self.learning_rate*self.GWkl[i,j]
                
        #hiddenlayer2 to hiddenlayer1 backward propagation    
        #δh2out/δh2in partial derivative
        self.h2outByh2in=np.zeros((3,1),dtype=float)
        for i in range(self.h2outByh2in.size):
            self.h2outByh2in[i,0]=self.hidden2_out[0,i]*(1-self.hidden2_out[0,i])
  
        #δh2in/δW partial derivative
        self.h2inByW=np.zeros((3,1),dtype=float)
        for i in range(self.h2inByW.size):
            self.h2inByW[i,0]=self.hidden1_out[0,i]
            
        # δEtotal/δh2out
        self.EtotalByh2out=np.zeros((3,1),dtype=float)
        self.EtotalByh2out[0][0]=0
        for i in range(self.EtotalByh2out.size):
            for j in range(self.EtotalByh2out.size):
                self.EtotalByh2out[i,0]+=self.EbyOout[j,0]*self.OoutByOin[j,0]*self.Wkl[i,j]
        
        # create δGWjk  
        self.GWjk=np.zeros((3,3),dtype=float)
        for i in range(3):
            for j in range(3):
                self.GWjk[i,j]=self.EtotalByh2out[j][0]*self.h2outByh2in[j,0]*self.h2inByW[i,0]

        #update weight W'jk
        self.UWjk=np.zeros((3,3),dtype=float)
        for i in range(3):
            for j in range(3):
                self.Wjk[i,j]=self.Wjk[i,j]-self.learning_rate*self.GWjk[i,j]

        #δh1out/δh1in partial derivative
        self.h1outByh1in=np.zeros((3,1),dtype=float)
        for i in range(self.h1outByh1in.size):
           if self.hidden1_net[0,i]>0:
              self.h1outByh1in[i,0]=1
           else:
               self.h1outByh1in[i,0]=0
 
        #δh1in/W partial derivative
        self.h1inByW=np.zeros((3,1),dtype=float)
        self.h1inByW=self.initial_layer.T
        #δEtotal/δh1out
        self.EtotalByh1out=np.zeros((3,1),dtype=float)
        for i in range(3):
            for j in range(3):
               self.EtotalByh1out[i,0]+=self.EtotalByh2out[i,0]*self.h2outByh2in[j,0]*self.Wjk[i,j]
        # create δGWij 
        self.GWij=np.zeros((3,3),dtype=float)
        for i in range(3):
            for j in range(3):
                self.GWij[i,j]=self.EtotalByh1out[j][0]*self.h1outByh1in[j,0]*self.h1inByW[i,0]
        #update weight W'ij
        self.UWij=np.zeros((3,3),dtype=float)
        for i in range(3):
            for j in range(3):
                self.Wij[i,j]=self.Wij[i,j]-self.learning_rate*self.GWij[i,j]   
        
    # logitic function
    def logistic(self,z):
        return 1/1+np.exp(-z)
    
    def train_data(self):
        for i in range(1000):
            self.feed_forward()
            self.back_propagation()
        self.cross_entropy()

        
if __name__=='__main__':
    NN=DNN()
    NN.train_data()