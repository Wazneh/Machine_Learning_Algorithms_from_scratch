# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 19:41:27 2016

@author: Elie Kawerk
"""
import numpy as np
import numpy.random as rd

class lin_reg(object):
    
    def __init__(self,eta=10**(-2)):
        self.eta = eta
        self.eps = 10**(-14)
    
    def fit(self,Xin,y):
        converge = False
        X = self.initialize_X_(Xin)
        self.initialize_w_(X.shape[1])
        count = 0
        self.cost_ =[]
        while not converge:
            self.GD(X,y)    
            self.compute_cost_(X,y)
            if count > 1 and abs(self.cost_[count]-self.cost_[count-1])<self.eps:
                converge = True
            count += 1
             
    def initialize_X_(self,Xin):
        Xtemp = np.zeros((Xin.shape[0],Xin.shape[1]+1))
        for xx,xi in zip(Xtemp,Xin):
            xx[0] = 1.0 ; xx[1:] = xi
        return Xtemp
       
    def initialize_w_(self,m):
        self.w_ = rd.rand(m)
        self.w_ = np.reshape(self.w_,(m,1))
        
    def GD(self,X,y):
        m = len(y)
        error = self.compute_error_(X,y)
        self.w_ -=  (self.eta/m) * X.T.dot(error)
        
    def compute_cost_(self,X,y):
        error = self.compute_error_(X,y)
        cost = (error.T.dot(error))/(2*y.shape[0])
        cost = np.reshape(cost,())
        self.cost_.append(cost)

    def compute_error_(self,X,y):
        error = np.zeros((y.shape[0],1))   
        for i,x in enumerate(X):
            error[i] = self.ht(x) - y[i]
        return error
                
    def ht(self,x):
        w = np.reshape(self.w_,(self.w_.shape[0]))
        return np.dot(w,x)        

    def predict(self,x):
        x=np.reshape(x, (x.shape[0],1))
        x=self.initialize_X_(x).T
        x = np.reshape(x,(x.shape[0]))
        return self.ht(x)
    