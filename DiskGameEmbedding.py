# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math
import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate

class DiscGameEmbed:
    #if abs(x) < machine_epsilon, x = 0 
    machine_epsilon = np.finfo(float).eps
    
    #user defined integrating method for omega omega
    integrating_method_oo = [] 
    integrating_name_oo = []
    
    #user defined integrating method for omega
    integrating_method_o = [] 
    integrating_name_o = []
    
    #decorator for integration
    @staticmethod
    def decorator_omega(f1,f2, pi_x):
        def wrapper_omega(*args):
            x = np.array(args)
            return f1(x)*f2(x)*pi_x(x)
        return wrapper_omega
            
    @staticmethod
    def decorator_omega_omega(f,f1,f2,pi_x):
        def wrapper_omega_omega(*args):
            n = len(args)//2
            x = np.array(args[0:n])
            y = np.array(args[n:2*n])
            return f(x,y)*f1(x)*f2(y)*pi_x(x)*pi_x(y)
        return wrapper_omega_omega
    
    #static method
    @staticmethod
    def AddIntegrationOmeOme(f,name):
        DiscGameEmbed.integrating_method_oo.append(f)
        DiscGameEmbed.integrating_name_oo.append(name)
    
    @staticmethod
    def AddIntegrationOme(f,name):
        DiscGameEmbed.integrating_method.append(f)
        DiscGameEmbed.integrating_name.append(name)
    
    #inner product
    @staticmethod
    def inner_product_omega(f1,f2,pi_x, xmin, xmax,xsample,method):
        if method == "quad":
            return DiscGameEmbed.integrate_omega_quad(f1,f2, pi_x, xmin, xmax)
        #can define other integration method: e.g empirical measure/ normal, user defined measure etc
        if method == "empirical":
            return DiscGameEmbed.integrate_omega_empirical(f1,f2,xsample)
        if method not in DiscGameEmbed.integrating_name_o:
            raise Exception("Invalid integrating method")
        idx = DiscGameEmbed.integrating_name_o.index(method)
        return DiscGameEmbed.integrating_method_o[idx](f1,f2,pi_x,xmin,xmax)
    
    @staticmethod
    def inner_product_omega_omega(f, f1, f2, pi_x, xmin, xmax, method = "quad"):
        if method == "quad":
            return DiscGameEmbed.integrate_omega_omega_quad(f, f1, f2, pi_x, xmin, xmax)
        if method not in DiscGameEmbed.integrating_name_oo:
            raise Exception("Invalid integrating method")
        idx = DiscGameEmbed.integrating_name_oo.index(method)
        return DiscGameEmbed.integrating_method_oo[idx](f,f1,f2,pi_x,xmin,xmax)
        
    #numerical integration
    @staticmethod
    def integrate_omega_quad(f1,f2,pi_x, xmin, xmax):
        integrand = DiscGameEmbed.decorator_omega(f1, f2, pi_x)
        int_range = []
        if type(xmin) == int:
            int_range = [[xmin,xmax]]
        else:
            for i in range(len(xmin)):
                int_range.append([xmin[i],xmax[i]])
        return integrate.nquad(integrand, int_range)[0]
    
    @staticmethod
    def integrate_omega_omega_quad(f,f1,f2,pi_x, xmin, xmax):
        integrand = DiscGameEmbed.decorator_omega_omega(f, f1, f2, pi_x)
        int_range = []
        if type(xmin) == int:
            int_range = [[xmin,xmax],[xmin,xmax]]
        else:
            ## work twice. 1st time for x, second time for y
            for i in range(len(xmin)):
                int_range.append([xmin[i],xmax[i]])
            for i in range(len(xmin)):
                int_range.append([xmin[i],xmax[i]])
        
        I = integrate.nquad(integrand, int_range)[0]
        return I
    
    #Empirical measure
    @staticmethod
    def integrate_omega_empirical(f1,f2,xsample):
        n = len(xsample)
        inner_prod = 0
        for i in range(n):
            inner_prod += f1(xsample[i]) * f2(xsample[i])
        return inner_prod/n
        

    #function creation
    @staticmethod
    def function_sum(f_list, coef_v):
        def f_sum(x):
            temp = 0
            for i in range(len(f_list)):
                temp += coef_v[i]*f_list[i](x)
            return temp
        return f_sum
    
    @staticmethod
    def function_scale(f,factor):
        def f_scale(x):
            return factor*f(x)
        return f_scale
        
    
    #constructor
    #sample should be of sorted of increasing value
    #f_sample is the matrix of samples f(xi, xj). It should be arranged by sorting xi, xj respectively.
    def __init__(self, basis_list, measure, f = 0, xmin = 0, xmax = 0, pi_x = 0, sample = 0, f_sample = 0):
        #'set of orginal basis
        #'@ param basis set of orginal basis
        self.basis = basis_list
        self.f = f
        self.xmin = xmin
        self.xmax = xmax
        self.pi_x = pi_x
        self.basis_orthogonal = []
        self.gram_coef = np.empty(shape = (0,len(basis_list)))
        self.projection = np.empty(shape = (0,0))
        self.discgame_embedding = []
        self.rank = 0
        self.embed_coef = np.empty(shape = (0,0))
        self.embed_coef_ortho = np.empty(shape = (0,0))
        self.v_lambda = np.empty(shape = (0,0))
        self.method = measure 
        self.sample = sample
        self.f_sample = f_sample
    
    #GramSchmidth
    #update basis_orthorgonal and gram_coef
    #gram_coef: row i = basis_orthogonal[i]'s coef 
    def GramSchmidt(self):
        if len(self.basis) == 0:
            raise Exception('There should be at least 1 function basis ')
        self.basis_orthogonal = []
        self.gram_coef = np.empty(shape = (0,len(self.basis)))
        n = len(self.basis)
        row_idx_v = []
        for i in range(n):
            coef_v = np.zeros(n)
            coef_v[i] = 1
            for j in range(len(self.basis_orthogonal)):
                row_idx = row_idx_v[j]
                coef = DiscGameEmbed.inner_product_omega(self.basis[i], self.basis_orthogonal[j], self.pi_x, self.xmin, self.xmax, self.sample, self.method)
                coef_v -= coef*self.gram_coef[row_idx]
            #create the orthogonal basis
            ortho_basis = DiscGameEmbed.function_sum(self.basis, coef_v)
            #check linear independce
            norm = DiscGameEmbed.inner_product_omega(ortho_basis, ortho_basis, self.pi_x, self.xmin, self.xmax,self.sample, self.method)
            norm = math.sqrt(norm)
            
            if norm > DiscGameEmbed.machine_epsilon:
                self.gram_coef = np.vstack((self.gram_coef,1/norm*coef_v))
                self.basis_orthogonal.append(DiscGameEmbed.function_scale(ortho_basis,1/norm))
                row_idx_v.append(i)
                
        return
    
    
    def UpdateProjection(self):
        n = len(self.basis)
        B = np.zeros((n,n))
        if self.method =="quad":
            for i in range(n):
                for j in range(n):
                   B[i][j] = DiscGameEmbed.inner_product_omega_omega(self.f, self.basis[i], self.basis[j], self.pi_x, self.xmin, self.xmax)
        if self.method == "empirical":
            C = np.zeros((len(self.sample),len(self.basis)))
            for i in range(len(self.sample)):
                for j in range(len(self.basis)):
                    C[i][j] = self.basis[j](self.sample[i])
            B = 1/((len(self.sample))**2)*C.T@self.f_sample@C
        self.projection = self.gram_coef@B@self.gram_coef.T
        if self.projection.shape[0]%2 == 1:
            np.pad(self.projection,[(0,1),(0,1)],mode = "constant", constant_values = 0)
        return
    
    def UpdateEmbedding(self):

        #Schur decomp
        T , Q = la.schur(self.projection)
        eigen = np.zeros(T.shape[0]//2)
        #get lambda and drop 0 eigenvalues
        for i in range(T.shape[0]//2):
            eigen[i] = T[2*i,2*i+1]
        eigen = eigen[abs(eigen) > DiscGameEmbed.machine_epsilon*T.shape[0]]
        
        #update rank
        self.rank = len(eigen)
        
        #update lambda
        self.v_lambda = np.sqrt(abs(eigen))
        
        #update embedding
        sort_idx = np.argsort(np.abs(eigen))
        m_coef = np.zeros(Q.shape)
        for i in range(len(sort_idx)):
            idx = sort_idx[i]
            _lambda = eigen[idx]
            temp = 0
            #switch the rows if the top right corner of the block diagonal matrix is negative
            if _lambda < 0:
                temp = math.sqrt(-1*_lambda)
                m_coef[2*i] = temp * Q.T[2*idx+1]
                m_coef[2*i+1] = temp * Q.T[2*idx]
            else:
                temp = math.sqrt(_lambda)
                m_coef[2*i] = temp*Q.T[2*idx]
                m_coef[2*i+1] = temp*Q.T[2*idx+1]
                
        n = len(self.basis_orthogonal)
        self.embed_coef_ortho = m_coef[:,0:n]
        self.embed_coef = self.embed_coef_ortho @ self.gram_coef
        
        #create embedding functions
        for i in range(self.rank):
            x = DiscGameEmbed.function_sum(self.basis, self.embed_coef[2*i])
            y = DiscGameEmbed.function_sum(self.basis, self.embed_coef[2*i+1])
            self.discgame_embedding.append((x,y))
        
        return
    
    def SolveEmbedding(self):
        self.GramSchmidt()
        self.UpdateProjection()
        self.UpdateEmbedding()
        return
    
    
    def EvaluateDiscGame(self,i,x,y):
        if i > self.rank:
            raise Exception("please enter a proper index of disc games. The index should be less than the rank")
        
        if i < 0:
            raise Exception("The index of disc game embedding should be non-negative.")
            
        f1 = self.discgame_embedding[i-1][0]
        f2 = self.discgame_embedding[i-1][1]
        
        return f1(x)*f2(y) - f1(y)*f2(x)
    
    def EvalSumDiscGame(self,i,x,y):
        temp = 0
        for i in range(i):
            temp += self.EvaluateDiscGame(i+1, x, y)
        return temp
    