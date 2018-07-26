class DoubleListNode:
    def __init__(self,x):
        self.val = x
        self.prev = None
        self.next = None
class kQueue:
    def __init__(self,size):
        self.first = None
        self.last = None
        self.limit = size
        self.size = 0
    def add(self,value):
        new_node = DoubleListNode(value)
        if self.size == 0:
            self.first = new_node
            self.last = new_node
        else:
            new_node.next = self.last
            self.last.prev = new_node
            self.last = new_node
        if self.size >= self.limit:
            self.remove()
        self.size +=1
     
    def remove(self):
        if self.size == 0:
            return False
        return_value = self.first.val
        self.first = self.first.prev
        self.size -=1
        return return_value
        
    def peek(self):
        return self.first.val
    def __str__(self):
        a = self.first
        num_eles = 0
        while a:
            print (a.val)
            a = a.prev
            num_eles +=1
        return 'End of Queue, {} elements found'.format(num_eles)
    def nonempty(self):
        return (self.first != None and self.last != None)







#!/usr/bin/env python

# Generative Adversarial Networks (GAN) example in PyTorch.
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Data params
data_mean = 4
data_stddev = 0

# Model params
g_input_size = 1     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 1    # size of generated output vector
d_input_size = 100   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
minibatch_size = d_input_size

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 400
print_interval = 10
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1

# ### Uncomment only one of these
#(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

print("Using data [%s]" % (name))

# ##### DATA: Target data and generator input data

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)

q_size = 3 
#placeholder value

D_queue = kQueue(q_size)
G_queue = kQueue(q_size)
#initialize the Queue with a fresh D,G
first_discrim = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
first_gen = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)   

D_queue.add(first_discrim)
G_queue.add(first_gen)



#It might be the case that the gaussian distribution is changing everytime with new calls of the get_distribution_sampler function,
#which is decidedly bad. 
#Never mind, it seems that it's initialized to the same thing every time. It could be useful to just store it memory to save computing time?

for i in range(10):
    
    #stopping condition is when the newest discrimator outputs real/fake with .5 + epsilon probability. Not sure how to code that. Or, it could just be more
    #simply a set number of epochs.  I'll leave this ambiguous for now.

    D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
    #make a new discrimator, to train against each of the generators
    G_size = G_queue.size

    for i in range(G_size):
    #this "for" is so we don't run through the elements in the queue more than one time
        G = G_queue.remove()
        #print(G)
        #take the top element of the queue. Now, do just the discrimator training from as given in the medium article code


        gi_sampler = get_generator_input_sampler()
        d_sampler = get_distribution_sampler(data_mean, data_stddev)
        #G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
        #D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
        criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
        d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
        g_optimizer = optim.Adam(G.parameters(), lr= 0, betas=optim_betas)
        #the learning rate here is 0, we don't want G to train. I know there is no code here but it is for symmetry purposes

        for epoch in range(num_epochs):
            for d_index in range(d_steps):
                # 1. Train D on real+fake
                D.zero_grad()

                #  1A: Train D on real
                d_real_data = Variable(d_sampler(d_input_size))
                d_real_decision = D(preprocess(d_real_data))
                d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
                d_real_error.backward() # compute/store gradients, but don't change params

                #  1B: Train D on fake
                d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
                d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_decision = D(preprocess(d_fake_data.t()))
                d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
                d_fake_error.backward()
                d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
        #print("REACHED")
        G_queue.add(G)
        #add the element to the end of the queue so the structure is preserved. Note that due to the fixed length of the queue, this command will implicitly remove
        #elements as we go to the queue_size variable


    D_queue.add(D)
    #add the newly trained element back into the discrimator queue

    #the modifications made below are largely the same as above. Maybe it's a good idea to also modify the print statement below, in order to get more relevant metrics 
    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)   
    D_size = D_queue.size

    for i in range(D_size):

        D = D_queue.remove()
        gi_sampler = get_generator_input_sampler()
        d_sampler = get_distribution_sampler(data_mean, data_stddev)
        criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
        d_optimizer = optim.Adam(D.parameters(), lr=0, betas=optim_betas)
        #once again, learning rate is 0. We don't want D to train.
        g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
        for epoch in range(num_epochs):
            for d_index in range(d_steps):
                # 1. Train D on real+fake
                D.zero_grad()

                #  1A: Train D on real
                d_real_data = Variable(d_sampler(d_input_size))
                d_real_decision = D(preprocess(d_real_data))
                d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
                d_real_error.backward() # compute/store gradients, but don't change params

                #  1B: Train D on fake
                d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
                d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_decision = D(preprocess(d_fake_data.t()))
                d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
                d_fake_error.backward()
                d_optimizer.step()

            for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
                G.zero_grad()

                gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
                g_fake_data = G(gen_input)
                dg_fake_decision = D(preprocess(g_fake_data.t()))
                g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine

                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters

            #if epoch % print_interval == 0:
                #print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                                    #extract(d_real_error)[0],
                                                                    #extract(d_fake_error)[0],
                                                                    #extract(g_error)[0],
                                                                    #stats(extract(d_real_data)),
                                                                    #stats(extract(d_fake_data))))

        D_queue.add(D)
        #print("REACHED")
    G_queue.add(G)
    
print("Training has been completed")
import matplotlib.pyplot as plt

final_generator = G_queue.last.val
final_discrim = D_queue.last.val


final_generator.zero_grad()
gi_sampler = get_generator_input_sampler()
gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
g_fake_data = final_generator(gen_input)
fake_sample = g_fake_data.data.numpy()

d_sampler = get_distribution_sampler(data_mean, data_stddev)
d_real_data = Variable(d_sampler(d_input_size))
real_data = d_real_data.data.numpy()

print(fake_sample)
print(real_data)

plt.scatter(fake_sample[:,0],np.zeros(minibatch_size), label='Fake', marker='+', color='r')
plt.scatter(real_data,np.zeros(minibatch_size),label='Real',marker = '+',color = 'b')
plt.legend()
plt.show()
