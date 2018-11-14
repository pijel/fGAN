#trying a mix of 2 1d MOGS this time.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Data params
data_mean = 4
data_stddev = 0.2
data_mean2 = 2
data_stddev2 = 0.1

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
d_steps = 2  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 2

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

window_size = 5
#placeholder value

D_queue = []
G_queue = []
#initialize the Queue with a fresh D,G
first_discrim = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
first_gen = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)   

D_queue.append(first_discrim)
G_queue.append(first_gen)

import random
def generator_strategy(gen_array,output_size):
    #this time we are passed an array of generators
    #the idea will be that they generate once, and then we sample
    
    #this is when they generate
    
    fake_data_array = []
    gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
    for i in gen_array:
        i.zero_grad()
        new_data = i(gen_input)
        fake_data_array.append(new_data)
        
    #fake_data_array should contain all generator's fake data. now, we need to sample
    out = []
    for i in range(output_size):
        
        #choose a random generator
        
        generator_choice = random.randint(0,len(gen_array)-1)
        index = random.randint(0,output_size - 1)
        
        out.append([fake_data_array[generator_choice][index]])
    
    #!!! dim is only 0 for this 1d mog !!!
    return torch.tensor((out))

def discriminator_strategy(dis_array,faked_data):
    #dis_collection is an array of discriminators
    #faked_data is whatever the generator we are trying to train has come up with
    
    
    #take discrimator out
    
    decisions = []
    
    for i in dis_array:
        i.zero_grad()
        decisions.append(i(preprocess(faked_data.t())))
        
    return sum(torch.cat(decisions,dim = 0))/(len(dis_array))

for i in range(50):
    
    #stopping condition is when the newest discrimator outputs real/fake with .5 + epsilon probability. Not sure how to code that. Or, it could just be more
    #simply a set number of epochs.  I'll leave this ambiguous for now.

    D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
    #make a new discrimator, to train against each of the generators
    gi_sampler = get_generator_input_sampler()
    d_sampler = get_distribution_sampler(data_mean, data_stddev)
    d_sampler2 = get_distribution_sampler(data_mean2, data_stddev2)
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
    
    


       
    for d_index in range(1000):
         # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = torch.cat((Variable(d_sampler(d_input_size//2)),Variable(d_sampler2(d_input_size//2))),1)
        d_real_decision = D(preprocess(d_real_data))
        d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params

        #  1B: Train D on fake
        
        d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_decision = D(preprocess(generator_strategy(G_queue, d_input_size).t()))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
      



    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)   
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)  

    for g_index in range(10000):
        # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()
            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            g_fake_data = G(gen_input)
            #print(g_fake_data) # Rahul - the genrators show no signs of training. Check out this line to see how generic all outputs are 
            dg_fake_decision = discriminator_strategy(D_queue,g_fake_data)
            g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters



    D_queue.append(D)
    
    G_queue.append(G)
    
    
    if len(D_queue)>window_size:
        D_queue.pop(0)
        
    if len(G_queue) > window_size:
        G_queue.pop(0)
        
    
print("Training has been completed")



import matplotlib.pyplot as plt

#final_generator = G_queue.last.val
#final_discrim = D_queue.last.val


#final_generator.zero_grad()
#gi_sampler = get_generator_input_sampler()
#gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
g_fake_data = (generator_strategy(G_queue,d_input_size))
fake_sample = g_fake_data.data.numpy()

d_sampler = get_distribution_sampler(data_mean, data_stddev)
d_sampler2 = get_distribution_sampler(data_mean2, data_stddev2)
d_real_data = Variable(d_sampler(d_input_size))
real_data = torch.cat((Variable(d_sampler(d_input_size//2)),Variable(d_sampler2(d_input_size//2))),1)

print(fake_sample)
print(real_data)

plt.scatter(fake_sample[:,0],np.zeros(minibatch_size), label='Fake', marker='+', color='r')
plt.scatter(real_data,np.zeros(minibatch_size),label='Real',marker = '+',color = 'b')
plt.legend()
plt.show()
