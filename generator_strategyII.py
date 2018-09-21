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
    
    
    return torch.tensor((out))