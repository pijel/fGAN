def discriminator_strategy(dis_array,faked_data):
    #dis_collection is an array of discriminators
    #faked_data is whatever the generator we are trying to train has come up with
    
    
    #take discrimator out
    
    decisions = []
    
    for i in dis_array:
        i.zero_grad()
        decisions.append(i(preprocess(faked_data.t())))
        
    return sum(torch.cat(decisions,dim = 0))/(len(dis_array))
