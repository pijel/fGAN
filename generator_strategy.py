import random
class RandomizedCollection(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.array = []
        self.dic = {}
        

    def insert(self, val):
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        self.array.append(val)
        if val not in self.dic:
            self.dic[val] = set([len(self.array)-1])
            return True
        else:
            self.dic[val].add(len(self.array)-1)
            return False

    def remove(self, val):
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.dic:
            return False
        
        index = self.dic[val].pop()
        self.array[index] = self.array[-1]
        self.dic[self.array[-1]].add(index)
        self.dic[self.array[-1]].remove(len(self.array)-1)
        self.array.pop()
        
        if len(self.dic[val]) == 0:
            del self.dic[val]
        
        
        return True
        
        

    def getRandom(self):
        """
        Get a random element from the collection.
        :rtype: int
        """
        return self.array[random.randint(0,len(self.array)-1)]



#import random
def generator_mixture(generator_collection, output_size):
    
    #generator_collection should be a Randomized Collection that contains only generators.
    output = []
    gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
    for i in range(output_size):
        #get a random generator       
        chosen_generator = generator_collection.getRandom()
        index = random.randint(0,output_size-1)
        chosen_generator.zero_grad()
        generated_data = chosen_generator(gen_input)
        output.append((generated_data)[index])

    #arbitrary design choice - I use a random index, so it is possible to repeat. I thought this would be more "in tune" with how we sampled generators, but it can be reworked so it doesn't happen
        
    return torch.cat(output,dim = 0)
    #the "dim" input here is very specific to what we are generating. it won't always be 0, but it should be known before hand. it's 0 here because we are generating from a 1D distribution
    
     
    