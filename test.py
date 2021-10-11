import random
options = "ran"
n = 19
lowerbound = 1
upperbound = 10

int_tuples = [(lowerbound, lowerbound)] * n

for option in options.split('-'):
    
    # ran: Set each value to a random one within lowerbound/upperbound
    if option == "ran":
        for i in range(len(int_tuples)):
            int_tuples[i] = (random.randrange(lowerbound, upperbound), random.randrange(lowerbound, upperbound))
    
print(int_tuples)
