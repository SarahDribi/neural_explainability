# 
# here I should implement the optimistic approach algorithm
# this algorithm gives an upper bound of the neurons that should be included in the 
# minimal nap
"""
 For instance, for label 0,OptAdvPrunedetects 618 essential neurons and correctly identify
 445 and 160 of the 480 neurons in the minimal NAP found by Coarsen.
"""

# the intuition behind it is to consider all the adversarial examples of a certain class
# for each xj of a class there are many xj' s ,find the neurons for wich the  activation pattern disagrees with the activation pattern 
#of the rest of the xj 's 
# the set we find is all those important neurons
# I need to test the optimistic approach 



