# Here I should take out the metrics that should matter 


# Metrics that should matter 
#  comparison between the heuristics 
#  what heuristic coarsenes the best 
#  frequency  of the coarsening 
# frequency of the coarsening 
# I need to define all the heuristics comparison metrics
# results is a list of dictionaries each representing a heuristic


def minimal_set_intersection(results,heuristic_names):
    intersection_matrix=[[[] for j in range (len(heuristic_names))] for i in range(len(heuristic_names))]
    for heuristic_name in heuristic_names:
        # compter combien de neurones il y a en commun 
        # like a confusion matrix









def heuristic_statistics_summary():
    return




def print_summary_comparison(runtimes):
    # how many times / runtimes did the heuristic produce a minimal set 
    return

# the coarsened set that gives the highest logit (safest)
# the minimal 
# timeouts 

def analyze_coarsened_set():
    return

# The neurons that remain in the minimal set  across  a lot of heuristics 
def get_always_kept_neurons():
    # the neurons that aren t abstracted always
    return 