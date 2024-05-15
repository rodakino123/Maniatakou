#main libs
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import argsort
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
#extra plot libs
from numpy import arange
from numpy import square
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import random
import statistics
import math

# fitness plot rastrigin multimodal function
def fitness_rastrigin_plot(x,y):
     return (x**2 - 10 * cos(2 * pi * x)) + (y**2 - 10 * cos(2 * pi * y)) + 20

# fitness function
def fitness(v):
     x, y = v   
     return (x**2 - 10 * cos(2 * pi * x)) + (y**2 - 10 * cos(2 * pi * y)) + 20
 
# Ελέγχος, εάν ένας υποψήφιος ή παιδιού είναι εντός ορίων
def in_limits(candidate, limits):
    # LOOP στα όρια για έλεγχο υποψήφιου ή παιδιού
    for d in range(len(limits)):
        # Ελέγχος αν είναι εκτός ορίων
        if candidate[d] < limits[d, 0] or candidate[d] > limits[d, 1]:
            return False
    return True

# Evolution Strategy αλγόριθμος με επιλογή για πρόσθεση επιλεγμένων γονέων
def evolution_algorithm(objective, limits, n_iter, step_size, selected_parents, population_size):
     best, best_eval = None, 1e+10
     # υπολογισμός παιδιών ανα γονείς
     n_children = int(population_size / selected_parents)
     # Αρχικοποίηση Πληθυσμού
     population = list()
     # Loop για Πρόσθεση Υποψηφίων βάσει μεγέθους δημιουργούμενου πληθυσμού
     for _ in range(population_size):
          candidate = None 
          while candidate is None or not in_limits(candidate, limits):
               candidate = limits[:, 0] + rand(len(limits)) * (limits[:, 1] - limits[:, 0])
               population.append(candidate)
     # Loop βάσει παραμέτρου επαναλλήψεων
     child_per_gen = list()
     for generation in range(n_iter):
          # Υπολογισμός βαθμολογιών και αποθήκευση σε ξεχωριστή παράλληλη λίστα
          scores = [fitness(c) for c in population]
          # Κατάταξη βαθμολογιών σε αύξουσα σειρά με διπλή κλήση ταξινόμησης
          ranks = argsort(argsort(scores))
          # Επιλογή δεικτών γονέων με την καλύτερη κατάταξη
          selected = [i for i,_ in enumerate(ranks) if ranks[i] < selected_parents]
          # Δημιουργία παιδιών απο γονείς
          children = list()
          for i in selected:
               # ΈΛεγχος αν αυτός ο γονέας είναι η βέλτιστη λύση και εκτύπωση στατιστικών
               if scores[i] < best_eval:
                    best, best_eval = population[i], scores[i]
                    print('%4d, ΚΑΛΥΤΕΡΟ: f(%s) = %.5f' % (generation, best, best_eval))
                    bst_gen = generation
               child_gen = list()
               # Δημιουργία παιδιών απο γονείς
               for _ in range(n_children):
                    child = None
                    while child is None or not in_limits(child, limits):
                         child = population[i] + randn(len(limits)) * step_size
                         children.append(child)

          # Αντικατάσταση πληθυσμού απο παιδιά και γονείς
          population = children
          len_pop = len(population)

     return [bst_gen, len_pop, best, best_eval]

def control():    
     # Παράμετροι για την λειτουργία των εξελικτικών αλγορίθμων
     #    
     # Σπόρος της γεννήτριας ψευδοτυχαίων αριθμών
     seed(1)
     # Oρισμός ελάχιστου και μέγιστου ορίου
     r_min, r_max = -5.0, 5.0
     # Ορισμός συνολικών επαναλλήψεων
     max_iterations = 5000
     # Ορισμός για το μέγιστο μεγέθος βήματος
     step_size = 0.15
     # Αριθμός επιλεγμένων γονέων
     selected_parents = 20
     # Μέγεθος δημιουργούμενου πληθυσμού
     population_size = 100
     # Oρισμός πίνακα ορίων
     limits = asarray([[r_min, r_max], [r_min, r_max]])
     #          
     bst_gen, len_pop, best, score = evolution_algorithm(fitness, limits, max_iterations, step_size, selected_parents, population_size)    
     print('\nΟΛΟΚΛΗΡΩΣΗ!')
     print('ΤΕΛΕΙΟ: f(%s) = %f' % (best, score))

     # Εκτύπωση Διαγραμμάτων
     #
     # 1. Gauss plot
     # Εύρος εισόδου δειγμάτων ομοιόμορφα σε προσαυξήσεις 0,15
     xaxis = arange(r_min, r_max, step_size)
     mean = statistics.mean(xaxis)
     std = statistics.stdev(xaxis) 
     # Υπολογισμός στόχων
     result = norm.pdf(xaxis, mean, std)
     # Εμφάνιση διαγράμματος
     pyplot.plot(xaxis, result)
     pyplot.title("Gauss Graph")
     pyplot.show()

     #2. Rastrigin plot
     # Εύρος εισόδου δειγμάτων ομοιόμορφα σε προσαυξήσεις 0,15
     xaxis = arange(0, bst_gen, step_size)
     yaxis = arange(0, len_pop, step_size)
     # Δημιουργήστε ένα πλέγμα από τον άξονα
     x, y = meshgrid(xaxis, yaxis)
     # Υπολογισμός στόχων
     result = fitness_rastrigin_plot(x, y)
     # Δημιουργία διαγράμματος με σχήμα jet color
     fig = pyplot.figure()
     axis = fig.add_subplot(111, projection='3d')
     axis.plot_surface(x, y, result, cmap='jet')
     # Εμφάνιση διαγράμματος
     axis.set_xlabel('Αριθμός Γενεών')
     axis.set_ylabel('Πληθυσμός')
     #axis.set_zlabel('Z axis')        
     pyplot.title("Rastrigin Graph")
     pyplot.show()
        
if __name__ == "__main__":
     print("\nΕΞΕΛΙΚΤΙΚΟΣ ΑΛΓΟΡΙΘΜΟΣ\n")
     control()

