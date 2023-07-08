import os
from matplotlib import pyplot as plt
os.chdir("C:/Users/Aleksander/Desktop/TFG/Script Lua modificado/1942/")

generations = []
maxFitness = []
numberOfSpecies = []

for i in range(1, 101):
    with open(f"backup.{i}.1942.state.Pool", 'r') as file:
        pool = file.read()
    
    pool = pool.split('\n')
    
    generations.append(pool[0])
    maxFitness.append(pool[1])
    numberOfSpecies.append(pool[2])

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.plot(generations, maxFitness)
ax2.plot(generations, numberOfSpecies)
ax1.set_xlabel("Generations")
ax2.set_xlabel("Generations")
ax1.set_xticks(range(1, 100, 10))
ax2.set_xticks(range(1, 100, 10))
ax1.set_ylabel('Max Fitness')
ax2.set_ylabel('Number of Species')
fig.suptitle("Elitism 3 x4 speed segunda vuelta")
plt.show()