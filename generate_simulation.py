from simulation import simulation
import numpy as np
import json

def write(text, arg="w"):
    with open("data/info.csv", arg) as file:
        file.write(f"{text}\n")
        file.close()

write("P, g_sur_l")

i = 0
n = 5 # soit environ 50min
# générer tous ces tests va prendre n*n*2min
for P in np.linspace(-0.1, 0.1, n):
    for g_sur_l in np.linspace(0.1, 0.8, n):
        write(f"{P}, {g_sur_l}", "a+")
        kwargs = {"P":P, "g_sur_l":g_sur_l, "phi":50, "alpha":1e-2, "rho":1e-3, "pi":1}
        sim = simulation(**kwargs)
        for _ in range(6000):
            sim.simulation()
        json.dump(sim.X, open(f'data/{i}.json', 'w'))
        i += 1
