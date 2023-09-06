import math
import random
import numpy as np
import matplotlib.pyplot as plt

def funkcija(x,y):
    return 3*x**2 + y**4

def generisi_jedinku():
    x = random.uniform(-10,10)
    y = random.uniform(-10, 10)
    return [x,y,funkcija(x,y)]

def generisi_pocetnu_populaciju(velicina_populacije):
    pop = [[0,0,0]]*velicina_populacije
    for i in range(0,velicina_populacije):
        pop[i] = generisi_jedinku()
    return pop

def turnirska_selekcija(populacija,K):
    k_torka = random.sample(populacija,K)
    a = [-1]*K
    for i in range(0,K-1):
        a[i] = k_torka[i][2]
    najbolji_indeks = np.argmin(a)
    return k_torka[najbolji_indeks]

def kum_sum(a):
    ks = []
    j = 0
    for i in range(0, len(a)):
        j += a[i]
        ks.append(j)
    return ks

def prosek_generacije(populacija):
    a = [0]*len(populacija)
    for i in range(len(populacija)):
        a[i] = populacija[i][2]
    return np.sum(a)/len(a)

def ruletska_selekcija(populacija):
    a = [0]*len(populacija)
    for i in range(0,len(populacija)):
        a[i] = 1/(populacija[i][2])
    population_fitness = np.sum(a)
    kumulativna_suma = kum_sum(a)
    r = random.uniform(0,population_fitness)
    for i in range(len(a)):
        if r < kumulativna_suma[i]:
            return populacija[i]

def ukrstanje(r1,r2):
    d1 = [r1[0], r2[1], funkcija(r1[0], r2[1])]
    d2 = [r1[1], r2[0], funkcija(r1[1], r2[0])]
    return d1,d2

def elitizam(populacija):
    a = [0]*len(populacija)
    for i in range(len(populacija)):
        a[i] = populacija[i][2]
    index_prvi = np.argmin(a)
    a[index_prvi] = 1000000
    index_drugi = np.argmin(a)
    return index_prvi, index_drugi

def mutacija(jedinka,stepen):
    for i in range(len(jedinka)-2):
        s = random.random()
        if s < stepen:
            jedinka[i]  = jedinka[i] + random.uniform(-1,1)

def genetski_algoritam(selekcija ,velicina_populacije=20, broj_generacija=100,K=3, stepen_mutacije=0.2):
    populacija = generisi_pocetnu_populaciju(velicina_populacije)
    fitness_najbolje_jedinke = []*broj_generacija
    prosecan_fitness_generacije = []
    prosecan_fitness_generacije.append(prosek_generacije(populacija))
    el1, el2 = elitizam(populacija)
    fitness_najbolje_jedinke.append(populacija[el1][2])
    print("Pocnetna opulacija: ")
    for i in range(0,len(populacija)):
        print(populacija[i])
    for i in range(1,broj_generacija-1):
        nova_populacija = []*velicina_populacije
        el1,el2 = elitizam(populacija)
        fitness_najbolje_jedinke.append(populacija[el1][2])
        nova_populacija.append(populacija[el1])
        nova_populacija.append(populacija[el2])
        while len(nova_populacija) < len(populacija):
            if selekcija == "turnir":
                roditelj1 = turnirska_selekcija(populacija, K)
                roditelj2 = turnirska_selekcija(populacija, K)
            else:
                roditelj1 = ruletska_selekcija(populacija)
                roditelj2 = ruletska_selekcija(populacija)
            d1,d2 = ukrstanje(roditelj1,roditelj2)
            mutacija(d1,stepen_mutacije)
            mutacija(d2, stepen_mutacije)
            nova_populacija.append(d1)
            nova_populacija.append(d2)
        prosecan_fitness_generacije.append(prosek_generacije(populacija))
        populacija = nova_populacija
        print(" -------------- Populacija: " + str(i) + '------------------')
        for j in range(0,len(populacija)):
            print(populacija[j])
        if i > 7:
            if abs(fitness_najbolje_jedinke[i] - fitness_najbolje_jedinke[i-6]) < 0.001:
                print(fitness_najbolje_jedinke)
                return fitness_najbolje_jedinke, prosecan_fitness_generacije
    print(fitness_najbolje_jedinke)
    return fitness_najbolje_jedinke, prosecan_fitness_generacije


grafik1, grafik2 = genetski_algoritam("turnir", broj_generacija=30)
plt.plot(grafik1)
#plt.plot(grafik2)
plt.show()
