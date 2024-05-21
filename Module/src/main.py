#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sympy as sym
import numpy as np
from termcolor import colored

def fun_v_jakobian(bulk, var_in_dolžine_slovar, levi_robni, desni_robni, lr_cut, slovar_vrednosti):
    """
    Odvajanje funkcij podane v funkcijski obliki za ustvarjanje jakobiana.

    Parametri:
    - bulk: funkcija, ki sprejme indekse in vrne sympy izraz za enačbe v večspremenljivčnem sistemu
    - var_in_dolžine_slovar: slovar, ki določa število komponent za vsako spremenljivko
    - levi_robni: funkcija, ki vrne izraz za levi robni pogoj
    - desni_robni: funkcija, ki vrne izraz za desni robni pogoj
    - lr_cut: slovar, kjer je prvi element seznam dolžin levega robnega pogoja za vsako spremenljivko, 
      drugi pa seznam dolžin desnega robnega pogoja za vsako spremenljivko.

    Rezultat:
    - Jakobijeva matrika za seznam enačb, vključno z robnimi pogoji
    """
    # Generiranje simbolov za vse spremenljivke v vektorjih spremenljivk
    vektor_spremenljivk = {var: [sym.symbols(f'{var}{i}') for i in range(var_in_dolžine_slovar[var])] 
                           for var in var_in_dolžine_slovar}
    
    # Sploščitev seznama vseh vektorjev spremenljivk z njihovimi komponentami
    vse_komponente = [komponenta for komponente in vektor_spremenljivk.values() for komponenta in komponente]
    
    vsi_odvodi = []

    # Dodajanje levega robnega pogoja
    if levi_robni:
        levi_odvodi = levi_robni(vektor_spremenljivk)  # Več robnih pogojev
        for odvod in levi_odvodi:
            vsi_odvodi.append([sym.diff(odvod, komponenta) for komponenta in vse_komponente])

    # Obdelava vsake spremenljivke neodvisno
    for var, components in vektor_spremenljivk.items():
        left_cut = lr_cut[0][var]
        right_cut = lr_cut[1][var]
        
        # Generiranje izraza za vsak indeks, ob upoštevanju levega in desnega roba
        for i in range(left_cut, len(components) - right_cut):
            fun = bulk(var, i, vektor_spremenljivk)
            
            fun_odvodi = []
            
            # Diferenciacija generirane funkcije glede na vsako komponento
            for komponenta in vse_komponente:
                odvod = sym.diff(fun, komponenta)
                fun_odvodi.append(odvod)
            
            vsi_odvodi.append(fun_odvodi)

    # Dodajanje desnega robnega pogoja
    if desni_robni:
        desni_odvodi = desni_robni(vektor_spremenljivk)  # Več robnih pogojev
        for odvod in desni_odvodi:
            vsi_odvodi.append([sym.diff(odvod, komponenta) for komponenta in vse_komponente])

    # Ustvarjanje matrike na podlagi odvodov. Ta je nerazvrščena
    jakobian_nerazvrščen = sym.Matrix(vsi_odvodi)
    
    #Tvorimo seznam posameznih dolžin spremenljivk
    dolžine_spremenljivk = list(var_in_dolžine_slovar.values())
    
    #Funkcija ki pravilno razvrsti zgornje vrstice (RP)
    def premik_dol(matrika, dolžine_spremenljivk):
        vrednost_premikov = []
        for i, num in enumerate(dolžine_spremenljivk):
            if i == 0:
                vrednost_premikov.append(0)
            else:
                vrednost_premikov.append((len(dolžine_spremenljivk) - i - 1) + sum(dolžine_spremenljivk[:i]) - i - 1)

        vrednost_premikov.pop(0)

        for positions in vrednost_premikov:
            if len(matrika) < 2:
                return matrika

            row_to_move = matrika.row(1)
            matrika.row_del(1)
            new_index = min(1 + positions, len(matrika) - 1)
            matrika = matrika.row_insert(new_index, row_to_move)

        return matrika
    
    jakobian_zgoraj_razvrščen = sym.Matrix(premik_dol(jakobian_nerazvrščen, dolžine_spremenljivk))
    
    #Funkcija, ki pravilno razvrsti spodnje vrstice (RP)
    def premik_gor(matrika, dolžine_spremenljivk):
        dolžine_spremenljivk = dolžine_spremenljivk[::-1]
        vrednost_premikov = []
        for i, num in enumerate(dolžine_spremenljivk):
            if i == 0:
                vrednost_premikov.append(0)
            else:
                vrednost_premikov.append((len(dolžine_spremenljivk) - i -1) + sum(dolžine_spremenljivk[:i])-1)

        vrednost_premikov.pop(0)
        
        for positions in vrednost_premikov:
            if len(matrika) < 2:
                return matrika

            row_to_move = matrika.row(-2)
            matrika.row_del(-2)
            new_index = min(-1- positions, len(matrika))
            matrika = matrika.row_insert(new_index, row_to_move)

        return matrika

    
    jakobian = sym.Matrix(premik_gor(jakobian_zgoraj_razvrščen, dolžine_spremenljivk))
    
    #Funkcija, ki vstavi vrednosti v matriko
    def vstavitev_vrednosti(slovar_vrednosti, matrix):

        symbols = list(matrix.free_symbols)
        lambdified_func = sym.lambdify(symbols, matrix, 'numpy')
        matrix_np = lambdified_func(**slovar_vrednosti)
        return matrix_np
    
    vrednosti_jakobiana = vstavitev_vrednosti(slovar_vrednosti, jakobian)
    jac = [jakobian, vrednosti_jakobiana]
    
    # Za rdeč tekst
    def print_red(text):
        print(colored(text, 'red'))

    # Javljanje napake oz., če je kaj narobe z jakobianom
    if not np.all(np.diagonal(vrednosti_jakobiana)):
       print_red(f"POZOR! Ničle na diagonali.")
    if jakobian.shape[0] != jakobian.shape[1]:
       print_red(f"POZOR! Jakobian ni kvadratna matrika.")
    return jac

# Definicije funkcij in robnih pogojev
def bulk(var, i, vars):
    u = vars['u']
    v = vars['v']
    z = vars['z']
    D, dt, dx, u_old = sym.symbols('D dt dx u_old')
    if var == 'u':
        return D * (u[i - 1] - 2 * u[i] + u[i + 1]) * dt / dx**2 + u_old - u[i] + z[i - 1] + z[i]
    elif var == 'v':
        return v[i] * 2
    elif var == 'z':
        return z[i] - z[i - 1]

def levi_robni(vars):
    u = vars['u'][0]
    v = vars['v'][0]
    z = vars['z'][0]
    return [u - v - 100 + z, z + v, v * 2]

def desni_robni(vars):
    u = vars['u'][-1]
    v = vars['v'][-1]
    z = vars['z'][-1]
    return [u + v - 200 - z, v + z, z * 2]

# Dolžine spremenljivk in robnih pogojev
var_in_dolžine_slovar = {'u': 5, 'v': 4, 'z': 6}
lr_cut = [{'u': 1, 'v': 1, 'z': 1}, {'u': 1, 'v': 1, 'z': 1}]
slovar_vrednosti = {'D': 0.1, 'dt': 0.2, 'dx': 0.3}

# Generiranje Jakobijeve matrike in preverjanje njene oblike
jakobian = fun_v_jakobian(bulk, var_in_dolžine_slovar, levi_robni, desni_robni, lr_cut, slovar_vrednosti)

# Izpis Jakobijeve matrike za preverjanje njene strukture
jakobian[0]

