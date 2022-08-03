### AUTHORS: Benjamin Caris & Tom Jacobs ###
### Department of Mathematics & Computer Science, Eindhoven University of Technology ###
### VERSION: February 14, 2022 ###



import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time as clock
import itertools

from numpy import linalg as LA
from scipy.spatial import distance_matrix
from typing import Callable
from io import TextIOWrapper



# ----------------- CONSTANTS -----------------

# This should eventually completely go into the dictionaries
sigma_h = 0
epsilon_h = 0

sigma_o = 3.15061
epsilon_o = 0.66386

# Data structure per molecule
# Hydrogen
sigma_h2 = f"{sigma_h} \t {sigma_h}"
epsilon_h2 = f"{epsilon_h} \t {epsilon_h}"

# Water
sigma_w = f"{sigma_o} \t {sigma_h} \t {sigma_h}"
epsilon_w = f" {epsilon_o} \t {epsilon_h} \t {epsilon_h}"

# Oxygen
sigma_o2 = f"{sigma_o} \t {sigma_o}"
epsilon_o2 = f" {epsilon_o} \t {epsilon_o}"

# Carbonmonoxide
#TO DO: find sigma and epsilon

atomsdict = {"Water": ['O','H','H'], 
    "Hydrogen": ['H','H'], 
    "Oxygen": ['O','O'], 
    "Carbonmonoxide": ['C','O'],
    "Ethanol": ['C', 'H', 'H', 'H', 'C', 'H', 'H', 'O', 'H'],
    "Water_and_Ethanol": [['O', 'H', 'H'], ['C', 'H', 'H', 'H', 'C', 'H', 'H', 'O', 'H']]
    }

sigma_ethanol = f"{3.5} \t {2.5} \t {2.5} \t {2.5} \t {3.5} \t {2.5} \t {2.5} \t {3.12} \t {0}"
epsilon_ethanol = f"{0.276144} \t {0.125520} \t {0.125520} \t {0.125520} \t {0.276144} \t {0.125520} \t {0.125520} \t {0.711280} \t {0}"

# initial positions
initial_e = np.array([[-0.231579, -0.350841, -0.037475],[-0.976393, -1.144198, 0.191635], [-0.709445, 0.352087, -0.754607],[0.619613, -0.833754, -0.565710], \
     [0.229441, 0.373160, 1.224850], [-0.628785, 0.860022, 1.736350], [0.952253, 1.174538, 0.962081], [0.868228, -0.551628, 2.114423], [0.204846, -1.119563, 2.483509]])
initial_w = np.array([[ 0.27681463, -0.2339608, 1.19034543], [ 0.12994903, 0.68795046, 0.97887271], [0.64512218, -0.57235047, 0.36933205]])
initial_w_and_e = [initial_w, initial_e]

# dihedral force parameters
C = np.array([[0.62760, 1.88280, 0.00000, -3.91622], [0.97905, 2.93716, 0.00000, -3.91622], [-0.44310, 3.83255, 0.72801, -4.11705], [0.94140, 2.82420, 0.00000, -3.76560]])
     
constantsdict = {"Water": [np.array([[15.9994, 15.9994, 15.9994], [1.00797, 1.00797, 1.00797], [1.00797, 1.00797, 1.00797]]), 5024.16, 0.9572, 628.02, 104.52*(math.pi/180), sigma_w, epsilon_w], 
    "Hydrogen": [1.00797, 245.31, 0.74, 0, 0, sigma_h2, epsilon_h2], 
    "Oxygen": [15.9994, 1, 1.208, 0, 0, sigma_o2, epsilon_o2], 
    "Carbonmonoxide": [np.array([[12.0107, 12.0107, 12.0107], [15.9994, 15.9994, 15.9994]]), 1, 1.43, 0, 0],
    "Ethanol": [np.array([[12.0107, 12.0107, 12.0107], [1.00797, 1.00797, 1.00797], [1.00797, 1.00797, 1.00797], [1.00797, 1.00797, 1.00797], [12.0107, 12.0107, 12.0107], \
         [1.00797, 1.00797, 1.00797], [1.00797, 1.00797, 1.00797], [15.9994, 15.9994, 15.9994], [1.00797, 1.00797, 1.00797]]), \
         np.array([2242.624, 2845.120, 2677.760, 4627.500]), np.array([1.529, 1.090, 1.410, 0.945]), np.array([292.880, 276.144, 313.800, 414.400, 460.240, 292.880]), \
             np.array([108.5, 107.8, 110.7, 109.5, 108.5, 109.5])*(math.pi/180), sigma_ethanol, epsilon_ethanol, C]}   # format [m, k_b, r_0, k_theta, theta_0, sigma, epsilon, C (if applicable)]



# ----------------- FUNCTIONS TO GENERATE INITIAL COORDINATES AND TOPOLOGY -----------------

def genMoleculesCube(n: int, molecule: str, coords: np.ndarray, shift: float, shift_large: float = 0, amount_of_water: int = 0):
    """Generates a cube of n**3 molecules (i.e. n x n x n) of a single type, or an 'almost' cube 
    shape of n**3 molecules for a mixture of water and ethanol with 'amount_of_water' water molecules,
    in the form of an .xyz file containing the coordinates, called "Many{'molecule'}.xyz".
    This is done by starting from initial position 'coords' and repeatedly shifting the atoms in the 
    x-, y- and z-direction. In case of a single type of molecule, the shift length is 'shift'. 
    In case of a mixture, the shift length between water molecules is 'shift', and the shift length 
    between water and ethanol molecules, as well as between ethanol molecules, is 'shift_large'. In
    case of a mixture, some random rows get a few molecules extra, whereas the other rows get some
    molecules less. Due to this, the shape is not exactly cube. 

    'molecule' should be one of the following: 'Water', 'Hydrogen', 'Oxygen', 'Carbonmonoxide', 'Ethanol', 'Water_and_Ethanol'.
    For 'coords', one may use e.g. 'coords_w' (water), 'coords_e' (ethanol) or 'coords_w_and_e' (mixture). In any case, 
    the initial coordinates for water and ethanol, in case of a mixture, should have the same center of mass."""

    if amount_of_water == 0:   # single molecule type
        atoms = atomsdict[molecule]
        molecules = open(f"Many{molecule}.xyz", "w")
        molecules.write(f"{n**3 * len(atoms)} \n")
        molecules.write(f"{molecule} \n")
    
        for i in range(0,n):
            for j in range(0,n):
                for k in range(0,n):
                    for l in range(0, len(atoms)):
                        molecules.write(f"{atoms[l]} \t {coords[l,0] + i * shift} \t {coords[l,1] + j * shift} \t {coords[l,2] + k * shift} \n")

    else:   # mixture
        atoms_w = atomsdict['Water']
        atoms_e = atomsdict['Ethanol']
        
        molecules = open(f"ManyWater_and_Ethanol.xyz", "w")
        molecules.write(f"{len(atoms_w)*amount_of_water + len(atoms_e)*(n**3-amount_of_water)} \n")
        molecules.write(f"Water and Ethanol \n")

        amount_of_ethanol = n**3 - amount_of_water
        ethanol_per_row = amount_of_ethanol // n**2   # place this amount of ethanol in each row
        water_per_row = amount_of_water // n**2   # idem
        remainder_e = amount_of_ethanol - n**2 * ethanol_per_row   # at the end, place this additional amount of ethanol somewhere
        extra_e = random.sample(range(0, n**2), int(remainder_e))   # rows for extra ethanol molecules

        # remark: loops for water and ethanol are not joined, because format of .xyz file should be: all water molecules first, then all ethanol molecules (for later functions)
        for i in range(0,n):
            for j in range(0,n):
                for k in range(0,water_per_row):
                    for l in range(0, 3):
                        molecules.write(f"{atoms_w[l]} \t {coords[0][l][0] + i * shift} \t {coords[0][l][1] + j * shift} \t {coords[0][l][2] + k * shift} \n")
                if i*n + j not in extra_e:   # row with no extra ethanol, i.e. with extra water
                    for l in range(0, 3):
                        molecules.write(f"{atoms_w[l]} \t {coords[0][l][0] + i * shift} \t {coords[0][l][1] + j * shift} \t {coords[0][l][2] - shift} \n")
         
        for i in range(0,n):
            for j in range(0,n):
                for k in range(0,ethanol_per_row):
                    for l in range(0, 9):
                        molecules.write(f"{atoms_e[l]} \t {coords[1][l][0] + i * shift} \t {coords[1][l][1] + j * shift} \t {coords[1][l][2] + (water_per_row - 1) * shift + (k + 1) * shift_large} \n")
                if i*n + j in extra_e:   # row with extra ethanol
                    for l in range(0, 9):
                        molecules.write(f"{atoms_e[l]} \t {coords[1][l][2] + (water_per_row - 1) * shift + (ethanol_per_row + 1) * shift_large}  \t {coords[1][l][1] + j * shift} \t {coords[1][l][0] + i * shift}\n")

    molecules.close()

def genTopology(n: int, molecule: str, mixed: bool = False, amount_of_water: int = 0):
    """Generates a topology for 'n' molecules, in the form of a .txt file called "Topology{'molecule'}.txt". 
    'molecule' should be one of the following: 'Water', 'Hydrogen', 'Oxygen', 'Carbonmonoxide', 'Ethanol', 'Water_and_Ethanol'."""

    if mixed == False:
        _, k_b, r_0, k_theta, theta_0, sigma, epsilon = constantsdict[molecule][0:7]   # find parameters of single molecule

        if molecule == 'Ethanol':   # dihedral forces; extra constants needed
            C = constantsdict[molecule][7]

        topology = open(f"Topology{molecule}.txt", "w")
    
        if type(k_theta) == float and k_theta == 0:   # diatomic molecule
            topology.write(f"Molecules: {int(n)} \t Bonds: {int(n)} \t Angles: {int(0)} \t Dihedrals: {int(0)} \n")
        elif type(k_theta) == float and k_theta == 628.02:   # water
            topology.write(f"Molecules: {int(n)} \t Bonds: {int(2*n)} \t Angles: {int(n)} \t Dihedrals: {int(0)}  \n")
        else:   # ethanol
            topology.write(f"Molecules: {int(n)} \t Bonds: {int(8*n)} \t Angles: {int(13*n)} \t Dihedrals: {int(12*n)}  \n")
        
        topology.write(f"sigma {int(n)} \n")   # LJ parameters
        for i in range(0,n):
            topology.write(f"{sigma} \n")
        
        topology.write(f"epsilon {int(n)} \n")    
        for i in range(0,n):
            topology.write(f"{epsilon} \n")
    
        if type(k_theta) == float and k_theta == 0:   # diatomic molecule; bonds only
            topology.write(f"bonds {int(n)} \n")

            for i in range(0,n):
                topology.write(f"{2*i} \t {2*i+1} \t {k_b} \t {r_0} \n")

        elif type(k_theta) == float and k_theta == 628.02:   # water; bonds and angles
            topology.write(f"bonds {int(2*n)} \n")

            for i in range(0,n):
                topology.write(f"{3*i} \t {3*i+1} \t {k_b} \t {r_0} \n")
                topology.write(f"{3*i} \t {3*i+2} \t {k_b} \t {r_0} \n")

            topology.write(f"angles {int(n)} \n")

            for i in range(0,n):
                topology.write(f"{3*i+1} \t {3*i} \t {3*i+2} \t {k_theta} \t {theta_0} \n")
        else:   # ethanol; bonds, angles and dihedrals
            topology.write(f"bonds {int(8*n)} \n")
        
            for i in range(0,n):
                topology.write(f"{9*i} \t {9*i+4} \t {k_b[0]} \t {r_0[0]} \n")
                topology.write(f"{9*i+2} \t {9*i} \t {k_b[1]} \t {r_0[1]} \n")
                topology.write(f"{9*i+3} \t {9*i} \t {k_b[1]} \t {r_0[1]} \n")
                topology.write(f"{9*i+1} \t {9*i} \t {k_b[1]} \t {r_0[1]} \n")
                topology.write(f"{9*i+5} \t {9*i+4} \t {k_b[1]} \t {r_0[1]} \n")
                topology.write(f"{9*i+6} \t {9*i+4} \t {k_b[1]} \t {r_0[1]} \n")
                topology.write(f"{9*i+4} \t {9*i+7} \t {k_b[2]} \t {r_0[2]} \n")
                topology.write(f"{9*i+8} \t {9*i+7} \t {k_b[3]} \t {r_0[3]} \n")
          
            topology.write(f"angles {int(13*n)} \n")
        
            for i in range(0,n):
                topology.write(f"{9*i+1} \t {9*i} \t {9*i+4} \t {k_theta[0]} \t {theta_0[0]} \n")
                topology.write(f"{9*i+2} \t {9*i} \t {9*i+4} \t {k_theta[0]} \t {theta_0[0]} \n")
                topology.write(f"{9*i+3} \t {9*i} \t {9*i+4} \t {k_theta[0]} \t {theta_0[0]} \n")
                
                topology.write(f"{9*i+3} \t {9*i} \t {9*i+2} \t {k_theta[1]} \t {theta_0[1]} \n")
                topology.write(f"{9*i+3} \t {9*i} \t {9*i+1} \t {k_theta[1]} \t {theta_0[1]} \n")
                topology.write(f"{9*i+2} \t {9*i} \t {9*i+1} \t {k_theta[1]} \t {theta_0[1]} \n")
                topology.write(f"{9*i+5} \t {9*i+4} \t {9*i+6} \t {k_theta[1]} \t {theta_0[1]} \n")
                
                topology.write(f"{9*i} \t {9*i+4} \t {9*i+6} \t {k_theta[2]} \t {theta_0[2]} \n")
                topology.write(f"{9*i} \t {9*i+4} \t {9*i+5} \t {k_theta[2]} \t {theta_0[2]} \n")
                
                topology.write(f"{9*i} \t {9*i+4} \t {9*i+7} \t {k_theta[3]} \t {theta_0[3]} \n")
                
                topology.write(f"{9*i+4} \t {9*i+7} \t {9*i+8} \t {k_theta[4]} \t {theta_0[4]} \n")
                
                topology.write(f"{9*i+5} \t {9*i+4} \t {9*i+7} \t {k_theta[5]} \t {theta_0[5]} \n")
                topology.write(f"{9*i+6} \t {9*i+4} \t {9*i+7} \t {k_theta[5]} \t {theta_0[5]} \n")
                
            topology.write(f"dihedrals {int(12*n)} \n")

            for i in range(0,n):
                topology.write(f"{9*i+1} \t {9*i} \t {9*i+4} \t {9*i+5} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
                topology.write(f"{9*i+2} \t {9*i} \t {9*i+4} \t {9*i+5} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
                topology.write(f"{9*i+3} \t {9*i} \t {9*i+4} \t {9*i+5} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
                topology.write(f"{9*i+1} \t {9*i} \t {9*i+4} \t {9*i+6} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
                topology.write(f"{9*i+2} \t {9*i} \t {9*i+4} \t {9*i+6} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
                topology.write(f"{9*i+3} \t {9*i} \t {9*i+4} \t {9*i+6} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
                
                topology.write(f"{9*i+1} \t {9*i} \t {9*i+4} \t {9*i+7} \t {C[1,0]} \t {C[1,1]} \t {C[1,2]} \t {C[1,3]} \n")
                topology.write(f"{9*i+2} \t {9*i} \t {9*i+4} \t {9*i+7} \t {C[1,0]} \t {C[1,1]} \t {C[1,2]} \t {C[1,3]} \n")
                topology.write(f"{9*i+3} \t {9*i} \t {9*i+4} \t {9*i+7} \t {C[1,0]} \t {C[1,1]} \t {C[1,2]} \t {C[1,3]} \n")
                
                topology.write(f"{9*i} \t {9*i+4} \t {9*i+7} \t {9*i+8} \t {C[2,0]} \t {C[2,1]} \t {C[2,2]} \t {C[2,3]} \n")
                
                topology.write(f"{9*i+5} \t {9*i+4} \t {9*i+7} \t {9*i+8} \t {C[3,0]} \t {C[3,1]} \t {C[3,2]} \t {C[3,3]} \n")
                topology.write(f"{9*i+6} \t {9*i+4} \t {9*i+7} \t {9*i+8} \t {C[3,0]} \t {C[3,1]} \t {C[3,2]} \t {C[3,3]} \n")
                
    else:   # mixture
        _, k_b_w, r_0_w, k_theta_w, theta_0_w, sigma_w, epsilon_w = constantsdict[molecule[0]][0:7]   # parameters of both water and ethanol
        _, k_b_e, r_0_e, k_theta_e, theta_0_e, sigma_e, epsilon_e = constantsdict[molecule[1]][0:7]
        C = constantsdict[molecule[1]][7]
        
        topology = open(f"Topology{molecule[0]}_and_{molecule[1]}.txt", "w")
        
        topology.write(f"Molecules: {int(n)} \t Bonds: {int(2*amount_of_water + 8*(n-amount_of_water))} \t Angles: {int(amount_of_water+ 13*(n-amount_of_water))} \t Dihedrals: {int(12*(n-amount_of_water))} \n")
        
        topology.write(f"sigma {int(n)} \n")   # LJ parameters per molecule
        for i in range(0,amount_of_water):
            topology.write(f"{sigma_w} \n")
        
        for i in range(amount_of_water,n):
            topology.write(f"{sigma_e} \n")
        
        topology.write(f"epsilon {int(n)} \n")    
        for i in range(0,amount_of_water):
            topology.write(f"{epsilon_w} \n")
            
        for i in range(amount_of_water,n):
            topology.write(f"{epsilon_e} \n")
            
        topology.write(f"bonds {int(amount_of_water*2+ (n-amount_of_water)*8)} \n")
        
        for i in range(0,amount_of_water):   # bonds water
                topology.write(f"{3*i} \t {3*i+1} \t {k_b_w} \t {r_0_w} \n")
                topology.write(f"{3*i} \t {3*i+2} \t {k_b_w} \t {r_0_w} \n")
                
        for i in range(0,n-amount_of_water):   # bonds ethanol
                topology.write(f"{amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {k_b_e[0]} \t {r_0_e[0]} \n")
                topology.write(f"{amount_of_water*3+9*i+2} \t {amount_of_water*3+9*i} \t {k_b_e[1]} \t {r_0_e[1]} \n")
                topology.write(f"{amount_of_water*3+9*i+3} \t {amount_of_water*3+9*i} \t {k_b_e[1]} \t {r_0_e[1]} \n")
                topology.write(f"{amount_of_water*3+9*i+1} \t {amount_of_water*3+9*i} \t {k_b_e[1]} \t {r_0_e[1]} \n")
                topology.write(f"{amount_of_water*3+9*i+5} \t {amount_of_water*3+9*i+4} \t {k_b_e[1]} \t {r_0_e[1]} \n")
                topology.write(f"{amount_of_water*3+9*i+6} \t {amount_of_water*3+9*i+4} \t {k_b_e[1]} \t {r_0_e[1]} \n")
                topology.write(f"{amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {k_b_e[2]} \t {r_0_e[2]} \n")
                topology.write(f"{amount_of_water*3+9*i+8} \t {amount_of_water*3+9*i+7} \t {k_b_e[3]} \t {r_0_e[3]} \n")

        topology.write(f"angles {int(amount_of_water + (n-amount_of_water)*13)} \n")
        
        for i in range(0,amount_of_water):   # angles water
                topology.write(f"{3*i+1} \t {3*i} \t {3*i+2} \t {k_theta_w} \t {theta_0_w} \n")
                
        for i in range(0,n-amount_of_water):   # angles ethanol
                topology.write(f"{amount_of_water*3+9*i+1} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {k_theta_e[0]} \t {theta_0_e[0]} \n")
                topology.write(f"{amount_of_water*3+9*i+2} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {k_theta_e[0]} \t {theta_0_e[0]} \n")
                topology.write(f"{amount_of_water*3+9*i+3} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {k_theta_e[0]} \t {theta_0_e[0]} \n")
                
                topology.write(f"{amount_of_water*3+9*i+3} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+2} \t {k_theta_e[1]} \t {theta_0_e[1]} \n")
                topology.write(f"{amount_of_water*3+9*i+3} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+1} \t {k_theta_e[1]} \t {theta_0_e[1]} \n")
                topology.write(f"{amount_of_water*3+9*i+2} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+1} \t {k_theta_e[1]} \t {theta_0_e[1]} \n")
                topology.write(f"{amount_of_water*3+9*i+5} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+6} \t {k_theta_e[1]} \t {theta_0_e[1]} \n")
                
                topology.write(f"{amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+6} \t {k_theta_e[2]} \t {theta_0_e[2]} \n")
                topology.write(f"{amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+5} \t {k_theta_e[2]} \t {theta_0_e[2]} \n")
                
                topology.write(f"{amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {k_theta_e[3]} \t {theta_0_e[3]} \n")
                
                topology.write(f"{amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {amount_of_water*3+9*i+8} \t {k_theta_e[4]} \t {theta_0_e[4]} \n")
                
                topology.write(f"{amount_of_water*3+9*i+5} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {k_theta_e[5]} \t {theta_0_e[5]} \n")
                topology.write(f"{amount_of_water*3+9*i+6} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {k_theta_e[5]} \t {theta_0_e[5]} \n")
                
        topology.write(f"dihedrals {int(12*(n-amount_of_water))} \n")
    
        for i in range(0,n-amount_of_water):   # dihedrals ethanol
            topology.write(f"{amount_of_water*3+9*i+1} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+5} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
            topology.write(f"{amount_of_water*3+9*i+2} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+5} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
            topology.write(f"{amount_of_water*3+9*i+3} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+5} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
            topology.write(f"{amount_of_water*3+9*i+1} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+6} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
            topology.write(f"{amount_of_water*3+9*i+2} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+6} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
            topology.write(f"{amount_of_water*3+9*i+3} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+6} \t {C[0,0]} \t {C[0,1]} \t {C[0,2]} \t {C[0,3]} \n")
                    
            topology.write(f"{amount_of_water*3+9*i+1} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {C[1,0]} \t {C[1,1]} \t {C[1,2]} \t {C[1,3]} \n")
            topology.write(f"{amount_of_water*3+9*i+2} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {C[1,0]} \t {C[1,1]} \t {C[1,2]} \t {C[1,3]} \n")
            topology.write(f"{amount_of_water*3+9*i+3} \t {amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {C[1,0]} \t {C[1,1]} \t {C[1,2]} \t {C[1,3]} \n")
                    
            topology.write(f"{amount_of_water*3+9*i} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {amount_of_water*3+9*i+8} \t {C[2,0]} \t {C[2,1]} \t {C[2,2]} \t {C[2,3]} \n")
                    
            topology.write(f"{amount_of_water*3+9*i+5} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {amount_of_water*3+9*i+8} \t {C[3,0]} \t {C[3,1]} \t {C[3,2]} \t {C[3,3]} \n")
            topology.write(f"{amount_of_water*3+9*i+6} \t {amount_of_water*3+9*i+4} \t {amount_of_water*3+9*i+7} \t {amount_of_water*3+9*i+8} \t {C[3,0]} \t {C[3,1]} \t {C[3,2]} \t {C[3,3]} \n")



# ----------------- FUNCTIONS TO READ COORDINATE AND TOPOLOGY FILES -----------------

def readCoordinates(filename: str):
    """Reads coordinates of molecules in .xyz file into a NumPy array."""

    file = open(filename, "r")
    coordlst = []

    for num, line in enumerate(file):
        if num > 1:
            line = line.split()[1:]
            coords = [float(x) for x in line]
            coordlst.append(coords)

    coords = np.array(coordlst)

    return coords

def epsilon_and_sigma_block(epsilon1: np.ndarray, epsilon2: np.ndarray, sigma1: np.ndarray, sigma2: np.ndarray):
    """Creates epsilon and sigma matrices, by mixing epsilons 
    in array epsilon1 and epsilon2 element-wise corresponding to a molecule, 
    idem for sigmas, according to Lorentz-Berthelot mixing rules."""

    sigma_matrix = np.zeros([len(sigma1),len(sigma2)]) 
    epsilon_matrix = np.zeros([len(epsilon1), len(epsilon2)])
    
    for i in range(0, len(sigma1)):
        for j in range(0, len(sigma2)):
                sigma_matrix[i,j] = (sigma1[i]+ sigma2[j])/2
                epsilon_matrix[i,j] = np.sqrt(epsilon1[i] * epsilon2[j])
                
    return epsilon_matrix, sigma_matrix

def readTopology(filename: str):
    """Reads topology .txt file into NumPy arrays containing bonds, angles, dihedrals, and their respective constants."""

    topology = open(filename, "r")

    sigmalst = []
    epsilonlst = []
    
    bondlst = []
    anglelst = []
    dihedrallst = []

    firstline = topology.readline().split()   # first line contains number of molecules, bonds, angles and dihedrals
    moleculenumber = int(firstline[1])
    bondnumber = int(firstline[3])
    anglenumber = int(firstline[5])
    dihedralnumber = int(firstline[7])
   
    for num, line in enumerate(topology):
        if 0 < num < moleculenumber + 1:   # sigmas
            lst = [float(x) for x in line.split()]
            sigmalst.append(lst)
    
        if num > moleculenumber + 1 and num < 2 * (moleculenumber + 1):   # epsilons
            lst = [float(x) for x in line.split()]
            epsilonlst.append(lst)
        
        if num > 2 * (moleculenumber + 1) and num < (bondnumber + 1) + 2 * (moleculenumber + 1):   # bonds
            lst = [float(x) for x in line.split()]
            bondlst.append(lst)

        if num > (bondnumber + 1) + 2 * (moleculenumber + 1) and num < (anglenumber + 1) + (bondnumber + 1) + 2 * (moleculenumber + 1):   # angles
            lst = [float(x) for x in line.split()]
            anglelst.append(lst)

        if num > (anglenumber + 1) + (bondnumber + 1) + 2 * (moleculenumber + 1):   # dihedrals
            lst = [float(x) for x in line.split()]
            dihedrallst.append(lst)
   
    sigma = list(itertools.chain.from_iterable(sigmalst))
    epsilon = list(itertools.chain.from_iterable(epsilonlst))
    
    bonds = np.array(bondlst, dtype = int)[:,0:2]   # convert lists to arrays, and float to int for indices
    bondconsts = np.array(bondlst)[:,2:4]

    if 0 < dihedralnumber < moleculenumber * 12:   # mixture
        amount_of_ethanol = dihedralnumber // 12
        amount_of_water =  moleculenumber - amount_of_ethanol
        
        sigma_water = sigma[0:3]
        sigma_ethanol = sigma[-9:]
        epsilon_water = epsilon[0:3]
        epsilon_ethanol = epsilon[-9:]

        # create blocks for large epsilon and sigma matrices
        ww_m = epsilon_and_sigma_block(epsilon_water, epsilon_water, sigma_water, sigma_water)
        ee_m = epsilon_and_sigma_block(epsilon_ethanol, epsilon_ethanol, sigma_ethanol, sigma_ethanol)
        we_m = epsilon_and_sigma_block(epsilon_water, epsilon_ethanol, sigma_water, sigma_ethanol)
        
        WW_eps = np.block([[ww_m[0]]*amount_of_water]*amount_of_water) - np.kron(np.eye(amount_of_water),ww_m[0])   # Subtracting diagonal blocks to prevent self interaction
        EE_eps = np.block([[ee_m[0]]*amount_of_ethanol]*amount_of_ethanol) - np.kron(np.eye(amount_of_ethanol),ee_m[0])
        WE_eps = np.block([[we_m[0]]*amount_of_ethanol]*amount_of_water)
        EW_eps = np.transpose(WE_eps)
                
        WW_sig = np.block([[ww_m[1]]*amount_of_water]*amount_of_water) - np.kron(np.eye(amount_of_water),ww_m[1])   # Subtracting diagonal blocks to prevent self interaction
        EE_sig = np.block([[ee_m[1]]*amount_of_ethanol]*amount_of_ethanol) - np.kron(np.eye(amount_of_ethanol),ee_m[1])
        WE_sig = np.block([[we_m[1]]*amount_of_ethanol]*amount_of_water)
        EW_sig = np.transpose(WE_sig)
        
        epsilon_matrix = np.block([[WW_eps, WE_eps],[EW_eps, EE_eps]])   # construct matrices from blocks
        sigma_matrix = np.block([[WW_sig, WE_sig],[EW_sig, EE_sig]])
                
    elif anglenumber > moleculenumber:   # ethanol
       sigma_ethanol = sigma[0:9]
       epsilon_ethanol = epsilon[0:9]
       
       ee_m = epsilon_and_sigma_block(epsilon_ethanol, epsilon_ethanol, sigma_ethanol, sigma_ethanol)
       
       epsilon_matrix = np.block([[ee_m[0]]*moleculenumber]*moleculenumber) - np.kron(np.eye(moleculenumber),ee_m[0])   # Subtracting diagonal blocks to prevent self interaction
       sigma_matrix = np.block([[ee_m[1]]*moleculenumber]*moleculenumber) - np.kron(np.eye(moleculenumber),ee_m[1])
   
    else:
       sigma_water = sigma[0:3]
       epsilon_water= epsilon[0:3]
       
       ww_m = epsilon_and_sigma_block(epsilon_water, epsilon_water, sigma_water, sigma_water)
       
       epsilon_matrix = np.block([[ww_m[0]]*moleculenumber]*moleculenumber) - np.kron(np.eye(moleculenumber),ww_m[0])   # Subtracting diagonal blocks to prevent self interaction
       sigma_matrix = np.block([[ww_m[1]]*moleculenumber]*moleculenumber) - np.kron(np.eye(moleculenumber),ww_m[1])

    if anglelst != []:    # create array with angle indices and array with angle constants
        angles = np.array(anglelst, dtype = int)[:,0:3]
        angleconsts = np.array(anglelst)[:,3:5]
    
    else:
        angles = np.array([])
        angleconsts = np.array([])

    if dihedrallst != []:   # create array with dihedral indices and array with dihedral constants
        dihedrals = np.array(dihedrallst, dtype = int)[:,0:4]
        dihedralconsts = np.array(dihedrallst)[:,4:8]
    
    else:
        dihedrals = np.array([])
        dihedralconsts = np.array([])

    return bonds, bondconsts, angles, angleconsts, sigma_matrix, epsilon_matrix, dihedrals, dihedralconsts



# ----------------- FUNCTIONS TO COMPUTE INTERNAL FORCES -----------------

def bond_force(coords: np.ndarray, bonds: np.ndarray, bondconsts: np.ndarray):
    """Computes bond forces between atoms with positions 'coords', with the
    bond indices contained in 'bonds' and their constants in 'bondconsts'."""

    dr = coords[bonds[:,1]] - coords[bonds[:,0]]   # displacements along bonds
    Fa = (bondconsts[:,0] * (1 - bondconsts[:,1] / LA.norm(dr, axis = 1))).reshape(len(bonds),1) * dr   # bond force on atoms A due to bond AB
    Fb = -Fa   # bond force on atoms B due to bond AB
    J_bond = np.sum(np.power((LA.norm(dr,axis = 1)- bondconsts[:,1]),2)*bondconsts[:,0])   # potential energy

    Ja = np.zeros((len(coords), 3))    # array to store forces on atoms A due to bonds AB
    Jb = np.zeros((len(coords), 3))   # array to store forces on atoms B due to bonds AB
    np.add.at(Ja, bonds[:,0], Fa)
    np.add.at(Jb, bonds[:,1], Fb)
    J = Ja + Jb   # total bond force on every atom

    return J, J_bond

def angular_force(J: np.ndarray, coords: np.ndarray, angles: np.ndarray, angleconsts: np.ndarray):
    """Computes angular forces between atoms with positions 'coords',
    with the angle indices contained in 'angles' and their constants in
    'angleconsts', and adds these forces to the forces in array J."""

    atom_coords = coords[angles]   # coordinates of atoms that make angles

    r_1 = atom_coords[:,0,:] - atom_coords[:,1,:]    # displacement along sides AB which enclose angles ABC
    r_2 = atom_coords[:,2,:] - atom_coords[:,1,:]   # displacement along sides BC which enclose angles ABC
    cos_theta = np.sum(r_1*r_2, axis=1)/ (LA.norm(r_1, axis = 1) * LA.norm(r_2, axis = 1))   # cosines of angles
    cos_theta = np.clip(cos_theta, -1, 1)   # rounding errors could make cosine greater than 1 in magnitude, which gives error in next line, so use np.clip
    theta = np.arccos(cos_theta)   # angles

    magnitude = np.zeros((len(angles),2))    # array to add magnitude of angular forces into
    magnitude[:,0] = -angleconsts[:,0] * (theta - angleconsts[:,1]) / LA.norm(r_1, axis = 1)   # magnitude of force on atoms A due to angles ABC
    magnitude[:,1] = -angleconsts[:,0] * (theta - angleconsts[:,1]) / LA.norm(r_2, axis = 1)    # magnitude of force on atoms C due to angles ABC

    J_angle = np.sum(np.power((theta - angleconsts[:,1]),2)*angleconsts[:,0])   # angular potential

    direction = np.zeros((len(angles),2,3))    # array in which to store directions of angular forces
    u = -np.cross(r_1, r_2)   # vector pointing out of plane in which angle ABC lies
    direction[:,0,:] = np.cross(u, r_1)   # angle of force on atom A due to angle ABC
    direction[:,1,:] = -np.cross(u, r_2)    # angle of force on atom C due to angle ABC
    direction = np.divide(direction, LA.norm(direction, axis = 2).reshape(len(angles),2,1), \
        where = LA.norm(direction, axis = 2).reshape(len(angles),2,1) != 0, out = np.zeros_like(direction))   # normalized directions
           
    F_angular = magnitude.reshape(len(angles),2,1) * direction   # angular forces on atoms A and C due to angles ABC
    F_angular = np.insert(F_angular, 1, -(F_angular[:,0,:] + F_angular[:,1,:]), axis = 1)   # insert angular forces on atoms B due to angles ABC

    np.add.at(J, angles, F_angular)   # add total angular force on every atom to J

    return J, J_angle

def dihedral_force(J: np.ndarray, coords: np.ndarray, dihedrals: np.ndarray, dihedralconsts: np.ndarray):
    """Computes angular forces between atoms with positions 'coords',
    with the dihedral indices contained in 'dihedrals' and their constants in
    'dihedralconsts', and adds these forces to the forces in array J.
    Details about the calculation can be found in e.g.
    https://hal-mines-paristech.archives-ouvertes.fr/hal-00924263 """
    
    atom_coords = coords[dihedrals]    # coordinates of atoms that make dihedrals
    nr_molecules = int(len(dihedrals) / 12)

    # bonds that make dihedrals
    ba = atom_coords[:,0,:] - atom_coords[:,1,:]
    bc = atom_coords[:,2,:] - atom_coords[:,1,:]
    cd = atom_coords[:,3,:] - atom_coords[:,2,:]

    # compute relevant angles 
    C1 = np.cross(ba, bc)
    C2 = np.cross(cd, -bc)
    ab_abs = LA.norm(ba, axis = 1)
    cd_abs = LA.norm(cd, axis = 1)

    cos_phi = np.sum(C1*C2, axis=1) / (LA.norm(C1, axis = 1) * LA.norm(C2, axis = 1))
    cos_phi = np.clip(cos_phi, -1, 1)
    phi = np.arccos(cos_phi)

    cos_A1 = np.sum(ba*bc, axis=1) / (LA.norm(ba, axis = 1) * LA.norm(bc, axis = 1))
    cos_A1 = np.clip(cos_A1, -1, 1)
    A1 = np.arccos(cos_A1)

    cos_A2 = np.sum(-bc*cd, axis=1) / (LA.norm(-bc, axis = 1) * LA.norm(cd, axis = 1))
    cos_A2 = np.clip(cos_A2, -1, 1)
    A2 = np.arccos(cos_A2)

    # magnitude of dihedral forces on atoms A and D due to dihedrals ABCD
    Fa = (np.divide(1/2, ab_abs*np.sin(A1), where = ab_abs*np.sin(A1) != 0, out = np.zeros_like(ab_abs*np.sin(A1))) \
        * (dihedralconsts[:,0]*np.sin(phi)-2*dihedralconsts[:,1]*np.sin(2*phi)+3*dihedralconsts[:,2]*np.sin(3*phi) \
        - 4*dihedralconsts[:,3]*np.sin(4*phi))).reshape(12*nr_molecules,1) * C1 / LA.norm(C1, axis = 1).reshape(12*nr_molecules,1)
    Fd = (np.divide(1/2, cd_abs*np.sin(A2), where = cd_abs*np.sin(A2) != 0, out = np.zeros_like(cd_abs*np.sin(A2))) \
        * (dihedralconsts[:,0]*np.sin(phi)-2*dihedralconsts[:,1]*np.sin(2*phi)+3*dihedralconsts[:,2]*np.sin(3*phi) \
        - 4*dihedralconsts[:,3]*np.sin(4*phi))).reshape(12*nr_molecules,1) * C2 / LA.norm(C2, axis = 1).reshape(12*nr_molecules,1)
    oc = bc/2

    tc = -(np.cross(oc, Fd)+ np.cross(cd,Fd)/2+ np.cross(ba,Fa)/2)
    oc_abs = LA.norm(oc, axis = 1)
    Fc =  (1/oc_abs**2).reshape(12*nr_molecules,1) * np.cross(tc, oc)   # magnitude of dihedral forces on atoms C due to dihedrals ABCD

    Fb = -Fa-Fc-Fd   # # magnitude of dihedral forces on atoms B due to dihedrals ABCD
   
    J_dihedral = np.sum(1/2*(dihedralconsts[:,0]*(1+np.cos(phi)) + dihedralconsts[:,1]*(1+np.cos(2*phi)) + dihedralconsts[:,2]*(1+np.cos(3*phi)) + dihedralconsts[:,3]*(1+np.cos(4*phi))))   # dihedral potential
    
    # arrays to store dihedral forces on atoms A, B, C, D, respectively, due to dihedrals ABCD
    Ja = np.zeros((len(coords), 3))
    Jb = np.zeros((len(coords), 3))
    Jc = np.zeros((len(coords), 3))
    Jd = np.zeros((len(coords), 3))

    np.add.at(Ja, dihedrals[:,0], Fa)
    np.add.at(Jb, dihedrals[:,1], Fb)
    np.add.at(Jc, dihedrals[:,2], Fc)
    np.add.at(Jd, dihedrals[:,3], Fd)

    J += Ja + Jb + Jc + Jd   # add total dihedral force on every atom to J

    return J, J_dihedral

def internal_forces(coords: np.ndarray, molecule: str, bonds: np.ndarray, bondconsts: np.ndarray, \
    angles: np.ndarray, angleconsts: np.ndarray, dihedrals: np.ndarray, dihedralconsts: np.ndarray):
    """Calculates internal forces for molecules of type 'molecule', with atom positions 'coords', 
    indices of bonds, angles, dihedrals, respectively, given by 'bonds', 'angles', 'dihedrals', 
    and their respective constants given by 'bondconsts', 'angleconsts', 'dihedralconsts'.
    'molecule' should be one of the following: 'Water', 'Hydrogen', 'Oxygen', 'Carbonmonoxide', 'Ethanol', 'Water_and_Ethanol'."""
    
    J_bond = 0
    J_angle = 0
    J_dihedral = 0

    if molecule == 'Hydrogen' or molecule == 'Oxygen' or molecule == 'Carbonmonoxide':
        J, J_bond = bond_force(coords, bonds, bondconsts)
    
    elif molecule == 'Water': 
        J, J_bond = bond_force(coords, bonds, bondconsts)
        J, J_angle = angular_force(J, coords, angles, angleconsts)
        
    else:
        J, J_bond = bond_force(coords, bonds, bondconsts)
        J, J_angle = angular_force(J, coords, angles,angleconsts)
        J, J_dihedral = dihedral_force(J, coords, dihedrals, dihedralconsts)
        
    J = J.reshape(int(len(coords)), 3, order = 'F')
   
    Potential = J_bond + J_angle + J_dihedral
    
    return J, Potential



# ----------------- FUNCTIONS TO IMPLEMENT PBC AND COMPUTE LJ FORCES -----------------

def indices(sigma_matrix: np.ndarray, epsilon_matrix: np.ndarray, nr_atoms: int):
    """Finds for every atom in the simulation box, the indices of 
    atoms in the simulation box and in copy boxes from which it 
    experiences a nonzero LJ force. Returns the result in the form
    of an array of (2,1) arrays, where every (2,1) array has an index 
    of an atom in the simulation box as first entry, and as second entry 
    an index of an atom in either the simulation box or a copy box."""

    prod_matrix = sigma_matrix * epsilon_matrix   # entry in prod_matrix is nonzero iff corresponding entries in epsilon and sigma matrix are nonzero iff LJ between atoms is nonzero
    nz_args = np.argwhere(prod_matrix)   # combinations of atoms in sim box between which there is nonzero LJ
    indices_extended = np.zeros((len(nz_args) * 27, 2))   # copy indices of nonzero entries 27 times, since there are 27 boxes
    counter = 0

    for indices in nz_args:   
        copy_indices = np.zeros((27, 2))
        copy_indices[:,0] = np.repeat(indices[0], 27)   # repeat atom indices[0] 27 times, to pair with all 27 copies of atom indices[1]
        copy_indices[:,1] = np.arange(indices[1], indices[1] + 27 * nr_atoms, nr_atoms, dtype = int)   # all 27 copies of atom indices[1]
        indices_extended[counter * 27:(counter + 1) * 27, :] = copy_indices   # add all pairings between atom indices[0] and copies of atom indices[1]
        counter += 1

    indices_extended = np.array(indices_extended, dtype = int).reshape(np.shape(indices_extended))
    return indices_extended

def adjacencies(coords: np.ndarray, cutoff: float, indices: np.ndarray):
    """Returns an array of adjacencies, containing the combinations of 
    atoms in 'indices' between which the distance is at most 'cutoff'."""
    
    R = LA.norm(coords[indices[:,0]] - coords[indices[:,1]], axis = 1)
    adjacency_list = indices[R <= cutoff]

    return adjacency_list

def LJ_forces_PBC(coords: np.ndarray, adjacency_list: np.ndarray, sigma_matrix: np.ndarray, epsilon_matrix: np.ndarray, nr_atoms: int):
    """Compute LJ forces on atoms in simulation box, with positions 'coords',
    based on adjacencies contained in 'adjacencies', and epsilon and sigma parameters 
    between atoms contained in 'epsilon_matrix' and 'sigma_matrix', respectively."""

    reduced_adj_list = adjacency_list.copy()
    reduced_adj_list[:,1] = reduced_adj_list[:,1] % nr_atoms   # reduce index to original atom number in sim box, to look up epsilon and sigma corresponding to copy
    sigmas = sigma_matrix[reduced_adj_list[:,0], reduced_adj_list[:,1]]
    epsilons = epsilon_matrix[reduced_adj_list[:,0], reduced_adj_list[:,1]]

    direction_vector = coords[adjacency_list[:,1]] - coords[adjacency_list[:,0]]   # directions of LJ forces
    R = LA.norm(direction_vector, axis = 1)
    normed_directions = direction_vector/ R.reshape(len(direction_vector), 1)   # normalized directions

    # magnitudes of LJ forces
    W = sigmas / R   
    LJ = 4 * epsilons * (12 * W**12 - 6 * W**6) / R

    LJ_pot = np.sum(4 * epsilons * (np.power(W,12) - np.power(W,6)))   # LJ potential

    K_components = LJ.reshape(len(LJ),1) * normed_directions   # LJ forces
    K = np.zeros((nr_atoms, 3))
    np.add.at(K, adjacency_list[:,0], K_components)   # total LJ force on every atom in sim box

    return -1*K, LJ_pot

def dx_shift(x: np.ndarray, dx: np.ndarray, boxSize: float, boxBounds: np.ndarray):
    """Takes as input displacements 'dx' and coordinates 'x' of atoms in all boxes, and returns shifted 
    displacements 'dx_shifted'. Applying 'dx_shifted' to 'x' gives the positions that one obtains when applying 
    'dx' to 'x' and, if atoms move outside their box on one side, projecting the resulting coordinates back into 
    their corresponding boxes on the opposite side, according to PBC. Thus, 'dx_shifted' is effectively the addition 
    of PBC to displacement 'dx'. The bounds in x-, y- and z-direction of the sim box are contained in 'boxBounds', 
    and the  size of each of the boxes is 'boxSize'."""

    dx_shifted = dx.copy()
    nr_atoms = len(dx)

    for k in np.argwhere(dx > boxSize):   # reduce shift modulo boxSize in magnitude
        n = dx[k[0], k[1]] // boxSize
        dx_shifted[k[0], k[1]] -= boxSize * n

    for k in np.argwhere(dx < -boxSize):   # idem
        m = (-dx[k[0], k[1]]) // boxSize
        dx_shifted[k[0], k[1]] += boxSize * m

    for i in range(0,3):
        for k in np.argwhere((x[0:nr_atoms, :] + dx_shifted)[:,i] > boxBounds[i,1]):   # if atom moves outside box in positive x-, y- or z-direction, make the displacement boxSize smaller in that direction
            dx_shifted[k[0], i] -= boxSize

        for k in np.argwhere((x[0:nr_atoms, :] + dx_shifted)[:,i] < boxBounds[i,0]):   # if atom moves outside box in negative x-, y- or z-direction, make the displacement boxSize larger in that direction
            dx_shifted[k[0], i] += boxSize
         
    dx_shifted = np.array(27 * [dx_shifted]).reshape(27 * nr_atoms, 3)   # make 27 copies so that we can shift atoms in each box as in the sim box

    return dx_shifted



# ----------------- INITIAL VELOCITY -----------------

def initializeVelocity(nrOfAtoms: int):
    """ Returns randomized initial velocities for 'nrOfAtoms' atoms, 
    drawn from uniform distribution on [-1,1]."""
    
    return(np.random.uniform(-1,1,(nrOfAtoms,3)))



# ----------------- INTEGRATORS -----------------

def Euler(x: np.ndarray, x_true: np.ndarray, v: np.ndarray, F: np.ndarray, dt: float, bonds: np.ndarray, bondconsts: np.ndarray, \
    angles: np.ndarray, angleconsts: np.ndarray, dihedrals: np.ndarray, dihedralconsts: np.ndarray, molecule: str, \
    boxSize: float, boxBounds: np.ndarray, sigma_matrix: np.ndarray, epsilon_matrix: np.ndarray, adjacency_list: np.ndarray, amount_of_water: int = 0):
    """Single forward Euler step over time 'dt' for molecules of type 'molecule', or a mixture of water and ethanol with amount 
    "amount_of_water' water molecules. 'x' are the positions of the atoms projected back into their boxes (PBC), whereas 'x_true' are  
    the positions without  projecting back into the boxes. 'v' are the velocities of the atoms, 'F' the total forces on the atoms. 

    'molecule' should be one of the following: 'Water', 'Hydrogen', 'Oxygen', 'Carbonmonoxide', 'Ethanol', 'Water_and_Ethanol'.

    Other parameters are input for the LJ and internal force update; description of these parameters can be found in the docstring for 
    functions 'LJ_forces_PBC' and 'internal_forces'."""

    if amount_of_water == 0: # single molecule type
        m = constantsdict[molecule][0]   # mass
        nr_atoms_molecule = len(atomsdict[molecule])   # atoms per molecule
        nr_atoms = len(x_true)
        nr_molecules = nr_atoms / nr_atoms_molecule   # total number of molecules
        
        a = F.reshape((int(nr_molecules), int(nr_atoms_molecule), 3)) / m   # acceleration
        a = a.reshape(np.shape(F))

    else:   # mixture; same steps for single molecule type, but acceleration is computed for water and ethanol separately
        m_w = constantsdict['Water'][0]
        m_e = constantsdict['Ethanol'][0]
        
        nr_atoms_molecule_w = len(atomsdict['Water'])
        nr_atoms_molecule_e = len(atomsdict['Ethanol'])
        nr_molecules_w = amount_of_water
        nr_atoms_w = nr_molecules_w * nr_atoms_molecule_w
        nr_atoms_e = len(x_true) - nr_atoms_w
        nr_molecules_e = nr_atoms_e / nr_atoms_molecule_e

        a_w = F[0:nr_atoms_w, :].reshape((int(nr_molecules_w), int(nr_atoms_molecule_w), 3)) / m_w
        a_e = F[nr_atoms_w:, :].reshape((int(nr_molecules_e), int(nr_atoms_molecule_e), 3)) / m_e
        a_w = a_w.reshape(np.shape(F[0:nr_atoms_w, :]))
        a_e = a_e.reshape(np.shape(F[nr_atoms_w:, :]))
        a = np.vstack((a_w, a_e))

    dx = dt * v + dt * dt/2 * a   # position update
    dx_shifted = dx_shift(x, dx, boxSize, boxBounds)   # shift dx to take PBC into account

    v = v + dt * a   # update velocity

    x += dx_shifted   # update positions of atoms in all boxes with PBC
    x_true += dx   # update positions of atoms in sim box without PBC
    
    nr_atoms = len(x_true)
    LJ = LJ_forces_PBC(x, adjacency_list, sigma_matrix, epsilon_matrix, nr_atoms)   # update LJ force
    Inner_force = internal_forces(x_true, molecule, bonds, bondconsts, angles, angleconsts, dihedrals, dihedralconsts)   # update internal force
    F = Inner_force[0] + LJ[0]   # total updated force
    
    return x, x_true, v, F, LJ[1], Inner_force[1]

def Velocity_Verlet(x: np.ndarray, x_true: np.ndarray, v: np.ndarray, F: np.ndarray, dt: float, bonds: np.ndarray, bondconsts: np.ndarray, \
    angles: np.ndarray, angleconsts: np.ndarray, dihedrals: np.ndarray, dihedralconsts: np.ndarray, molecule: str, \
    boxSize: float, boxBounds: np.ndarray, sigma_matrix: np.ndarray, epsilon_matrix: np.ndarray, adjacency_list: np.ndarray, amount_of_water: int = 0):
    """Single velocity Verlet step over time 'dt' for molecules of type 'molecule', or a mixture of water and ethanol with amount 
    "amount_of_water' water molecules. 'x' are the positions of the atoms projected back into their boxes (PBC), whereas 'x_true' are  
    the positions without  projecting back into the boxes. 'v' are the velocities of the atoms, 'F' the total forces on the atoms. 

    'molecule' should be one of the following: 'Water', 'Hydrogen', 'Oxygen', 'Carbonmonoxide', 'Ethanol', 'Water_and_Ethanol'.

    Other parameters are input for the LJ and internal force update; description of these parameters can be found in the docstring for 
    functions 'LJ_forces_PBC' and 'internal_forces'."""

    if amount_of_water == 0:   # single molecule type
        m = constantsdict[molecule][0]
        nr_atoms_molecule = len(atomsdict[molecule])   # atoms per molecule
        nr_atoms = len(x_true)
        nr_molecules = nr_atoms / nr_atoms_molecule   # total number of molecules
    
        a = F.reshape((int(nr_molecules), int(nr_atoms_molecule), 3)) / m   # acceleration
        a = a.reshape(np.shape(F))

        dx = dt * v + dt * dt/2 * a   # position update
        dx_shifted = dx_shift(x, dx, boxSize, boxBounds)   # shift dx to take PBC into account
        x += dx_shifted   # update positions of atoms in all boxes with PBC
        x_true += dx   # update positions of atoms in sim box without PBC

        a_old = a.copy()
    
        nr_atoms = len(x_true)
        LJ = LJ_forces_PBC(x, adjacency_list, sigma_matrix, epsilon_matrix, nr_atoms)   # update LJ force
        Inner_force = internal_forces(x_true, molecule, bonds, bondconsts, angles, angleconsts, dihedrals, dihedralconsts)   # update internal force
        F = Inner_force[0] + LJ[0]   # total updated force
        a = F.reshape((int(nr_molecules), int(nr_atoms_molecule), 3)) / m   # update acceleration
        a = a.reshape(np.shape(F))

        v = v + dt/2 * (a + a_old)   # update velocity
    
    else:   # mixture; same steps for single molecule type, but acceleration is computed for water and ethanol separately
        m_w = constantsdict['Water'][0]
        m_e = constantsdict['Ethanol'][0]
        
        nr_atoms_molecule_w = len(atomsdict['Water'])
        nr_atoms_molecule_e = len(atomsdict['Ethanol'])
        nr_molecules_w = amount_of_water
        nr_atoms_w = nr_molecules_w * nr_atoms_molecule_w
        nr_atoms_e = len(x_true) - nr_atoms_w
        nr_molecules_e = nr_atoms_e / nr_atoms_molecule_e

        a_w = F[0:nr_atoms_w, :].reshape((int(nr_molecules_w), int(nr_atoms_molecule_w), 3)) / m_w 
        a_e = F[nr_atoms_w:, :].reshape((int(nr_molecules_e), int(nr_atoms_molecule_e), 3)) / m_e
        a_w = a_w.reshape(np.shape(F[0:nr_atoms_w, :]))
        a_e = a_e.reshape(np.shape(F[nr_atoms_w:, :]))
        a = np.vstack((a_w, a_e))

        dx = dt * v + dt * dt/2 * a
        dx_shifted = dx_shift(x, dx, boxSize, boxBounds)
        x += dx_shifted
        x_true += dx

        a_old = a.copy()
    
        nr_atoms = len(x_true)
        LJ = LJ_forces_PBC(x, adjacency_list, sigma_matrix, epsilon_matrix, nr_atoms)
        Inner_force = internal_forces(x_true, molecule, bonds, bondconsts, angles, angleconsts, dihedrals, dihedralconsts)
        F = Inner_force[0] + LJ[0]
        a_w = F[0:nr_atoms_w, :].reshape((int(nr_molecules_w), int(nr_atoms_molecule_w), 3)) / m_w
        a_e = F[nr_atoms_w:, :].reshape((int(nr_molecules_e), int(nr_atoms_molecule_e), 3)) / m_e
        a_w = a_w.reshape(np.shape(F[0:nr_atoms_w, :]))
        a_e = a_e.reshape(np.shape(F[nr_atoms_w:, :]))
        a = np.vstack((a_w, a_e))

        v = v + dt/2 * (a + a_old)
   
    return x, x_true, v, F, LJ[1], Inner_force[1]

def Leapfrog(x: np.ndarray, x_true: np.ndarray, v: np.ndarray, F: np.ndarray, dt: float, bonds: np.ndarray, bondconsts: np.ndarray, \
    angles: np.ndarray, angleconsts: np.ndarray, dihedrals: np.ndarray, dihedralconsts: np.ndarray, molecule: str, \
    boxSize: float, boxBounds: np.ndarray, sigma_matrix: np.ndarray, epsilon_matrix: np.ndarray, adjacency_list: np.ndarray, amount_of_water: int = 0):
    """Single Leapfrog step over time 'dt' for molecules of type 'molecule', or a mixture of water and ethanol with amount 
    "amount_of_water' water molecules. 'x' are the positions of the atoms projected back into their boxes (PBC), whereas 'x_true' are  
    the positions without  projecting back into the boxes. 'v' are the velocities of the atoms, 'F' the total forces on the atoms. 

    'molecule' should be one of the following: 'Water', 'Hydrogen', 'Oxygen', 'Carbonmonoxide', 'Ethanol', 'Water_and_Ethanol'.

    Other parameters are input for the LJ and internal force update; description of these parameters can be found in the docstring for 
    functions 'LJ_forces_PBC' and 'internal_forces'."""

    if amount_of_water == 0:   # single molecule type
        m = constantsdict[molecule][0]
        nr_atoms_molecule = len(atomsdict[molecule])   # atoms per molecule
        nr_atoms = len(x_true)
        nr_molecules = nr_atoms / nr_atoms_molecule   # total number of molecules
        
        a = F.reshape((int(nr_molecules), int(nr_atoms_molecule), 3)) / m   # acceleration
        a = a.reshape(np.shape(F))

        v_leap = v + a * dt/2    # auxiliary velocity
        dx = v_leap * dt   # position update
        dx_shifted = dx_shift(x, dx, boxSize, boxBounds)   # shift dx to take PBC into account

        x += dx_shifted   # update positions of atoms in all boxes with PBC
        x_true += dx   # update positions of atoms in sim box without PBC
        
        nr_atoms = len(x_true)
        LJ = LJ_forces_PBC(x, adjacency_list, sigma_matrix, epsilon_matrix, nr_atoms)   # update LJ force
        Inner_force = internal_forces(x_true, molecule, bonds, bondconsts, angles, angleconsts, dihedrals, dihedralconsts)   # update internal force
        F = Inner_force[0] + LJ[0]   # total updated force
        a = F.reshape((int(nr_molecules), int(nr_atoms_molecule), 3)) / m   # update acceleration
        a = a.reshape(np.shape(F))

        v = v_leap + a * dt/2   # update velocity

    else:   # mixture; same steps for single molecule type, but acceleration is computed for water and ethanol separately
        m_w = constantsdict['Water'][0]
        m_e = constantsdict['Ethanol'][0]
        
        nr_atoms_molecule_w = len(atomsdict['Water'])
        nr_atoms_molecule_e = len(atomsdict['Ethanol'])
        nr_molecules_w = amount_of_water
        nr_atoms_w = nr_molecules_w * nr_atoms_molecule_w
        nr_atoms_e = len(x_true) - nr_atoms_w
        nr_molecules_e = nr_atoms_e / nr_atoms_molecule_e

        a_w = F[0:nr_atoms_w, :].reshape((int(nr_molecules_w), int(nr_atoms_molecule_w), 3)) / m_w
        a_e = F[nr_atoms_w:, :].reshape((int(nr_molecules_e), int(nr_atoms_molecule_e), 3)) / m_e
        a_w = a_w.reshape(np.shape(F[0:nr_atoms_w, :]))
        a_e = a_e.reshape(np.shape(F[nr_atoms_w:, :]))
        a = np.vstack((a_w, a_e))

        v_leap = v + a * dt/2
        dx = v_leap * dt
        dx_shifted = dx_shift(x, dx, boxSize, boxBounds)

        x += dx_shifted
        x_true += dx
        
        nr_atoms = len(x_true)
        LJ = LJ_forces_PBC(x, adjacency_list, sigma_matrix, epsilon_matrix, nr_atoms)
        Inner_force = internal_forces(x_true, molecule, bonds, bondconsts, angles, angleconsts, dihedrals, dihedralconsts)
        F = Inner_force[0] + LJ[0]
        a_w = F[0:nr_atoms_w, :].reshape((int(nr_molecules_w), int(nr_atoms_molecule_w), 3)) / m_w
        a_e = F[nr_atoms_w:, :].reshape((int(nr_molecules_e), int(nr_atoms_molecule_e), 3)) / m_e
        a_w = a_w.reshape(np.shape(F[0:nr_atoms_w, :]))
        a_e = a_e.reshape(np.shape(F[nr_atoms_w:, :]))
        a = np.vstack((a_w, a_e))

        v = v_leap + a * dt/2

    return x, x_true, v, F, LJ[1], Inner_force[1]



# ----------------- THERMOSTAT TO CONTROL ENERGY -----------------

def thermostat(v: np.ndarray, m: np.ndarray, LJ_pot: float, Inner_pot: float):
    '''Velocity rescaling thermostat given the velocities (v) and mass (m) of the atoms
    the kinetic energy is determined. 
    Then the equipartition theorem is used to derive the temperature (T) and then 
    the velocities are rescaled to match the desired temperature (T_0).
    '''
    
    T_0 = 298.15   # Temperature (K)
    N_f = 3*len(v)   # Dimensionless (We have len(v) free particles)
    k_boltzmann = 0.83145*10**(-2)  # Boltzmann constant (amu*A^2/(0.1ps)^2/K)
    
    E2_kin = np.sum(np.power(LA.norm(v, axis = 1),2)*m)   # 2 times Kinetic Energy (A^2/(0.1ps)^2*amu)
    E_tot = E2_kin/2 + LJ_pot + Inner_pot
    
    T = E2_kin/(N_f*k_boltzmann) 
    
    v = v* np.sqrt(T_0/T)   # Rescaling the velocity
    
    return v, E_tot, E2_kin/2, LJ_pot, Inner_pot, T



# ----------------- FUNCTIONS FOR MEASURING RADIAL DISTRIBUTION DURING SIMULATION -----------------

def radial_bin(dr : float, boxSize : float):
    '''Determines the bins given binsize dr and the boxSize of the radial distribution'''
    
    radius = np.arange(0, boxSize, dr)
    
    if radius.all() <= boxSize:   # adding the last
        radius = np.append(radius, boxSize)
        
    return radius

def radial_dist(molecule : str, x : np.ndarray, nr_atoms : int, radial_bin : np.ndarray, amount_of_water : int):
    '''Creates a histogram of the radial distribution with binsize dr'''
    
    if molecule != 'Water_and_Ethanol':
        
        if molecule == 'Water':
            
            nr_atoms_per_molecule = 3
            molecules = int(nr_atoms/nr_atoms_per_molecule)
            
            O = x[::3]
            O_cut = O[:molecules, :]  # Take only the oxygen atoms in the box
            
            H1 = x[1::3]
            H2 = x[2::3]
            H = np.append(H1, H2, axis = 0)
            
            OO_self = np.array([range(molecules)])   # Creating set indices that need to be removed
            OO_selfXXL = np.array([OO_self]*2).flatten()
            OH_self = OO_self.copy()            
            OH2_self = 27*molecules + OH_self 
            OH_self = np.append(OH_self,OH2_self).copy()
        
            distOO =  distance_matrix(O_cut, O)
            distOO[OO_self[:],OO_self[:]] = 0   # Remove distance between atoms of the same molecule
            
            distOH = distance_matrix(O_cut,H)
            distOH[OO_selfXXL[:],OH_self[:]] = 0   # Remove distance between atoms of the same molecule
        
        
        elif molecule == 'Ethanol':
            
            nr_atoms_per_molecule = 9
            molecules = int(nr_atoms/nr_atoms_per_molecule)
            
            O = x[7::9] 
            O_cut = O[:molecules, :]   # Take only the oxygen atoms in the box
        
            H1 = x[1::9]
            H2 = x[2::9]
            H3 = x[3::9]
            H4 = x[5::9]
            H5 = x[6::9]
            H6 = x[8::9]
            H = np.concatenate((H1, H2, H3, H4, H5, H6), axis = 0)
            
            OO_self = np.array([range(molecules)])   # Creating set indices that need to be removed
            OO_selfXXL = np.array([OO_self]*6).flatten()
            OH_self = OO_self.copy()
            OH_self = np.array([27*molecules*i + OH_self for i in range(6)]).flatten()
         
            distOO =  distance_matrix(O_cut, O)
            distOO[OO_self[:],OO_self[:]] = 0   # Remove distance between atoms of the same molecule
        
            distOH = distance_matrix(O_cut,H)
            distOH[OO_selfXXL[:], OH_self[:]] = 0   # Remove distance between atoms of the same molecule
            
        binOO_index = np.digitize(distOO, radial_bin, right = True)
        binOH_index = np.digitize(distOH, radial_bin, right = True)
        
        histogramOO = np.histogram(binOO_index, bins = range(0, len(radial_bin)+1))[0]
        histogramOH = np.histogram(binOH_index, bins = range(0, len(radial_bin)+1))[0]
           
        return np.array([histogramOO, histogramOH])

    else:   # Mixture
         nr_atoms_per_molecule = 3
         length = len(x)//27
         
         index_w = np.array([range(amount_of_water*3)])   # Indices for seperating water and ethanol molecules
         indexXXL_w = np.array([length*i + index_w for i in range(27)]).flatten()
         
         index_e = amount_of_water*3 + np.array([range(length-amount_of_water*3)])
         indexXXL_e = np.array([length*i + index_e for i in range(27)]).flatten()
         
         x_w = x[indexXXL_w[:]]   # Atoms of the water molecules
                 
         O_w = x_w[::3]
         O_cut_w = O_w[:amount_of_water, :]   # Take only the oxygen atoms of water in the box
         
         H1_w = x_w[1::3]
         H2_w = x_w[2::3]
         H_w = np.append(H1_w, H2_w, axis = 0)
            
         OO_self_w = np.array([range(amount_of_water)])   # Creating set indices that need to be removed
         OO_selfXXL_w = np.array([OO_self_w]*2).flatten()
         OH_self_w = OO_self_w.copy()            
         OH2_self_w = 27*amount_of_water + OH_self_w
         OH_self_w = np.append(OH_self_w,OH2_self_w).copy()
        
         distOO_w =  distance_matrix(O_cut_w, O_w)   # Water-Water 
         distOO_w[OO_self_w[:],OO_self_w[:]] = 0
        
         distOH_w = distance_matrix(O_cut_w,H_w)
         distOH_w[OO_selfXXL_w[:],OH_self_w[:]] = 0
         
         x_e = x[indexXXL_e[:]]   # Atoms of the ethanol molecules
         
         molecules_e = len(x)//27-amount_of_water*3
         
         O_e = x_e[7::9] 
         O_cut_e = O_e[:molecules_e//9, :]   # Take only the oxygen atoms of ethanol in the box
        
         H1_e = x_e[1::9]
         H2_e = x_e[2::9]
         H3_e = x_e[3::9]
         H4_e = x_e[5::9]
         H5_e = x_e[6::9]
         H6_e = x_e[8::9]
         H_e = np.concatenate((H1_e, H2_e, H3_e, H4_e, H5_e, H6_e), axis = 0)
            
         OO_self_e = np.array([range(molecules_e//9)])   # Creating set indices that need to be removed
         OO_selfXXL_e = np.array([OO_self_e]*6).flatten()
         OH_self_e = OO_self_e.copy()
         OH_self_e = np.array([27*molecules_e*i//9 + OH_self_e for i in range(6)]).flatten()
         
         distOO_e =  distance_matrix(O_cut_e, O_e)   # Ethanol-Ethanol
         distOO_e[OO_self_e[:],OO_self_e[:]] = 0   # Remove distance between atoms of the same molecule
         
         distOH_e = distance_matrix(O_cut_e,H_e)
         distOH_e[OO_selfXXL_e[:], OH_self_e[:]] = 0   # Remove distance between atoms of the same molecule
    
         distOO_m = distance_matrix(O_cut_e, O_w)   # Ethanol-Water
         distOH_m = distance_matrix(O_cut_e, H_w)
         
         binOO_w_index = np.digitize(distOO_w, radial_bin, right = True)
         binOH_w_index = np.digitize(distOH_w, radial_bin, right = True)
    
         histogramOO_w = np.histogram(binOO_w_index, bins = range(0, len(radial_bin)+1))[0]
         histogramOH_w = np.histogram(binOH_w_index, bins = range(0, len(radial_bin)+1))[0]
         
         binOO_e_index = np.digitize(distOO_e, radial_bin, right = True)
         binOH_e_index = np.digitize(distOH_e, radial_bin, right = True)
                         
         histogramOO_e = np.histogram(binOO_e_index, bins = range(0, len(radial_bin)+1))[0]
         histogramOH_e = np.histogram(binOH_e_index, bins = range(0, len(radial_bin)+1))[0]
         
         binOO_m_index = np.digitize(distOO_m, radial_bin, right = True)
         binOH_m_index = np.digitize(distOH_m, radial_bin, right = True)
                         
         histogramOO_m = np.histogram(binOO_m_index, bins = range(0, len(radial_bin)+1))[0]
         histogramOH_m = np.histogram(binOH_m_index, bins = range(0, len(radial_bin)+1))[0]
         
         return np.array([histogramOO_w, histogramOH_w, histogramOO_e, histogramOH_e, histogramOO_m, histogramOH_m])
         
def normed(nr_atoms : int, Volume : float, Radial : np.ndarray, out : int, bins : np.ndarray, dr : int, counter : int, step : int, molecule_size : int, mixed = False, nr_atoms_out = 1):
    '''Normalizes the histogram (Radial) of a radial distribution. '''
    
    if mixed == False:
        
        rho = out*nr_atoms//molecule_size/Volume
       
        Radial_normed = np.divide(Radial, nr_atoms//molecule_size*rho*4*np.pi*np.power(bins,2)*dr*counter/step)
    
        return Radial_normed
    
    if mixed == True:
        
        rho = out*nr_atoms_out//molecule_size/Volume
       
        Radial_normed = np.divide(Radial, nr_atoms//molecule_size*rho*4*np.pi*np.power(bins,2)*dr*counter/step)
    
        return Radial_normed
    
def radial_hist(molecule : str, radial_dist : np.ndarray, nr_atoms : int, Volume : float, dr : float, bins : np.ndarray, counter : int, step : int, amount_of_water : int):
   '''Applies the normalization to the histograms corresponding to the radial distributions of each system'''
    
   if molecule == 'Water':
        
        RadialOO_normed = normed(nr_atoms, Volume, radial_dist[0], 1, bins, dr, counter, step,3)
        RadialOH_normed = normed(nr_atoms, Volume, radial_dist[1], 2, bins, dr, counter, step,3)
        
        return [RadialOO_normed, RadialOH_normed]
    
   if molecule == 'Ethanol':
         
        RadialOO_normed = normed(nr_atoms, Volume, radial_dist[0], 1, bins, dr, counter, step,9)
        RadialOH_normed = normed(nr_atoms, Volume, radial_dist[1], 6, bins, dr, counter, step,9)
        
        return [RadialOO_normed, RadialOH_normed]
    
   if molecule == 'Water_and_Ethanol':
        
        RadialOO_w_normed = normed(3*amount_of_water, Volume, radial_dist[0], 1, bins, dr, counter, step,3)
        RadialOH_w_normed = normed(3*amount_of_water, Volume, radial_dist[1], 2, bins, dr, counter, step,3)
        RadialOO_e_normed = normed(nr_atoms - 3*amount_of_water, Volume, radial_dist[2], 1, bins, dr, counter, step,9)
        RadialOH_e_normed = normed(nr_atoms - 3*amount_of_water, Volume, radial_dist[3], 6, bins, dr, counter, step,9)
        RadialOO_m_normed = normed(nr_atoms - 3*amount_of_water, Volume, radial_dist[4], 3*1, bins, dr, counter, step,9, True, 3*amount_of_water)
        RadialOH_m_normed = normed(nr_atoms - 3*amount_of_water, Volume, radial_dist[5], 3*2, bins, dr, counter, step,9, True, 3*amount_of_water)
        
        return [RadialOO_w_normed, RadialOH_w_normed, RadialOO_e_normed, RadialOH_e_normed, RadialOO_m_normed, RadialOH_m_normed]
        



# ----------------- SIMULATION FUNCTION -----------------

def sim(molecule: str, integrator: Callable, dt: float, endTime: float, cutoff_frac: float, box_shift: float, amount_of_water: int = 0):
    """Runs a simulation for molecules of type 'molecule', or a mixture of water and ethanol with 'amount_of_water' water molecules, 
    using integrator 'integrator' with time step 'dt' up to time 'endTime'. This needs a topology "Topology{'molecule'}.txt" and an 
    initial position file "Many{'molecule'}.xyz" to exist in the same directory, which can be achieved by calling the functions 
    'genMolecules' and 'genTopology' for input 'molecule' and the desired number of molecules in the simulation as 'n'. 

    For creating the cubic sim box, the largest positive and negative x-, y- and z-coordinates of atoms in "Many{'molecule'}.xyz" are 
    determined, to which 'box_shift'/2 is added in positive, respectively negative direction. After that, the largest side is taken to 
    be the size of the cubic box, and the other sides are extended symmetrically to that size. For the best results, 'box_shift' should 
    be taken equal to the 'shift' parameter in the function 'genMoleculesCube' in case of a single type of molecule, or equal to the 
    'shift_large' parameter in the same function in case of a water-ethanol mixture. 
    
    The cutoff for the LJ forces is taken to be the size of the box divided by 'cutoff_frac'.

    'molecule' should be one of the following: 'Water', 'Ethanol', 'Water_and_Ethanol'.
    'integrator' should be one of the following: 'Euler', 'Velocity_Verlet', 'Leapfrog'."""

    print('Initializing simulation...')

    intnamesdict = {Euler: 'e', Velocity_Verlet: 'v', Leapfrog: 'l'}

    # ------------- INITIAL DATA AND CONSTANTS -------------

    x = readCoordinates(f'Many{molecule}.xyz')
    x_true = x.copy()
    bonds, bondconsts, angles, angleconsts, sigma_matrix, epsilon_matrix, dihedrals, dihedralconsts = readTopology(f"Topology{molecule}.txt") 
    nr_atoms = len(x)
    v = initializeVelocity(nr_atoms)

    counter = 1
    time = 0
    dr = 0.1   # size of shells for radial distribution
    freq_xyz = 15   # amount of time steps in between consecutive frames in sim file
    freq_adjacency = 50   # adjaceny update frequency
    freq_radial = 250   # radial distribution frequency
    freq_typical_timer = 225   # frequency at which "typical" steps (without adjacency update or radial dsistribution) are timed
    freq_update_timer = 300   # frequency at which "update" steps (with adjacency update, without radial distribution) are timed
    freq_radial_timer = 250   # frequency at which "radial" steps (with adjacency update and radial distribution) are timed
    starting_treshold = 12500   # simulation initializes up to step starting_treshold, after this step thermostat kicks in and collection of data starts
    
    # create arrays of atom masses:
    if amount_of_water == 0:   # single molecule type
        atoms = atomsdict[molecule]
        nr_atoms_per_molecule = len(atoms)
        nr_molecules = int(nr_atoms/nr_atoms_per_molecule)
    
        m = np.array([constantsdict[molecule][0].flatten("F")[:int(nr_atoms_per_molecule)]]*nr_molecules).flatten()   # repeat masses nr_molecules times, to create array with masses of all simulated atoms
    
    else:   # mixture; arrays containing masses of atoms of water molecules, respectively ethanol molecules, are created separately and then stacked
        atoms = [atomsdict['Water'], atomsdict['Ethanol']]
        nr_atoms_w = 3 * amount_of_water
        nr_atoms_e = nr_atoms - nr_atoms_w
        
        nr_atoms_per_molecule_w = len(atoms[0])
        nr_molecules_w = int(nr_atoms_w/nr_atoms_per_molecule_w)
        
        nr_atoms_per_molecule_e = len(atoms[1])
        nr_molecules_e = int(nr_atoms_e/nr_atoms_per_molecule_e)
        
        m_w = np.array([constantsdict['Water'][0].flatten("F")[:int(nr_atoms_per_molecule_w)]]*nr_molecules_w).flatten()
        m_e = np.array([constantsdict['Ethanol'][0].flatten("F")[:int(nr_atoms_per_molecule_e)]]*nr_molecules_e).flatten()
        m = np.concatenate((m_w, m_e), axis = 0)   # array with masses of all simulated atoms

    # ------------- INITIALIZE STORAGE FOR MEASUREMENTS -------------

    Energy = np.array([])
    measure_time = []
    typical_step_time = []
    update_time = []
    update_step_time = []
    radial_time = []
    radial_step_time = []

    # ------------- CREATE SIM BOX -------------

    boxBounds = np.array([[np.min(x[:,0]), np.max(x[:,0])],
    [np.min(x[:,1]), np.max(x[:,1])],
    [np.min(x[:,2]), np.max(x[:,2])]]) + np.array([[-box_shift/2, box_shift/2], [-box_shift/2, box_shift/2], [-box_shift/2, box_shift/2]])   # initial box bounds in x, y, z directions
    
    boxSize = np.max(boxBounds.transpose()[1] - boxBounds.transpose()[0])   # set size of the cubic box to be largest of the 3 directions
    bins = radial_bin(dr, boxSize)   # bins for radial distribution
    
    for i in range(0,3):   # make the box cubic by extending it symmetrically to boxSize in the smaller directions
            length = (boxBounds.transpose()[1] - boxBounds.transpose()[0])[i]
            dl = boxSize - length
            boxBounds[i,0] -= dl/2
            boxBounds[i,1] += dl/2

    cutoff = boxSize / cutoff_frac
    Volume = boxSize**3

    # ------------- CREATE SURROUNDING COPY BOXES -------------

    # shift each atom by amount boxSize in all possible combinations of directions:
    for i in [0,-1,1]:
        for j in [0,-1,1]:
            for k in [0,-1,1]:
                displacement = np.array([i,j,k]) * np.array([boxSize, boxSize, boxSize])   # shift in direction [i,j,k]
                for k in range(0, nr_atoms):
                    x = np.append(x, x[k] + displacement.reshape(1,3), axis=0)   # append shifted coordinate

    x = x[nr_atoms:,:]   # [i,j,k] = [0,0,0] gives duplicate entries for atoms in sim box, so slice off original entries

    # ------------- INITIAL ADJACENCIES FOR LJ FORCES -------------

    index_list = indices(sigma_matrix, epsilon_matrix, nr_atoms)
    adjacency_list = adjacencies(x, cutoff, index_list)

    # ------------- CREATE .XYZ FILE TO WRITE ATOM POSITIONS TO -------------

    sim = open(f"sim_{intnamesdict[integrator]}_{molecule}.xyz","w")
    sim.write('{} \n'.format(nr_atoms))
    sim.write('t= {} \n'.format(time)) 

    # ------------- WRITE INITIAL POSITIONS TO SIM FILE AND COMPUTE INTIAL RADIAL DISTRIBUTION -------------

    if amount_of_water == 0:   # single molecule type
        for i in range(0, nr_atoms, int(len(atoms))):
            for k in range(0, int(len(atoms))):
                sim.write('{}\t {}\t {}\t {}\n'.format(atoms[k], x[i+k][0], x[i+k][1], x[i+k][2]))
        Radial = [np.zeros(len(bins))]*2   # two arrays for OH and OO
    else:   # mixture; first write positions of atoms of water molecules, then of atoms of ethanol molecules
        for i in range(0, nr_atoms_w, 3):
            for k in range(0, 3):
                sim.write('{}\t {}\t {}\t {}\n'.format(atoms[0][k], x[i+k][0], x[i+k][1], x[i+k][2]))
        for i in range(nr_atoms_w, len(x_true), 9):
            for k in range(0, 9):
                sim.write('{}\t {}\t {}\t {}\n'.format(atoms[1][k], x[i+k][0], x[i+k][1], x[i+k][2]))
        Radial = [np.zeros(len(bins))]*6   # six arrays for OH and OO i.e. water-ethanol, water-water, ethanol-ethanol
    
    # ------------- INITIAL FORCES -------------

    F = internal_forces(x_true, molecule, bonds, bondconsts, angles, angleconsts, dihedrals, dihedralconsts)[0] + \
         LJ_forces_PBC(x, adjacency_list, sigma_matrix, epsilon_matrix, nr_atoms)[0]

    # ------------- SIMULATION -------------
        
    while time < endTime:
        step_start_time = clock.time()

        x, x_true, v, F, LJ_pot, Inner_pot = integrator(x, x_true, v, F, dt, bonds, bondconsts, angles, angleconsts, dihedrals, dihedralconsts, \
            molecule, boxSize, boxBounds, sigma_matrix, epsilon_matrix, adjacency_list, amount_of_water)   # single integrator step

        v, E_tot, E_kin, LJ_pot, Inner_pot, T = thermostat(v, m, LJ_pot, Inner_pot)   # velocity rescaling with the thermostat

        # ------------- PERIODIC UPDATES AND MEASUREMENTS -------------

        if counter % freq_adjacency == 0:   # update adjacency list
            update_start_time = clock.time()
            adjacency_list = adjacencies(x, cutoff, index_list)
            update_end_time = clock.time()
            print(f'Updating adjacency list took {update_end_time - update_start_time} s.')
                    
        if counter % freq_radial == 0 and counter > starting_treshold:   # compute radial distribution
            radial_start_time = clock.time()
            Radial += radial_dist(molecule, x, nr_atoms, bins, amount_of_water)
            Energy = np.append(Energy, np.array([E_tot, E_kin, LJ_pot, Inner_pot, T]))
            radial_end_time = clock.time()
            measure_time.append(time)

        if counter % freq_xyz == 0 or counter == 1:   # write positions to sim file
            sim.write('{} \n'.format(nr_atoms))
            sim.write('t= {} \n'.format(time))

            if amount_of_water == 0:   # single molecule type
                for i in range(0, nr_atoms, int(len(atoms))):
                    for k in range(0, int(len(atoms))):
                        sim.write('{}\t {}\t {}\t {}\n'.format(atoms[k], x[i+k][0], x[i+k][1], x[i+k][2]))
            else:   # mixture; first write positions of atoms of water molecules, then of atoms of ethanol molecules 
                for i in range(0, nr_atoms_w, 3):
                    for k in range(0, 3):
                        sim.write('{}\t {}\t {}\t {}\n'.format(atoms[0][k], x[i+k][0], x[i+k][1], x[i+k][2]))
                for i in range(nr_atoms_w, len(x_true), 9):
                    for k in range(0, 9):
                        sim.write('{}\t {}\t {}\t {}\n'.format(atoms[1][k], x[i+k][0], x[i+k][1], x[i+k][2]))

        step_end_time = clock.time()
        
        if counter % freq_typical_timer == 0 and counter > starting_treshold and counter % freq_adjacency != 0:   # time "typical" steps without adjacency update (steps 225, 675, 1125, 1575, ...)
            typical_step_time.append(step_end_time - step_start_time)
        if counter % freq_update_timer == 0 and counter > starting_treshold and counter % freq_radial != 0:   # time "update" steps 300, 600, 900, ...
            update_time.append(update_end_time - update_start_time)
            update_step_time.append(step_end_time - step_start_time)
        if counter % freq_radial_timer == 0 and counter > starting_treshold:   # time "radial" steps 250, 500, 750, ...
            radial_time.append(radial_end_time - radial_start_time)
            radial_step_time.append(step_end_time - step_start_time)

        time += dt
        counter += 1
        print(f'Step {counter} took {step_end_time - step_start_time} s')

    sim.close()
    
    Energy = Energy.reshape(len(Energy)//5,5, order='C')
    Energy = np.transpose(Energy)
    Radial_normed = radial_hist(molecule, Radial, nr_atoms, Volume, dr, bins, counter - starting_treshold//freq_radial, freq_radial, amount_of_water)

    return(Radial_normed, bins, Energy, boxSize, typical_step_time, update_time, update_step_time, radial_time, radial_step_time)



# ----------------- FUNCTIONS FOR STORING AND PROCESSING SIMULATION RESULTS -----------------

def store(file: TextIOWrapper, res: np.ndarray, name: str):
    """Writes 'name' on new line in opened file 'file', 
    and writes items in array 'res' on next line."""

    file.write(f"{name} \n")
    for item in res:
        file.write(f"{item} \t")
    file.write("\n")
    
def store_results(Results: np.ndarray, molecule: str):
    """Stores output (except time measurements) of function 'sim' for molecules of type 
    'molecule', contained in array 'Results', to file named "Results_{'molecule'}.txt"."""

    Res = open(f"Results_{molecule}.txt","w")
    
    if molecule == 'Water' or molecule == 'Ethanol':
        
        # store bins for radial distribution, total energy, kinetic energy, LJ potential, inner force potential, temperature and box size
        bins = Results[1]
        store(Res, bins, 'bins')
        E_tot = Results[2][0]
        store(Res, E_tot, 'E_tot')
        E_kin = Results[2][1]
        store(Res, E_kin, 'E_kin')
        LJ_pot = Results[2][2]
        store(Res, LJ_pot, 'LJ_pot')
        Inner_pot = Results[2][3]
        store(Res, Inner_pot, 'Inner_pot')
        Temperature = Results[2][4]
        store(Res, Temperature, 'Temperature')
        boxSize = Results[3]
        Res.write("boxSize \n")
        Res.write(f"{boxSize} \t")
        Res.write("\n")
        
        # store radial distribution
        OO = Results[0][0]
        store(Res, OO, 'OO')
        OH = Results[0][1]
        store(Res, OH, 'OH')
        
        Res.close()
        
    if molecule == 'Water_and_Ethanol':
         
        # store bins for radial distribution, total energy, kinetic energy, LJ potential, inner force potential, temperature and box size
        bins = Results[1]
        store(Res, bins, 'bins')
        E_tot = Results[2][0]
        store(Res, E_tot, 'E_tot')
        E_kin = Results[2][1]
        store(Res, E_kin, 'E_kin')
        LJ_pot = Results[2][2]
        store(Res, LJ_pot, 'LJ_pot')
        Inner_pot = Results[2][3]
        store(Res, Inner_pot, 'Inner_pot')
        Temperature = Results[2][4]
        store(Res, Temperature, 'Temperature')
        boxSize = Results[3]
        Res.write("boxSize \n")
        Res.write(f"{boxSize} \t")
        Res.write("\n")
        
        # store radial distribution
        OO_w = Results[0][0]
        store(Res, OO_w, 'OO_w')
        OH_w = Results[0][1]
        store(Res, OH_w, 'OH_w')
        OO_e = Results[0][2]
        store(Res, OO_e, 'OO_e')
        OH_e = Results[0][3]
        store(Res, OH_e, 'OH_e')
        OO_m = Results[0][4]
        store(Res, OO_m, 'OO_m')
        OH_m = Results[0][5]
        store(Res, OH_m, 'OH_m')

def store_times(Results: np.ndarray, molecule: str):
    """Stores time measurement output of function 'sim' for molecules of type 
    'molecule', contained in array 'Results', to file named "Times_{'molecule'}.txt"."""

    file = open(f"Times_{molecule}.txt","w")
    typical_step_time, update_time, update_step_time, radial_time, radial_step_time =  Results[4], Results[5], Results[6], Results[7], Results[8]
    store(file, typical_step_time, 'Typical step times')
    store(file, update_time, 'Update times')
    store(file, update_step_time, 'Update step times')
    store(file, radial_time, 'Radial times')
    store(file, radial_step_time, 'Radial step times')

def read_results(molecule):
    Res = open(f"Results_{molecule}2.txt", "r")
    OO, OH, bins, E_tot, E_kin, LJ_pot, Inner_pot, Temperature, Timesteps, Simulationtime, boxSize = [], [], [], [], [], [], [], [], [], [], []
    OO_e, OH_e, OO_m, OH_m = [], [], [], []
    
    for num, line in enumerate(Res):
        if num ==1:
            bins = [float(x) for x in line.split()]
        if num ==3:
            E_tot = [float(x) for x in line.split()]
        if num ==5:
            E_kin = [float(x) for x in line.split()]
        if num ==7:
            LJ_pot = [float(x) for x in line.split()]
        if num ==9:
            Inner_pot = [float(x) for x in line.split()]
        if num ==11:
            Temperature = [float(x) for x in line.split()]
        if num ==13:
            Timesteps = [float(x) for x in line.split()]
        
        if num ==15:
            Simulationtime = [float(x) for x in line.split()]
        if num ==17:
            boxSize = [float(x) for x in line.split()]
        
        if num ==19:
            OO = [float(x) for x in line.split()]
        if num ==21:
            OH = [float(x) for x in line.split()]
        if num ==23:
            OO_e = [float(x) for x in line.split()]
        if num ==25:
            OH_e = [float(x) for x in line.split()]
        if num ==27:
            OO_m = [float(x) for x in line.split()]
        if num ==29:
            OH_m = [float(x) for x in line.split()]
            
    return(bins, E_kin, E_tot, LJ_pot, Inner_pot, Temperature, Timesteps, Simulationtime, boxSize, OO, OH, OO_e, OH_e, OO_m, OH_m)    

def read_times(molecule: str):
    """Reads time measurements stored in file "Times_{'molecule'}.txt" into NumPy arrays."""

    times = open(f'Times_{molecule}.txt', 'r')

    for num, line in enumerate(times):
        if num == 1:
            typical_step = [float(x) for x in line.split()]
        if num == 3:
            update = [float(x) for x in line.split()]
        if num == 5:
            update_step = [float(x) for x in line.split()]
        if num == 7:
            radial = [float(x) for x in line.split()]
        if num == 9:
            radial_step = [float(x) for x in line.split()]

    return typical_step, update, update_step, radial, radial_step
        


# ----------------- PERFORM SIM AND GENERATE RESULTS -----------------

genMoleculesCube(10, 'Water', initial_w, 2.595)
genTopology(10**3, 'Water')
simulation = sim('Water', Velocity_Verlet, 0.005, 1025, 5, 2.595)
store_results(simulation, 'Water')
store_times(simulation, 'Water')

genMoleculesCube(6, 'Ethanol', initial_e, 3.72)
genTopology(6**3, 'Ethanol')
simulation = sim('Ethanol', Velocity_Verlet, 0.002, 1025, 5, 3.72)
store_results(simulation, 'Ethanol')

genMoleculesCube(9, 'Water_and_Ethanol', initial_w_and_e, 2.46, 4.055, 625)
genTopology(9**3, ['Water','Ethanol'], True, 625)
simulation = sim("Water_and_Ethanol", Velocity_Verlet, 0.002, 1025, 5, 4.055, 625) 
store_results(simulation, 'Water_and_Ethanol')
store_times(simulation, 'Water_and_Ethanol')



# ----------------- PROCESS RESULTS -----------------

def Plotter(xlab, xrange,  ylab, pad, yrange, tit, fig, Xinput1, Yinput1, label1, Xinput2, Yinput2, label2, Xinput3, Yinput3, label3):
    plt.figure()
    plt.plot( Xinput1, Yinput1, label = label1, color = 'b', alpha = 0.5, linestyle = 'dashed')
    plt.plot(Xinput2, Yinput2, label = label2, color = 'orange', alpha = 0.5)
    plt.plot(Xinput3, Yinput3, label = label3, color = 'c', alpha = 0.5)
    
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xlabel(xlab)
    plt.ylabel(ylab, labelpad = pad)
    plt.title(tit)
    plt.legend(loc = 1)
    plt.savefig(f"{fig}.png")
    

Results_water = read_results('Water')
Results_ethanol = read_results('Ethanol')
Results_mixture = read_results('Water_and_Ethanol')

### Energy plots ###

E_k_water = np.array(Results_water[1])/(1000*3)
E_t_water = np.array(Results_water[2])/(1000*3)
E_LJ_water = np.array(Results_water[3])/(1000*3)
E_IN_water = np.array(Results_water[4])/(1000*3)
timesteps_water = np.array(Results_water[6])-25 #correct for starting later

print('These are the standard deviations:')
print(np.std(E_k_water), np.std(E_t_water), np.std(E_LJ_water),np.std(E_IN_water), np.std(E_k_water)+np.std(E_LJ_water)+np.std(E_IN_water))

E_k_ethanol = np.array(Results_ethanol[1])/(261*9)
E_t_ethanol = np.array(Results_ethanol[2])/(261*9)
E_LJ_ethanol = np.array(Results_ethanol[3])/(261*9)
E_IN_ethanol = np.array(Results_ethanol[4])/(261*9)
timesteps_ethanol = np.array(Results_ethanol[6])-25

print(np.std(E_k_ethanol), np.std(E_t_ethanol), np.std(E_LJ_ethanol), np.std(E_IN_ethanol), np.std(E_k_ethanol)+np.std(E_LJ_ethanol)+np.std(E_IN_ethanol))

E_k_mixture = np.array(Results_mixture[1])/(625*3+104*9)
E_t_mixture = np.array(Results_mixture[2])/(625*3+104*9)
E_LJ_mixture = np.array(Results_mixture[3])/(625*3+104*9)
E_IN_mixture = np.array(Results_mixture[4])/(625*3+104*9)

print(np.std(E_k_mixture), np.std(E_t_mixture), np.std(E_LJ_mixture), np.std(E_IN_mixture), np.std(E_k_mixture)+np.std(E_LJ_mixture)+np.std(E_IN_mixture))


timesteps_mixture = np.array(Results_mixture[6])-25

Plotter('Time ($0.1$ps)', (0,1000), 'Energy per atom (amu nm$^2$/ps$^2$)', 0, (-8,20), 'Kinetic energy', 'KinEn', timesteps_water, E_k_water, 'Water', timesteps_ethanol, E_k_ethanol, 'Ethanol', timesteps_mixture, E_k_mixture, 'Mixture')
Plotter('Time ($0.1$ps)', (0,1000), 'Energy per atom (amu nm$^2$/ps$^2$)', 0, (-8,20), 'Total energy', 'TotEn', timesteps_water, E_t_water, 'Water', timesteps_ethanol, E_t_ethanol, 'Ethanol', timesteps_mixture, E_t_mixture, 'Mixture')
Plotter('Time ($0.1$ps)', (0,1000), 'Energy per atom (amu nm$^2$/ps$^2$)', 0, (0,5), 'Intra molecular energy', 'IntraEn', timesteps_water, E_LJ_water, 'Water', timesteps_ethanol, E_LJ_ethanol, 'Ethanol', timesteps_mixture, E_LJ_mixture, 'Mixture')
Plotter('Time ($0.1$ps)', (0,1000), 'Energy per atom (amu nm$^2$/ps$^2$)', 0, (-8,20), 'Inner molecular energy', 'InnerEn', timesteps_water, E_IN_water, 'Water', timesteps_ethanol, E_IN_ethanol, 'Ethanol', timesteps_mixture, E_IN_mixture, 'Mixture')



### Temperature ###

temperature_water = Results_water[5]
temperature_ethanol = Results_ethanol[5]
temperature_mixture = Results_mixture[5]

sd_water = np.std(temperature_water)
sd_ethanol = np.std(temperature_ethanol)
sd_mixture = np.std(temperature_mixture)

Plotter('Time ($0.1$ps)', (0,1000), 'Temperature (K)', 4, (296,300), 'Temperature', 'Temp', timesteps_water, temperature_water, 'Water', timesteps_ethanol, temperature_ethanol, 'Ethanol', timesteps_mixture, temperature_mixture, 'Mixture')


### Radial distribution ###

def Plotter2(xlab, xrange,  ylab, yrange, tit, fig, Xinput1, Yinput1, label1, Xinput2, Yinput2, label2):
    plt.figure()
    plt.plot( Xinput1, Yinput1, label = label1, color = 'b')
    plt.plot(Xinput2, Yinput2, label = label2, color = 'orange')
    
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    plt.legend()
    plt.savefig(f"{fig}.png")
    
def Plotter3(xlab, xrange,  ylab, yrange, tit, fig, Xinput1, Yinput1, label1):
    plt.figure()
    plt.plot( Xinput1, Yinput1, label = label1, color = 'b')
    
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    plt.legend()
    plt.savefig(f"{fig}.png")
    
OO_water_water_water = Results_water[9]
OH_water_water_water = Results_water[10]
bins_water = Results_water[0]

OO_ethanol_ethanol_ethanol = Results_ethanol[9]
OH_ethanol_ethanol_ethanol = Results_ethanol[10]
bins_ethanol = Results_ethanol[0]

OO_water_water_mixture = Results_mixture[9]
OH_water_water_mixture = Results_mixture[10]
OO_ethanol_ethanol_mixture = Results_mixture[11]
OH_ethanol_ethanol_mixture = Results_mixture[12]
OO_ethanol_water_mixture = Results_mixture[13]
OH_ethanol_water_mixture = Results_mixture[14]
bins_mixture = Results_mixture[0]

Plotter2('$r(\AA)$', (0,10), '$g_{OO}(r)$', (0,5), 'Radial distribution $O_{water}-O_{water}$', 'rad_dist_OOww', bins_water, OO_water_water_water, 'Water', bins_mixture, OO_water_water_mixture, 'Mixture')
Plotter2('$r(\AA)$', (0,10), '$g_{OH}(r)$', (0,5), 'Radial distribution $O_{water}-H_{water}$', 'rad_dist_OHww', bins_water, OH_water_water_water, 'Water', bins_mixture, OH_water_water_mixture, 'Mixture')
Plotter2('$r(\AA)$', (0,10), '$g_{OO}(r)$', (0,5), 'Radial distribution $O_{ethanol}-O_{ethanol}$', 'rad_dist_OOee', bins_ethanol, OO_ethanol_ethanol_ethanol, 'Ethanol', bins_mixture, OO_ethanol_ethanol_mixture, 'Mixture')
Plotter2('$r(\AA)$', (0,10), '$g_{OH}(r)$', (0,5), 'Radial distribution $O_{ethanol}-H_{ethanol}$', 'rad_dist_OHee', bins_ethanol, OH_ethanol_ethanol_ethanol, 'Ethanol', bins_mixture, OH_ethanol_ethanol_mixture, 'Mixture')

Plotter3('$r(\AA)$', (0,10), '$g_{OO}(r)$', (0,5), 'Radial distribution $O_{ethanol}-O_{water}$', 'rad_dist_OOew', bins_mixture, OO_ethanol_water_mixture, 'Mixture')
Plotter3('$r(\AA)$', (0,10), '$g_{OH}(r)$', (0,5), 'Radial distribution $O_{ethanol}-H_{water}$', 'rad_dist_OHew', bins_mixture, OH_ethanol_water_mixture, 'Mixture')

### Simulation time analysis ###

typical_step, update, update_step, radial, radial_step = read_times('Water_and_Ethanol')
plt.plot(typical_step, color='b')
plt.xlabel('Sample')
plt.ylabel('Time (s)')
plt.title('Typical step times')
plt.savefig('Typical_times_Water_and_Ethanol.png')
plt.clf()

plt.plot(update_step, color = 'b', label = 'Entire step')
plt.plot(update, color = 'orange', label = 'Adjacency list update')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Time (s)')
plt.title('Update step times')
plt.savefig('Update_times_Water_and_Ethanol.png')
plt.clf()

plt.plot(radial_step, color = 'b', label = 'Entire step')
plt.plot(radial, color = 'orange', label = 'Radial distribution')
plt.legend(loc = 'upper left')
plt.xlabel('Sample')
plt.ylabel('Time (s)')
plt.title('Radial distribution step times')
plt.savefig('Radial_times_Water_and_Ethanol.png')