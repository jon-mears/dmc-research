import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde

def SHO(x):
    """
    SHO potential
    """
    return x**2/2

def doMCSteps(mcsteps=1000, ds=0.1, nWalkers=1000, minX=-3.0, maxX=3.0,
              numBins=30, V=SHO, oldWalkers=None, oldHist=None, alpha=1.0):
    vrefs = [] 
    cum_avg_vrefs = []
    diffs = []
    exact_value = 0.5  # the exact value for comparison, set as appropriate
    n0 = nWalkers 
    dt = ds**2  
    if oldWalkers is not None:
        Walkers = oldWalkers 
        nw = len(Walkers)
    else:
        Walkers = np.linspace(minX, maxX, n0) 
        nw = n0
    if oldHist is not None:
        h = oldHist 
    elif numBins > 0:
        h = np.zeros(numBins)  
    else:
        h = None
    vref = V(Walkers).sum() / nw  
    cumulative_sum_vref = 0.0  # Initialize the sum of vrefs for the cumulative average

    for i in range(mcsteps):
        Walkers += np.random.normal(size=nw) * ds  # Bump walkers
        dv = V(Walkers) - vref  # Change in V
        m_n = np.array(1.0 - dv * dt + np.random.rand(nw), int)  # Birth/death parameter
        k = Walkers.take(np.flatnonzero(m_n > 0))  # Keep walkers where m_n > 0
        d = Walkers.take(np.flatnonzero(m_n > 1))  # Duplicate those where m_n > 1
        Walkers = np.append(k, d)  # Rejoin the walkers
        nw = len(Walkers)  # Current number of walkers
        dn = nw - n0  # Difference from original number
        vavg = V(Walkers).sum() / nw  # Average potential energy of walkers
        vref = vavg - alpha * (dn / (n0 * dt))  # Adjust vref
        
        # Save the current vref and calculate cumulative average and difference
        vrefs.append(vref)
        cumulative_sum_vref += vref
        cum_avg = cumulative_sum_vref / (i + 1)
        diff = cum_avg - exact_value
        cum_avg_vrefs.append(cum_avg)
        diffs.append(diff)

        # Print the current step's details
        print(f"{i}\t{vref:.8f}\t{cum_avg:.8f}\t{exact_value:.8f}\t{diff:.8f}")

        if numBins > 0:  # Binning
            hvals, binArray = np.histogram(Walkers, bins=numBins, range=(minX, maxX))
            h += hvals  # Sum the bins
    
    # Return all the calculated values along with the histogram and walker positions
    return h, binArray, vrefs, Walkers, hvals, cum_avg_vrefs, diffs

# Example of calling the function
histogram, bins, vrefs, walkers, last_hvals, cum_avg_vrefs, diffs = doMCSteps(mcsteps=200, ds=0.5)



result = doMCSteps()
# This will run the Monte Carlo simulation with the default parameters.
h, binArray, vrefs, Walkers, hvals, cum_avg_vrefs, diffs = result

# Plotting the final distribution of walkers
'''plt.figure(figsize=(10, 5))
plt.bar(binArray[:-1], h, width=(binArray[1] - binArray[0]), color='blue', alpha=0.7)
plt.title("Final Distribution of Walkers")
plt.xlabel("Position")
plt.ylabel("Count")
plt.show()'''

# Plotting the evolution of reference energy
plt.figure(figsize=(10, 5))
plt.plot(vrefs, color='red')
plt.title("Evolution of Reference Energy")
plt.xlabel("Monte Carlo Step")
plt.ylabel("Reference Energy")
plt.show()

kde = gaussian_kde(Walkers, bw_method=0.015)
x_grid = np.linspace(-3, 3, 1000)
kde_values = kde.evaluate(x_grid)

# Plotting the density of walkers
plt.figure(figsize=(10, 5))
plt.plot(x_grid, kde_values, color='blue')
plt.title("Walker Density")
plt.xlabel("x")
plt.ylabel("ρ(x)")
plt.grid(False)
plt.show()



