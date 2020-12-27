__doc__ = """
Wang-Landau algorithm for 2D Ising model
"""

import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import numba as nb


@nb.jit(boundscheck=False, fastmath=True)  # this will yield a 40% speedup
def single_step(N, current_hamiltonian, f, sigma, logOmega, histogram, tot_spin, tot_histogram):
    ret = current_hamiltonian
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    new = -sigma[i, j]  # exactly what a flip should look like
    
    # calculate energy difference
    delta_hamiltonian = 0
    for dir in [(-1,0), (1,0), (0,1), (0,-1)]:
        adjacent = ((i+dir[0]) % N, (j+dir[1]) % N)
        delta_hamiltonian -= (1 if new == sigma[adjacent] else -1)  # if new spin is equal to the adjacent one, then the old spin is definitely not
    proposed_hamiltonian = current_hamiltonian + delta_hamiltonian // 2 
    
    if logOmega[current_hamiltonian] > logOmega[proposed_hamiltonian] \
        or np.random.random() < np.exp(logOmega[current_hamiltonian]-logOmega[proposed_hamiltonian]): 
        ret = proposed_hamiltonian 
        sigma[i, j] = new 

    logOmega[current_hamiltonian] += f 
    histogram[current_hamiltonian] += 1
    tot_histogram[current_hamiltonian] += 1
    tot_spin[current_hamiltonian] += np.sum(sigma)
    return ret 


class WangLandauSampling():
    """
    Wang-Landau Algorithm for 2D N*N Ising model. 

    J=1 is assumed throughout this code.

    The energy evidently lies between [-2N^2, 2N^2], and is equally distributed with distance = 2; 
    thus, if we want to map -2N^2:2N^2:2 to 0:N^2:1, which is the index of DOS array, a function: 
        e -> (e/2+N^2) / 2 
    is all we need. 

    So, in this code, all Hamiltonians appeared are in fact rescaled in this way. 
    """
    def __init__(self, N):
        self.N = N 
        self.sigma = 2 * np.random.randint(0, 2, (N, N)) - 1 # generate random spins  
        self.f = 1.
        self.current_hamiltonian = self.get_hamiltonian()

        self.logOmega = np.zeros(N*N + 1)
        self.histogram = np.zeros(N*N + 1)
        self.histogram[1] = self.histogram[-2] = np.nan   # these two energies cannot be achieved
        
        self.tot_histogram = np.zeros(N*N + 1)
        self.tot_spin = np.zeros(N*N + 1)  # save total spin 

        # if run on local, uncommment the following statement
        self.prepare_auto_update()
        

    def get_hamiltonian(self):
        horz = self.sigma * np.roll(self.sigma, -1, 0)
        vert = self.sigma * np.roll(self.sigma, 1, 1)
        return ((-np.sum(horz) -np.sum(vert)) // 2 + self.N * self.N) // 2  # rescale the hamiltonian


    def sample(self, threshold=1e-8):
        tot_steps = 0
        while self.f > threshold: 
            self.current_hamiltonian = single_step(self.N, self.current_hamiltonian, self.f, self.sigma, self.logOmega, self.histogram, self.tot_spin, self.tot_histogram)
            
            # if run on local, uncomment the following
            if not tot_steps % 250:  
                self.show_histogram(str(tot_steps // 250))
                
            
            if self.is_flat_enough():
                # self.show_histogram()
                # self.show_DOS()
                self.histogram = np.zeros(self.N*self.N+1) 
                self.histogram[1] = self.histogram[-2] = np.nan
                print(self.f)
                self.f /= 2.
            
            tot_steps += 1


        print('Total iterations: ', tot_steps)

    def is_flat_enough(self):
        return np.nanmin(self.histogram) > .8 * np.nanmean(self.histogram) 

    def prepare_auto_update(self): 
        """
        Get a visualization of what is going on
        """
        plt.ion()
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        self.line1, = ax.plot(self.logOmega, 'ro-') # Returns a tuple of line objects, thus the comma
        self.line2, = ax.plot(self.logOmega, 'r-')
        plt.ylim(0,2000)

    def do_statistics(self, funcs, beta):
        """
        Evidently, statistics are functions of energy, since we are dealing with micro-canonical ensemble. 
        """

        # normalize probability 
        self.logOmega -= np.max(self.logOmega) 
        rets = []
        for func in funcs: 
            logminus, logplus = -100, -100  # this would allieviate overflow to a certain extent, but no matter
            for i in range(self.N*self.N + 1): 
                e = (2*i - self.N * self.N) * 2 
                if func(e) > 0.:
                    logplus = np.logaddexp(logplus, np.log(func(e))+self.logOmega[i] - beta * e)
                elif func(e) < 0.:    
                    logminus = np.logaddexp(logminus, np.log(-func(e))+self.logOmega[i] - beta * e)

            rets.append(np.exp(logplus) - np.exp(logminus))
        return np.array(rets)


    def stat_on_spin(self, beta):
        # normalize probability 
        self.logOmega -= np.max(self.logOmega) 
        ret = 0. 
        logminus, logplus = -100, -100  # this would allieviate overflow to a certain extent, but no matter
        self.tot_spin /= self.tot_histogram * self.N ** 2
        for i in range(self.N*self.N + 1): 
            e = (2*i - self.N * self.N) * 2 
            if np.isnan(self.tot_spin[i]): continue
            if self.tot_spin[i] > 0.:
                logplus = np.logaddexp(logplus, np.log(self.tot_spin)+self.logOmega[i] - beta * e)
            elif self.tot_spin[i] < 0.:    
                logminus = np.logaddexp(logminus, np.log(-self.tot_spin)+self.logOmega[i] - beta * e)

        return np.array(ret)


    def show_histogram(self, label=None):
        self.line1.set_ydata(self.histogram)
        self.line2.set_ydata(.8 * np.nanmean(self.histogram) * np.ones_like(self.histogram))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        '''
        # plt.figure()
        plt.plot(self.histogram, 'ro-')
        plt.plot(.8 * np.nanmean(self.histogram) * np.ones_like(self.histogram), 'r-')
        plt.ylim(0,1500)
        plt.xlim(0,self.N*self.N+1)
        # plt.show() 
        plt.tight_layout()
        plt.savefig('snapshots/%s.png' % label, dpi=72)
        plt.cla()
        '''


    def show_spin(self):
        plt.matshow(self.sigma) 
        plt.show()

    def show_DOS(self):
        plt.ioff()
        plt.figure()
        plt.plot(self.logOmega)
        plt.show()

    

if __name__ == '__main__':
    wls = WangLandauSampling(4)
    wls.sample(1e-3) 
    # wls.logOmega = np.genfromtxt('logOmega.dat')
    wls.show_DOS()
    stats = []
    tot_spin = []
    betas = np.linspace(.2,1.5,501)
    for beta in betas:
        stats.append(wls.do_statistics([lambda _: 1, lambda _: _, lambda _: _ ** 2], beta))
        tot_spin.append(wls.stat_on_spin(beta))
    tot_spin = np.array(tot_spin)
    stats = np.array(stats) 
    plt.plot(betas, tot_spin / stats[:, 0])
    plt.show()
    plt.plot(betas, stats[:,1] / stats[:,0])
    plt.show()
    plt.plot(betas, betas**2*(stats[:,2] / stats[:, 0] - (stats[:,1] / stats[:, 0])**2))
    plt.show()
    