import numpy as np
import matplotlib.pyplot as plt 

class Force():
    ''' Count the total invocations of force calculation. Note that this feature is not covered in my report '''
    def __init__(self):
        self.count = 0 
    
    def __call__(self, r):
        self.count += 1 
        return -r  # in harmonic oscillator, force takes a very simple form


def velocity_verlet(r, p, f, s): 
    """The velocity Verlet integrator
    
    Parameters: 
    r: position 
    p: momentum 
    f: force function 
    s: time step 
    """
    
    p += f(r) * s / 2. 
    r += p * s 
    p += f(r) * s / 2. 
    return r, p


theta = 1/(2-2**(1/3))  # a constant essential for FR to work
def forest_ruth(x, p, f, s): 
    ''' The Forest Ruth integrator '''
    x += theta * p * s / 2. 
    p += theta * f(x) * s 
    x += (1.-theta) * p * s / 2. 
    p += (1.-2*theta) * f(x) * s
    x += (1.-theta) * p * s / 2.  
    p += theta * f(x) * s 
    x += theta * p * s / 2. 
    return x, p


class Solution():
    ''' The MD simulator ''' 
    def __init__(self, step=1e-2, func=velocity_verlet):
        self.step = step 
        self.x = 1.  # initial position 
        self.p = 0.  # inital momentum
        self.evolve_func = func
        self.force = Force()
        self.xs = [1.]  # stores position after each step
        self.es = []  # stores energy after each step
    
    def evolve(self, n=20000):
        for _ in range(n):
            self.x, self.p = self.evolve_func(self.x, self.p, self.force, self.step) 
            self.xs.append(self.x)
            self.es.append(self.x ** 2 * .5 + self.p ** 2 * .5)
        
        self.xs = np.array(self.xs)  # make them a NumPy brings convenience to manipulations afterwards
        self.es = np.array(self.es) 


def plot_position_error(): 
    ts, tot_steps = 0.1, 200000

    sol = Solution(ts, velocity_verlet,)
    sol.evolve(tot_steps)
    plt.plot(np.arange(tot_steps+1)*ts/(2*np.pi), np.abs(sol.xs-np.cos(ts*np.arange(tot_steps+1))), label='Velocity Verlet')

    sol = Solution(ts, forest_ruth,)
    sol.evolve(tot_steps)
    plt.plot(np.arange(tot_steps+1)*ts/(2*np.pi), np.abs(sol.xs-np.cos(ts*np.arange(tot_steps+1))), label='Forest Ruth')

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.show()  # note that you shouldn't save PDF style, vector graph of this size would take forever to render


def plot_DeltaE(): 
    ts, tot_steps = 0.01, 20000

    sol = Solution(ts, velocity_verlet,)
    sol.evolve(tot_steps)
    plt.plot(np.arange(tot_steps)*ts/(2*np.pi), np.cumsum(np.abs(sol.es - .5) / .5)/(1+np.arange(tot_steps)), label='Velocity Verlet')

    sol = Solution(ts, forest_ruth,)
    sol.evolve(tot_steps)
    plt.plot(np.arange(tot_steps)*ts/(2*np.pi), np.cumsum(np.abs(sol.es - .5) / .5)/(1+np.arange(tot_steps)), label='Forest Ruth')
    
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show() 


def plot_DeltaEb(): 
    ps = []  # save plot handles
    for ts, alpha in zip([1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1], [.9,.8,.7,.6,.5,.4,.3]):
        tot_steps = int(20. / ts)
        sol = Solution(ts, velocity_verlet,)
        sol.evolve(tot_steps)
        ps.append(plt.plot(np.arange(tot_steps)*ts/(2*np.pi), np.cumsum(np.abs(sol.es - .5) / .5)/(1+np.arange(tot_steps)), c='blue', alpha=alpha, label='Velocity Verlet %.1e'%(ts))[0])

    for ts, alpha in zip([1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1], [.9,.8,.7,.6,.5,.4,.3]):
        tot_steps = int(20. / ts)
        sol = Solution(ts, forest_ruth,)
        sol.evolve(tot_steps)
        ps.append(plt.plot(np.arange(tot_steps)*ts/(2*np.pi), np.cumsum(np.abs(sol.es - .5) / .5)/(1+np.arange(tot_steps)),  c='orange', alpha=alpha, label='Forest Ruth %.1e'%(ts))[0])
        
    plt.yscale('log')
    plt.legend(handles=ps,bbox_to_anchor=(1.05, 1), loc='upper left',)  # legend outside main plot
    plt.grid()
    plt.show() 


def plot_DeltaEb2(): 
    ps = []  # save plot handles
    for ts in [1e-4,5e-4,1e-3]:
        tot_steps = int(80. / ts)
        sol = Solution(ts, forest_ruth,)
        sol.evolve(tot_steps)
        ps.append(plt.plot(np.arange(tot_steps)*ts/(2*np.pi), np.cumsum(np.abs(sol.es - .5) / .5)/(1+np.arange(tot_steps)), label='%.1e'%(ts))[0])
        
    plt.yscale('log')
    plt.legend() 
    plt.grid()
    plt.show() 


def plot_DeltaEc(): 
    vv = []
    ts_list = np.exp(np.linspace(-2,-8, num=15))
    for ts in ts_list:
        tot_steps = int(20. / ts)
        last_few_steps = int(1. / ts) 
        sol = Solution(ts, velocity_verlet,)
        sol.evolve(tot_steps)
        vv.append(np.mean((np.cumsum(np.abs(sol.es - .5) / .5)/(1+np.arange(tot_steps)))[-last_few_steps:]))

    fr = []
    for ts in ts_list:
        tot_steps = int(20. / ts)
        sol = Solution(ts, forest_ruth,)
        sol.evolve(tot_steps)
        fr.append(np.mean((np.cumsum(np.abs(sol.es - .5) / .5)/(1+np.arange(tot_steps)))[-last_few_steps:]))
    
    plt.plot(ts_list, vv, '+-', label='Velocity Verlet')
    plt.plot(ts_list, fr, '+-', label='Forest Ruth')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend() 
    plt.grid()
    plt.show()  # this plot can be safely saved as PDF


def plot_DeltaEc_large_step(): 
    vv = []
    ts_list = np.exp(np.linspace(-1,1, num=40))
    for ts in ts_list:
        tot_steps = int(20. / ts)
        last_few_steps = int(1. / ts) 
        sol = Solution(ts, velocity_verlet,)
        sol.evolve(tot_steps)
        vv.append(np.mean((np.cumsum(np.abs(sol.es - .5) / .5)/(1+np.arange(tot_steps)))[-last_few_steps:]))

    fr = []
    for ts in ts_list:
        tot_steps = int(20. / ts)
        sol = Solution(ts, forest_ruth,)
        sol.evolve(tot_steps)
        fr.append(np.mean((np.cumsum(np.abs(sol.es - .5) / .5)/(1+np.arange(tot_steps)))[-last_few_steps:]))
    
    plt.plot(ts_list, vv, '+-', label='Velocity Verlet')
    plt.plot(ts_list, fr, '+-', label='Forest Ruth')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend() 
    plt.grid()
    plt.show()  # this plot can be safely saved as PDF


if __name__ == '__main__':
    plot_DeltaEb2()
    

