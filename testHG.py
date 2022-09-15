#! python3

############################################################
### LIBRARIES ##############################################
############################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gp import GaussianProcess as GP
from scipy.integrate import ode
from cvxopt import solvers, matrix

#############################################################

#############################################################
### PARAMETERS ##############################################
#############################################################

dim = 2

#############################################################

#############################################################
### FUNCTIONS ###############################################
#############################################################

### DYNAMICS
def dynamics(x, u):
    # Compute the dynamics of the system

    # Reshape vectors
    x = np.reshape(x, (2*dim, 1))
    u = np.reshape(u, (dim, 1))

    A = np.block([
        [np.zeros((dim, dim)), np.eye(dim)],
        [np.zeros((dim, dim)), np.zeros((dim, dim))]
    ])

    B = np.block([
        [np.zeros((dim, dim))],
        [np.eye(dim)]
    ])

    return A@x + B@u
### END DYNAMICS

### H
def h(x, obstacle):
    # Compute the distance of x from obstacle
    p = x[0:dim]   # Extract position
    return np.linalg.norm(p-obstacle['Position'])**2 - obstacle['Radius']**2
### END H

### HDT
def hdt(x, t, obstacle):
    # Compute directional derivative of the distance
    p=x[0:dim]
    dh = 2*(p-obstacle['Position'])

    dh = np.reshape(dh, (1, dim))
    # Control law
    w = 1
    u = np.array([-w**2*np.cos(w*t), w**2*np.sin(w*t)])

    dx = dynamics(x[0:2*dim], u)
    dx = dx[0:dim]
    dx = np.reshape(dx, (dim, 1))
    
    return dh@dx
### END HDT 
    
### DFITTER
def dfitter(fitter, x, t):
    # Compute direction derivative of the fitter
    # Control law
    w = 1
    u = np.array([-w**2*np.cos(w*t), w**2*np.sin(w*t)])

    dx = dynamics(x[0:2*dim], u)
    dx = dx[0:dim]

    return fitter.posterior_dtmean(x[0:dim], dx)
    
### ENDDFITTER

### SIMULATION
def simulation(t, x, obstacles):
    # Compute dx(t) for the ode solver
    
    # Control law
    w = 1
    u = np.array([-w**2*np.cos(w*t), w**2*np.sin(w*t)])

    dx = dynamics(x[0:2*dim], u)

    # Observer on minimum distance dynamics
    obs_dists = []
    for obs in obstacles:
        obs_dists.append( h(x, obs) )
    dz = HGobserverDynamics(x[-2:], min(obs_dists) )

    return np.append(dx, dz)
### END SIMULATION

### HGOBSERVERDYNAMICS
def HGobserverDynamics(z, y):
    l = 10.0
    lambda1 = 3.0
    lambda2 = 4.0

    k1 = lambda1+lambda2
    k2 = lambda1*lambda2

    dz = np.array([0, 0])
    dz[0] = z[1] + l*k1*(y - z[0])
    dz[1] = l**2*k2*(y - z[0])
    return dz
# END HGOBSERVERDYNAMICS


#############################################################

#############################################################
### MAIN ####################################################
#############################################################

def main():
    ### Initial condition
    x0 = np.array([1, 0, 0, 0])

    ### Obstacles

    # Define obstacles
    obs_num = 4
    obstacles = []
    obstacles.append({'Position': np.array([2, 2]), 'Radius': 0.5})
    obstacles.append({'Position': np.array([0.3, -2]), 'Radius': 0.5})
    obstacles.append({'Position': np.array([-0.5, -2]), 'Radius': 0.5})
    obstacles.append({'Position': np.array([-2, 2]), 'Radius': 0.5})
    
    # Define circles for plots
    circles = []
    for obs in obstacles:
        circles.append( plt.Circle(obs['Position'], obs['Radius'], color='r') )

    ### Gaussian processes initialiation
    fitterH = GP(dim, 1)            # Function
    fitterHD = GP(2*dim, 1)         # Time Derivative
    fitterH.set_hyperparams(np.ones((2*dim,))*0.2)
    fitterHD.set_hyperparams(np.ones((2*dim,))*0.2)

    # Add initial value to GPs
    fitterHD.add_sample(x0, np.array([0]))
    fitterHD.train()
    
    # Find distances
    obs_dists = []
    for obs in obstacles:
        obs_dists.append( h(x0, obs) )
    fitterH.add_sample(x0[0:2], min(obs_dists)) # Add minimum distance     
    fitterH.train()                             # Train GP

    ### ODE Solver
    r0 = np.append(x0, min(obs_dists))
    r0 = np.append(r0, 0)
    dt = 0.01                                               # Time step
    f = lambda t,x : simulation(t, x, obstacles)            # Lambda function to incorporate controllre and BF in the ode solver
    r = ode(f).set_integrator('dopri5', nsteps=10000)       # Set solver
    r.set_initial_value(r0, 0)                              # Set initial condition

    x = r0.reshape((2*dim+2, 1))                # State Trajectory
    hh = np.array([min(obs_dists)])             # Distance
    dhh = np.array([0])                         # Distance Derivative
    dff = np.array([0])                         # Fitter derivative
    while r.successful() and r.t < 2*3.14:
        # Integrate system equations
        print(r.t)
        r.integrate(r.t+dt)                                 # Numerically solve ODE for dt seconds
        x = np.append(x, (r.y).reshape((2*dim+2,1)), 1)     # Add actual state to state trajectory

        # Compute minimum distance
        obs_dists = []                              
        for obs in obstacles:
               obs_dists.append( h(r.y[0:dim], obs) )
        index_min = min(range(len(obs_dists)), key=obs_dists.__getitem__)
        hh = np.append(hh, min(obs_dists))
        dhh = np.append(dhh, hdt(r.y, r.t, obstacles[index_min]))
        dff = np.append(dff, dfitter(fitterH, r.y, r.t))

        # Add sample to GP
        # Check variance of current point
        if fitterH.posterior_variance(r.y[0:dim]) > 0.1 : 
            fitterH.add_sample(r.y[0:2], min(obs_dists))    # Add minimum distance     
            fitterH.train()                                 # Train GP
        if fitterHD.posterior_variance(r.y[0:2*dim]) > 0.1:
            fitterHD.add_sample(r.y[0:2*dim], r.y[2*dim+1])
            fitterHD.train()
    
    ### Compute the estimated derivative along the followed trajectory
    hdf = np.array([0])
    for i in range(x.shape[1]-1):
        hdf = np.append(hdf, fitterHD.posterior_mean(x[0:2*dim, i]))
            

    ### Plots

    # Define figure and axis for trajectory
    fig, ax = plt.subplots()

    # Add obstacles to plot
    for circle in circles:
        ax.add_patch(circle) 

    # Plot trajectory
    plt.plot(x[0, :], x[1, :], label='Trajectory')
    plt.plot(x0[0], x0[1], 'o', label='Starting Position')
    
    # Set plot limits
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    plt.legend()
    
    # Prepare GP plot
    dom_x = np.arange(-4, 4, 0.2)
    dom_y = np.arange(-4, 4, 0.2)
    xx, yy = np.meshgrid(dom_x, dom_y)
    zz = np.empty(xx.shape)
    zv = np.empty(xx.shape)
    zh = np.empty(xx.shape)
    for idx_x in range(dom_x.shape[0]):
        for idx_y in range(dom_y.shape[0]):
            zz[idx_x, idx_y] = fitterH.posterior_mean(np.array([dom_x[idx_x], dom_y[idx_y]]))
            zv[idx_x, idx_y] = fitterH.posterior_variance(np.array([dom_x[idx_x], dom_y[idx_y]]))
            obs_dists = []                              # Compute distances from obstacles
            for obs in obstacles:
                obs_dists.append( h(np.array([dom_x[idx_x], dom_y[idx_y]]), obs) )
            zh[idx_x, idx_y] = min( obs_dists)
    
    # GP plot
    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.contour3D(xx, yy, zz, 25, cmap='viridis', edgecolor='none')
    ax1.contour3D(xx, yy, zh, 25, cmap='viridis', edgecolor='none')
    plt.grid()

    # HG plot
    fig2, ax2 = plt.subplots()
    plt.plot(x[-2, :], label='Distance HG')
    plt.plot(hh, label='Distance')
    plt.legend()

    fig3, ax3 = plt.subplots()
    plt.plot(hh - x[-2, :], label='HG H Error')
    plt.legend()

    fig4, ax4 = plt.subplots()
    plt.plot(x[-1, :], label='HG Derivative')
    plt.plot(dhh, label='Real Derivative')
    plt.plot(dff, label='GP Derivative')
    plt.legend()

    fig5, ax5 = plt.subplots()
    plt.plot(hdf, label='HG GP')
    plt.plot(dhh, label='Real Derivative')
    plt.plot(dff, label='GP Derivative')
    plt.legend()

    fig6, ax6 = plt.subplots()
    plt.plot(dhh - hdf, label='HG GP Error')
    plt.plot(dhh - dff, label='GP Derivative Error')
    plt.legend()

    # Show plots
    plt.show()
    
if __name__ == '__main__':
    main()