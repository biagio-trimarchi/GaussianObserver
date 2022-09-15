#! python3

############################################################
### LIBRARIES ##############################################
############################################################

import os
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
    return np.linalg.norm(p-obstacle['Position'])**2 - (obstacle['Radius'] +0.2)**2
### END H

### HGRADIENT
def hgradient(x, obstacle):
    p = x[0:dim]
    dh = np.block([[2*(p-obstacle['Position'].reshape((2,1)))], [np.zeros((2,1))]])
    return dh
### END HGRADIENT

### HHESSIAN
def hhsessian(x):
    p = x[0:dim]
    ddh = 2*np.block([
                    [np.diag(p.squeeze()), np.zeros((2, 2))], 
                    [np.zeros((2, 2)), np.zeros((2, 2))]
                ])
    return ddh
### END HHESSIAN

### HDT
def hdt(x, obstacle, pGoal, fitterH, fitterHD):
    # Compute directional derivative of the distance
    p=x[0:dim]
    dh = 2*(p-obstacle['Position'])

    dh = np.reshape(dh, (1, dim))
    # Control law
    # w = 1
    # u = np.array([-w**2*np.cos(w*t), w**2*np.sin(w*t)])
    u = controller(x[0:2*dim], pGoal, fitterH, fitterHD, obstacle)


    dx = dynamics(x[0:2*dim], u)
    dx = dx[0:dim]
    dx = np.reshape(dx, (dim, 1))
    
    return dh@dx
### END HDT 
    
### DFITTER
def dfitter(fitter, x, pGoal, fitterH, fitterHD, obstacle):
    # Compute direction derivative of the fitter
    # Control law
    u = controller(x[0:2*dim], pGoal, fitterH, fitterHD, obstacle)

    dx = dynamics(x[0:2*dim], u)
    dx = dx[0:dim]

    return fitter.posterior_dtmean(x[0:dim], dx)
### ENDDFITTER

### CONTROLLER
def controller(x, pGoal, fitterH, fitterHD, obstacle):
    # Compute the control law
    
    # Reshape Vectors
    x = np.reshape(x, (2*dim, 1))
    pGoal = np.reshape(pGoal, (dim, 1))

    # Extract Position and Velocity (for clarity sake)
    p = x[0:dim]
    v = x[dim:2*dim]

    # Controller Parameters
    Kp = 1
    Kv = 2

    # Simple PD Controller
    uNom = -Kp*(p-pGoal) - Kv*v
    
    # Barrier Filter (Exponential CBF)

    # CBF Paramters
    k1 = 1      # Alpha function multiplier h
    k2 = 1      # Alpha function multiplier dh/dt 

    K = np.array([k1, k2])
    K = K.reshape((2, 1))

    hh = fitterH.posterior_mean(p)
    hhd = fitterH.posterior_dxmean(p)
    hhd = np.block([ 
        [hhd],
        [np.zeros((2,1))]
    ])

    hhdd = fitterH.posterior_ddxmean(p)
    hhdd = np.block([ 
        [hhdd, np.zeros((2,2))],
        [np.zeros((2,2)), np.zeros((2,2))]
    ])
    #hhd = fitterHD.posterior_mean(x)

    #H = np.array([hh, hhd])
    #H = H.reshape((2, 1))

    # Plant Dynamics ( A = f(x), B = g(x) )
    Adyn = np.block([
        [np.zeros((dim, dim)), np.eye(dim)],
        [np.zeros((dim, dim)), np.zeros((dim, dim))]
    ])

    Bdyn = np.block([
        [np.zeros((dim, dim))],
        [np.eye(dim)]
    ])

    # Gradient 
    gradH = fitterHD.posterior_dxmean(x)
    #hh = h(x.squeeze(), obstacle)
    dh = hgradient(x, obstacle)
    ddh = hhsessian(x)

    #print(hhd, dh)

    #print(k2*hhd - k2*dh.T@Adyn@x)

    # Define A and B for Quadratic Programming
    #print('Error1 = ', (x.T@Adyn.T@ddh + dh.T@Adyn) - gradH.T)
    #print('Error2 =' ,x.T@Adyn.T@ddh@Adyn@x + dh.T@Adyn@Adyn@x + k1*h(x.squeeze(), obstacle) + k2*dh.T@Adyn@x - K.T@H + gradH.T@Adyn@x )
    A = - ( x.T@Adyn.T@ddh@Bdyn + dh.T@Adyn@Bdyn)
    A = - (x.T@Adyn.T@hhdd@Bdyn + hhd.T@Adyn@Bdyn)
    B = x.T@Adyn.T@ddh@Adyn@x + dh.T@Adyn@Adyn@x + k1*h(x.squeeze(), obstacle) + k2*dh.T@Adyn@x
    B = x.T@Adyn.T@hhdd@Adyn@x + hhd.T@Adyn@Adyn@x + k1*hh + k2*hhd.T@Adyn@x
    # Define Matrices for quadratic programming
    A = matrix(A.astype('float'))   # A
    B = matrix(B.astype('float'))   # B

    H = matrix( np.eye(2).astype('float') )   # Quadratic Cost
    F = matrix( -uNom.astype('float') )       # Linear Cost

    # Solve quadratic programming
    sol = solvers.qp(H, F, A, B)            # Solve
    u = np.squeeze(np.array(sol['x']))      # Store solution

    u = u.reshape((2, 1))                   # Reshape and return
    return u
### END CONTROLLER

### SIMULATION
def simulation(t, x, pGoal, fitterH, fitterHD, obstacles):
    # Compute dx(t) for the ode solver
    
    # Control law
    #w = 1
    #u = np.array([-w**2*np.cos(w*t), w**2*np.sin(w*t)])

    u = controller(x[0:2*dim], pGoal, fitterH, fitterHD, obstacles[0])
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
    l = 15
    lambda1 = 3
    lambda2 = 4

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
    ### SETUP
    solvers.options['show_progress'] = False # Turn off solver verbose

    ### Initial condition
    x0 = np.array([-2, -2, 0, 0])
    pGoal = np.array([3, 3.5])

    goals = []
    goals.append(np.array([0, -1]))
    goals.append(np.array([1, 1]))
    goals.append(np.array([0, -2]))
    goals.append(np.array([0, 0]))
    goals.append(np.array([1, 2]))


    ### Obstacles

    # Define obstacles
    # obs_num = 4
    obstacles = []
    obstacles.append({'Position': np.array([0, 1]), 'Radius': 0.5})
    # obstacles.append({'Position': np.array([2, -2]), 'Radius': 0.5})
    # obstacles.append({'Position': np.array([-2, -2]), 'Radius': 0.5})
    # obstacles.append({'Position': np.array([-2, 2]), 'Radius': 0.5})
    
    # Define circles for plots
    circles = []
    for obs in obstacles:
        circles.append( plt.Circle(obs['Position'], obs['Radius'], color='r') )

    ### Gaussian processes initialiation
    fitterH = GP(dim, 1)            # Function
    fitterHD = GP(2*dim, 1)         # Time Derivative

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
    f = lambda t,x : simulation(t, x, pGoal, fitterH, fitterHD, obstacles)            # Lambda function to incorporate controllre and BF in the ode solver
    #r = ode(f).set_integrator('dopri5', nsteps=10000)       # Set solver
    r = ode(f).set_integrator('dop853')
    r.set_initial_value(r0, 0)                              # Set initial condition

    x = r0.reshape((2*dim+2, 1))                # State Trajectory
    hh = np.array([min(obs_dists)])             # Distance
    dhh = np.array([0])                         # Distance Derivative
    dff = np.array([0])                         # Fitter derivative
    # while r.successful() and r.t < 3:
    #     # Integrate system equations
    #     print(r.t, r.y)
    #     r.integrate(r.t+dt)                                 # Numerically solve ODE for dt seconds
    #     x = np.append(x, (r.y).reshape((2*dim+2,1)), 1)     # Add actual state to state trajectory

    #     # Compute minimum distance
    #     obs_dists = []                              
    #     for obs in obstacles:
    #            obs_dists.append( h(r.y[0:dim], obs) )
    #     index_min = min(range(len(obs_dists)), key=obs_dists.__getitem__)
    #     hh = np.append(hh, min(obs_dists))
    #     dhh = np.append(dhh, hdt(r.y, obstacles[index_min], pGoal, fitterH, fitterHD))
    #     dff = np.append(dff, dfitter(fitterH, r.y, pGoal, fitterH, fitterHD))

    #     # Add sample to GP
    #     # Check variance of current point
    #     if fitterH.posterior_variance(r.y[0:dim]) > 0.1 : 
    #         fitterH.add_sample(r.y[0:2], min(obs_dists))    # Add minimum distance  
    #         print('train1')   
    #         fitterH.train()                                 # Train GP
    #     if fitterHD.posterior_variance(r.y[0:2*dim]) > 0.1:
    #         fitterHD.add_sample(r.y[0:2*dim], r.y[2*dim+1]) # Add Observer Measurement
    #         print('train2')
    #         fitterHD.train()                                # Train GP
    
    t = 0
    x_act = r0
    idx_goal = 0
    while t < 50:
        pGoal = goals[idx_goal]
        pGoal = pGoal.reshape((2,1))

        # Compute minimum distance
        obs_dists = []                              
        for obs in obstacles:
               obs_dists.append( h(x_act[0:dim], obs) )
        index_min = min(range(len(obs_dists)), key=obs_dists.__getitem__)
        hh = np.append(hh, min(obs_dists))
        dhh = np.append(dhh, hdt(x_act, obstacles[index_min], pGoal, fitterH, fitterHD))
        dff = np.append(dff, dfitter(fitterH, x_act, pGoal, fitterH, fitterHD, obstacles[0]))

        # Integrate system equations
        #print(t, x[0:4, -1])
        print(idx_goal, t)
        dx = simulation(t, x_act, pGoal, fitterH, fitterHD, obstacles)  # Numerically solve ODE for dt seconds
        x_act = x_act + dx*dt
        x = np.append(x, (x_act).reshape((2*dim+2,1)), 1)     # Add actual state to state trajectory

        # Add sample to GP
        # Check variance of current point
        if fitterH.posterior_variance(x_act[0:dim]) > 0.02 : 
            fitterH.add_sample(x_act[0:2], min(obs_dists))    # Add minimum distance  
            print('train1')   
            fitterH.train()                                 # Train GP
        if fitterHD.posterior_variance(x_act[0:2*dim]) > 0.02:
            fitterHD.add_sample(x_act[0:2*dim], x_act[2*dim+1]) # Add Observer Measurement
            print('train2')
            fitterHD.train()                                # Train GP
        
        t = t+dt    # Update time

        # Update Goal
        if np.linalg.norm( x_act[0:2] - pGoal.T ) < 10**-2:
            idx_goal = idx_goal+1
            if idx_goal == len(goals):
                break


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
    ax1.contour3D(xx, yy, zz, 25, cmap='viridis')
    ax1.contour3D(xx, yy, zh, 25, cmap='viridis')
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