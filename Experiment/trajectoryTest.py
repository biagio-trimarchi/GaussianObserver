#! python3

# LIBRARIES
from cmath import tau
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gp import GaussianProcess as GP
from scipy.integrate import ode
from cvxopt import solvers, matrix
import smoothMaxMin
from bezier import bezier
import os 
import random

DATA_PATH = "./Data"
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

PLOTS_PATH = "./Plots"
if not os.path.exists(PLOTS_PATH):
    os.mkdir(PLOTS_PATH)

class Simulation:
    def __init__(self, id):
        ### Counter
        self.id = id

        ### Random Stuff
        random.seed(id)
        self.noise = random.gauss
        self.mu1 = 0
        self.sigma1 = 0.001
        self.mu2 = 0
        self.sigma2 = 0.00
        
        ### State variables
        self.dim = 2                                # Space dimension
        self.rel_degree = 2                         # Relative degree
        self.x0 = np.array([0, 0, 0, 0])            # Initial state
        self.x0 = self.x0.reshape((2*self.dim, 1))  # Reshape 
        self.z0 = np.zeros((self.rel_degree, 1))    # Initialize observer

        self.xAct = self.x0                         # Actual state
        self.zAct = self.z0                         # Actual observer state
        self.p = self.xAct[0:self.dim]              # Actual position
        self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity

        self.x = np.empty((4, 1))                   # State Trajectory
        self.z = np.empty((2, 1))                   # Observer State Trajectory
        self.T = 6                                  # Total Time
        self.dt = 0.01                              # Simulation Time Step

        ### Dynamical Model
        # f(x) = A * x
        self.Adyn = np.block([
                [np.zeros((self.dim, self.dim)), np.eye(self.dim)],
                [np.zeros((self.dim, self.dim)), np.zeros((self.dim, self.dim))]
            ])
        
        # g(x) = B
        self.Bdyn = np.block([
                [np.zeros((self.dim, self.dim))],
                [np.eye(self.dim)]
            ])
        
        ### Obstacles
        self.obstacles = []
        self.obstacles.append({'Position': np.array([1.0, 2.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([3.0, 1.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([2.0, 3.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([3.0, 3.0]).reshape((self.dim,1)), 'Radius': 0.5})
        
        self.obstaclesNum = len(self.obstacles)

        # Define circles for plots
        self.circles = []
        for obs in self.obstacles:
            self.circles.append( plt.Circle(obs['Position'], obs['Radius'], color='r') )

        ### Parameters
        self.sm_a = 5.0   # Smooth min/max parameter
        self.safe_par = 0.0

        self.Kp = 8    # Position Error Gain
        self.Kv = 2     # Velocity Error Gain

        self.obsLambda1 = 3.0  # Observer pole 1
        self.obsLambda2 = 5.0  # Observer pole 2
        self.obsK1 = self.obsLambda1 + self.obsLambda2  # Observer gain 1
        self.obsK2 = self.obsLambda1 * self.obsLambda2  # Observer gain 2
        self.obsL = 20.0    # Observer high gain term

        ### Auxiliary functions
        self.shFun = lambda d: smoothMaxMin.fun(d, -self.sm_a)             # Smooth minimum
        self.dShFun = lambda d: smoothMaxMin.gradientFun(d, -self.sm_a)    # Gradient smooth minimum
        self.ddShFun = lambda d: smoothMaxMin.hessianFun(d, -self.sm_a)    # Hessin smooth minimum
        self.dist_fun = lambda : self.distanceReadingsSmooth()

        # Gaussian Process
        self.dist_fitter = GP(self.dim, 1)    # Fit h function
        self.dist_fitter.set_hyperparams(np.ones((self.dim,))*1)
        self.d_dist_fitter = GP(self.rel_degree*self.dim, 1)  # Fit L_f h function
        self.d_dist_fitter.set_hyperparams(np.array([0.5, 0.5, 0.5, 0.5]))

        # Plot quantities
        self.uNomStory = np.zeros((self.dim, 1))
        self.uStory = np.zeros((self.dim, 1))
        self.hStory = np.empty((0))
        self.hGPStory = np.empty((0))
        self.hGPoff = np.empty((0))

        self.desiredTrajectory = np.zeros((self.dim, 1))

        self.Lfh = np.empty((0))            # Real Lie Derivative
        self.LfhGP1on = np.empty((0))       # Lie Derivative GP online
        self.LfhGP1off = np.empty((0))      # Lie Derivative GP offline
        self.LfhGP2on = np.empty((0))       # Lie Derivative GP 2 online
        self.LfhGP2off = np.empty((0))      # Lie Derivative GP 2 offline
        self.gpSigma = np.empty((0))
    
    def distanceReadings(self):
        # Compute the distance of x from obstacle
        self.d = []
        for obstacle in self.obstacles:
            self.d.append(np.linalg.norm(self.p-obstacle['Position'])**2 - (obstacle['Radius'] + self.safe_par)**2)

        self.d = np.array(self.d)

        return self.d
    
    def gradientDistanceReadings(self):
        self.dd = []
        for obstacle in self.obstacles:
            dh = 2*(self.p-obstacle['Position'].reshape((2,1)))
            self.dd.append(dh)
        
        return self.dd

    def distanceReadingsSmooth(self):
        # Compute the distance of x from obstacle
        self.d = []
        for obstacle in self.obstacles:
            self.d.append(np.linalg.norm(self.p-obstacle['Position'])**2 - (obstacle['Radius'] + self.safe_par)**2)

        self.d = np.array(self.d)
        self.h = self.shFun(self.d)

        return self.h

    def trajectory(self):
        index = 0
        if self.t > self.T/3:
            index = 1
        if self.t > self.T/3*2:
            index = 2
        
        traj = []
        traj.append(np.array([
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
        ]))

        traj.append(np.array([
            [1.0, 2.0, 2.5],
            [1.0, 1.0, 1.5]
        ]))

        traj.append(np.array([
            [2.5, 3.0, 4.0],
            [1.5, 2.5, 2.0]
        ]))

        tau = (self.t - index*(self.T/3))/(self.T/3)
        self.pGoal = bezier(traj[index], tau)
        self.vGoal = bezier(np.diff(traj[index]), tau)

        self.desiredTrajectory = np.append(self.desiredTrajectory, self.pGoal, 1)

    def controller(self):

        self.trajectory()
        self.u = -self.Kp * ( self.p - self.pGoal ) - self.Kv * ( self.v - self.vGoal)

        return self.u
    
    def dynamics(self):
        self.dx = self.Adyn@self.xAct + self.Bdyn@self.u
        
        return self.dx
    
    def observerDynamics(self):
        self.dz = np.zeros((self.rel_degree, 1))
        self.dz[0] = self.zAct[1] + self.obsL*self.obsK1*(self.h_noise - self.zAct[0])
        self.dz[1] = self.obsL**2*self.obsK2*(self.h_noise - self.zAct[0])
        return self.dz

    def computeStateDerivative(self):
        self.controller()
        self.dynamics()
        self.observerDynamics()

    def save_Results(self):
        time_vec = np.linspace(0, self.T+self.dt, int((self.T + self.dt)/self.dt))
        DATA_PATH_SIM = DATA_PATH + "/SIM_" + "{:02d}".format(self.id)
        if not os.path.exists(DATA_PATH_SIM):
            os.mkdir(DATA_PATH_SIM)
        np.save(DATA_PATH_SIM + "/time.npy", time_vec)                                  # Time Vector
        np.save(DATA_PATH_SIM + "/initial_position.npy", self.x0)                       # Initial State
        np.save(DATA_PATH_SIM + "/trajectory.npy", self.x)                              # State Evolution
        np.save(DATA_PATH_SIM + "/desired_trajectory.npy", self.desiredTrajectory)      # Desired State Trajectory
        np.save(DATA_PATH_SIM + "/observer.npy", self.z)                                # Observer State
        np.save(DATA_PATH_SIM + "/distance.npy", self.hStory)                           # h(t)
        np.save(DATA_PATH_SIM + "/Lie_h.npy", self.Lfh)                                 # L_h h(t)
        np.save(DATA_PATH_SIM + "/hGP_on", self.hGPStory)                               # h_GP (t)
        np.save(DATA_PATH_SIM + "/Lie_hGP_on", self.LfhGP1on)                           # L_f h_GP(t) 
        np.save(DATA_PATH_SIM + "/Lie_hGP_off", self.LfhGP1off)                         # L_f h_GP(x) along followed trajectory
        np.save(DATA_PATH_SIM + "/mu_z_on", self.LfhGP2on)                              # mu_z (t) (Second GP)
        np.save(DATA_PATH_SIM + "/mu_z_off", self.LfhGP2off)                            # mu_z (x) (Along followed trajectory)
        
    def run(self):
        # Simulation time
        self.t = 0.0

        # Initialize Observer
        self.zAct[0] = self.dist_fun()
        self.zAct[1] = 0.0

        # Initialize GP h
        self.dist_fitter.add_sample(self.p, self.dist_fun())
        self.dist_fitter.train()

        # Initialize GP L_f h
        self.d_dist_fitter.add_sample(self.xAct, self.zAct[1])
        self.d_dist_fitter.train()
        
        while self.t < self.T:
            self.h_noise = self.dist_fun() + self.noise(self.mu1, self.sigma1)
            self.computeStateDerivative()
            self.xAct = self.xAct + self.dx*self.dt     # Update actual state
            self.zAct = self.zAct + self.dz*self.dt     # Update observer state
            self.p = self.xAct[0:self.dim]              # Actual position
            self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity
            
            # Update first GP
            new = True
            for x in self.dist_fitter.params.gp_x.T:
                if np.linalg.norm(self.p - x.reshape((2,1))) < 0.1:
                    new = False
                    break
            if new:
                self.dist_fitter.add_sample(self.p, self.h_noise)
                self.dist_fitter.train()

            # Update second GP
            new = True
            for x in self.d_dist_fitter.params.gp_x.T:
                if np.linalg.norm(self.xAct - x.reshape((4,1))) < 0.05:
                    new = False
                    break
            if new:
                self.d_dist_fitter.add_sample(self.xAct, self.zAct[1])
                self.d_dist_fitter.train()
            
            # Auxiliary Quantities
            self.distanceReadings()             # Distance readings
            self.gradientDistanceReadings()     # Gradients distance readings

            self.sh = self.shFun(self.d)        # Smooth min
            self.dSh = self.dShFun(self.d)      # Gradient smooth min
            self.ddSh = self.ddShFun(self.d)    # Hessian smooth min

            self.gradSh = np.zeros((self.dim, 1))
            for i in range(self.obstaclesNum):
                self.gradSh = self.gradSh + self.dd[i]*self.dSh[i]
        
            self.gradSh = np.block([ 
                [self.gradSh],
                [np.zeros((self.dim, 1))]
            ])

            # Compute Lie Derivatives
            self.x = np.append(self.x, (self.xAct).reshape((2*self.dim,1)), 1)          # Add actual state to state trajectory
            self.z = np.append(self.z, (self.zAct).reshape((self.rel_degree,1)), 1)     # Add actual state to state trajectory
            lfh = self.gradSh[0:2].T @ self.v                                           # Real Lie Derivative
            lfh_GP1 = self.dist_fitter.posterior_dxmean(self.p).T @ self.v              # Lie Derivative GP online
            lfh_GP2 = self.d_dist_fitter.posterior_mean(self.xAct)

            # Plot Quantities
            self.Lfh = np.append(self.Lfh, lfh)
            self.LfhGP1on = np.append(self.LfhGP1on, lfh_GP1)
            self.LfhGP2on = np.append(self.LfhGP2on, lfh_GP2)
            self.hStory = np.append(self.hStory, self.dist_fun())
            self.hGPStory = np.append(self.hGPStory, self.dist_fitter.posterior_mean(self.p))

            
            self.t = self.t + self.dt
        
        self.x = np.delete(self.x, 0, 1)
        self.z = np.delete(self.z, 0, 1)

        for xx in self.x.T:
            self.LfhGP2off = np.append(self.LfhGP2off, self.d_dist_fitter.posterior_mean(xx))
        for tt in range(np.size(self.x, 1)):
            self.hGPoff = np.append(self.hGPoff, self.dist_fitter.posterior_mean(self.x[0:2, tt]))
            self.LfhGP1off = np.append(self.LfhGP1off, self.dist_fitter.posterior_dxmean(self.x[0:2, tt]).T @ self.x[2:, tt])
        
        
        # self.plotResults()
        self.save_Results()
        

def main():
    n_sim = 50
    aim = 1
    if aim == 0:
        # Simulate 
        for i in range(1, n_sim+1):
            sim = Simulation(i)
            print("Simulation " + str(i) + " running...")
            sim.run()
    else:
        # Analyze Data

        # Allocate vectors
        Lie_h_mean = np.zeros(np.load('./Data/SIM_01/Lie_h.npy').shape)        # Mean Lie derivative of h
        Lie_h_variance = np.zeros(np.load('./Data/SIM_01/Lie_h.npy').shape)    # Variance Lie derivative of h

        observer_mean = np.zeros(np.load('./Data/SIM_01/observer.npy').shape)        # Mean value of the observer 
        observer_variance = np.zeros(np.load('./Data/SIM_01/observer.npy').shape)    # Variance of the observer

        Lie_hGP_on_mean = np.zeros(np.load('./Data/SIM_01/Lie_hGP_on.npy').shape)      # Mean of Lie of h_GP online
        Lie_hGP_on_variance = np.zeros(np.load('./Data/SIM_01/Lie_hGP_on.npy').shape)  # Variance of Lie of h_GP online

        Lie_hGP_off_mean = np.zeros(np.load('./Data/SIM_01/Lie_hGP_off.npy').shape)      # Mean of Lie of h_GP offline
        Lie_hGP_off_variance = np.zeros(np.load('./Data/SIM_01/Lie_hGP_off.npy').shape)  # Variance of Lie of h_GP offline

        mu_z_on_mean = np.zeros(np.load('./Data/SIM_01/mu_z_on.npy').shape)        # Mean value of mu_z online
        mu_z_on_variance = np.zeros(np.load('./Data/SIM_01/mu_z_on.npy').shape)    # Variance of of mu_z online

        mu_z_off_mean = np.zeros(np.load('./Data/SIM_01/mu_z_off.npy').shape)        # Mean value of mu_z offline
        mu_z_off_variance = np.zeros(np.load('./Data/SIM_01/mu_z_off.npy').shape)    # Variance of of mu_z offline

        # Compute mean
        # Sum data for mean
        for i in range(1, n_sim+1):
            DATA_PATH_SIM = DATA_PATH + "/SIM_" + "{:02d}".format(i)

            Lie_h_mean = Lie_h_mean + np.load(DATA_PATH_SIM + '/Lie_h.npy')
            observer_mean = observer_mean + np.load(DATA_PATH_SIM + '/observer.npy')
            Lie_hGP_on_mean = Lie_hGP_on_mean + np.load(DATA_PATH_SIM + '/Lie_hGP_on.npy')
            Lie_hGP_off_mean = Lie_hGP_off_mean + np.load(DATA_PATH_SIM + '/Lie_hGP_off.npy')
            mu_z_on_mean = mu_z_on_mean + np.load(DATA_PATH_SIM + '/mu_z_on.npy')
            mu_z_off_mean = mu_z_off_mean + np.load(DATA_PATH_SIM + '/mu_z_off.npy')

        # Divide by total number of simulation
        Lie_h_mean = Lie_h_mean / n_sim
        observer_mean = observer_mean / n_sim
        Lie_hGP_on_mean = Lie_hGP_on_mean / n_sim
        Lie_hGP_off_mean = Lie_hGP_off_mean / n_sim
        mu_z_on_mean = mu_z_on_mean / n_sim
        mu_z_off_mean = mu_z_off_mean / n_sim

         # Compute variance
        for i in range(1, n_sim+1):
            DATA_PATH_SIM = DATA_PATH + "/SIM_" + "{:02d}".format(i)

            Lie_h_variance = Lie_h_variance + np.square((np.load(DATA_PATH_SIM + '/Lie_h.npy') - Lie_h_mean))
            observer_variance = observer_variance + np.square(np.load(DATA_PATH_SIM + '/observer.npy') - observer_mean)
            Lie_hGP_on_variance = Lie_hGP_on_variance + np.square(np.load(DATA_PATH_SIM + '/Lie_hGP_on.npy') - Lie_hGP_on_mean)
            Lie_hGP_off_variance = Lie_hGP_off_variance + np.square(np.load(DATA_PATH_SIM + '/Lie_hGP_off.npy') - Lie_hGP_off_mean)
            mu_z_on_variance = mu_z_on_variance + np.square(np.load(DATA_PATH_SIM + '/mu_z_on.npy') - mu_z_on_mean)
            mu_z_off_variance = mu_z_off_variance + np.square(np.load(DATA_PATH_SIM + '/mu_z_off.npy') - mu_z_off_mean)

        # Divide by total number of simulations
        Lie_h_variance = Lie_h_variance / n_sim
        observer_variance = observer_variance / n_sim
        Lie_hGP_on_variance = Lie_hGP_on_variance / n_sim
        Lie_hGP_off_variance = Lie_hGP_off_variance / n_sim
        mu_z_on_variance = mu_z_on_variance / n_sim
        mu_z_off_variance = mu_z_off_variance / n_sim

        # Plot Data
        time_vec = np.load(DATA_PATH_SIM + '/time.npy')

        fig1, ax1 = plt.subplots()
        ax1.set_xlabel("t (s)")
        plt.plot(time_vec, Lie_h_mean, label = r'$\mathcal{L}_f h_s$', linewidth=2)
        plt.plot(time_vec, observer_mean[1, :], label = r'$z_2$', linewidth=2)
        plt.plot(time_vec, Lie_hGP_on_mean, label = r'$\mathcal{L}_f \mu_h$', linewidth=2)
        plt.plot(time_vec, mu_z_on_mean, label = r'$\mu_z$', linewidth=2)
        plt.legend()
        plt.title(r'Online Comparison of Estimation of $\mathcal{L}_f h_s$')
        fig1.savefig(PLOTS_PATH + "/LieON.png", format='png')

        fig5, ax5 = plt.subplots()
        ax5.set_xlabel("t (s)")
        plt.plot(time_vec, Lie_h_mean, label = r'$\mathcal{L}_f h_s$', linewidth=2)
        plt.plot(time_vec, Lie_hGP_off_mean, label = r'$\mathcal{L}_f \mu_h$', linewidth=2)
        plt.plot(time_vec, mu_z_off_mean, label = r'$\mu_z$', linewidth=2)
        plt.legend()
        plt.title(r'Offline Comparison of Estimation of $\mathcal{L}_f h_s$ along followed trajectory')
        fig5.savefig(PLOTS_PATH + "/LieOFF.png", format='png')

        fig6, ax6 = plt.subplots()
        ax6.set_xlabel("t (s)")
        plt.plot(time_vec, Lie_hGP_on_mean - Lie_h_mean, label = r'$\mathcal{L}_f \mu_h - \mathcal{L}_f h_s$', linewidth=2)
        plt.plot(time_vec, mu_z_on_mean - Lie_h_mean, label = r'$\mu_z - \mathcal{L}_f h_s$', linewidth=2)
        plt.title('Online Gaussian Processes Errors')
        plt.legend()
        fig6.savefig(PLOTS_PATH + "/errorsON.png", format='png')

        fig7, ax7 = plt.subplots()
        ax7.set_xlabel("t (s)")
        plt.plot(time_vec, Lie_hGP_off_mean - Lie_h_mean, label = r'$\mathcal{L}_f \mu_h - \mathcal{L}_f h_s$', linewidth=2)
        plt.plot(time_vec, mu_z_off_mean - Lie_h_mean, label = r'$\mu_z - \mathcal{L}_f h_s$', linewidth=2)
        plt.title('Offline Gaussian Processes Errors')
        plt.legend()
        fig7.savefig(PLOTS_PATH + "/errorsOFF.png", format='png')

        plt.show()
        plt.close()


if __name__ == '__main__':
    main()