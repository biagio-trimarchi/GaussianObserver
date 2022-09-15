#! python3

# LIBRARIES
from cmath import tau
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gp import GaussianProcess as GP
from gpConstr import ConstrainedGaussianProcess as cGP
from scipy.integrate import ode
from cvxopt import solvers, matrix
import smoothMaxMin
from bezier import bezier
import os 
import random

# FIGURE_PATH = "/home/biagio/Scrivania/Report_Observers/figs/Simulation_4"
# if not os.path.exists(FIGURE_PATH):
#    os.mkdir(FIGURE_PATH)

class Simulation:
    def __init__(self):
        ### Random Stuff
        random.seed(0)
        self.noise = random.gauss

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
        self.sm_a = 1.0   # Smooth min/max parameter
        self.safe_par = 0.0

        self.Kp = 20     # Position Error Gain
        self.Kv = 4     # Velocity Error Gain

        self.obsLambda1 = 5.0  # Observer pole 1
        self.obsLambda2 = 7.0  # Observer pole 2
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
        self.dist_fitter.set_hyperparams(np.ones((self.dim,))*0.05)
        self.d_dist_fitter = GP(self.rel_degree*self.dim, 1)  # Fit L_f h function
        self.d_dist_fitter.set_hyperparams(np.ones((2*self.dim,))*0.05)

        self.dist_fitter_constr = cGP(2*self.dim)        # Constrained GP for distance estimation
        self.dist_fitter_constr.params.L = (np.diag(np.array([0.05, 0.05, 0.05, 0.05])))    # Set hyperparameters

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

        self.hGPCon = np.empty((0))
        self.hGPCoff = np.empty((0))
        self.LhGPCon = np.empty((0))
        self.LGPCoff = np.empty((0))

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
        self.dist_fun()
        self.dz = np.zeros((self.rel_degree, 1))
        self.dz[0] = self.zAct[1] + self.obsL*self.obsK1*(self.h - self.zAct[0])
        self.dz[1] = self.obsL**2*self.obsK2*(self.h - self.zAct[0])
        return self.dz

    def computeStateDerivative(self):
        self.controller()
        self.dynamics()
        self.observerDynamics()

    def plotResults(self):
        plt.rcParams['text.usetex'] = True

        # Define figure and axis for trajectory
        fig1, ax1 = plt.subplots()

        # Add obstacles to plot
        for circle in self.circles:
            ax1.add_patch(circle) 

        # Plot trajectory
        plt.plot(self.x[0, :], self.x[1, :], label='Trajectory')
        plt.plot(self.desiredTrajectory[0, :], self.desiredTrajectory[1, :], label='Desired Trajectory')
        plt.plot(self.x0[0], self.x0[1], 'o', label='Starting Position')
        
        # Set plot limits
        ax1.set_xlim([-0.5, 5])
        ax1.set_ylim([-4, 4])
        plt.legend()
        plt.title("Agent Trajectory")
        # fig1.savefig(FIGURE_PATH + "/Trajectory.png")

        fig2, ax2 = plt.subplots()
        plt.plot(self.Lfh, label = r'$\mathcal{L}_f h_s$')
        plt.plot(self.z[1, :], label = r'$z_2$')
        plt.plot(self.LfhGP1on, label = r'$\mathcal{L}_f \mu_h$')
        plt.plot(self.LfhGP2on, label = r'$\mu_z$')
        plt.legend()
        plt.title(r'Online Comparison of Estimation of $\mathcal{L}_f h_s$')
        # fig2.savefig(FIGURE_PATH + "/LieON.png")

        fig3, ax3 = plt.subplots()
        plt.plot(self.hStory, label = r'$h_s$')
        plt.plot(self.z[0, :], label = r'$z_1$')
        plt.plot(self.hGPStory, label = r'$\mu_h$')
        plt.plot(self.hGPCon, label = r'Costr')
        plt.legend()
        plt.title(r'Online Comparison of Estimation of $h_s$')
        # fig3.savefig(FIGURE_PATH + "/hON.png")

        fig4, ax4 = plt.subplots()
        plt.plot(self.hStory, label = r'$h_s$')
        plt.plot(self.hGPoff, label = r'$\mu_h$')
        plt.plot(self.hGPCoff, label = r'Constr')
        plt.legend()
        plt.title(r'Offline Value of $\mu_h$ along followed trajectory')
        # fig4.savefig(FIGURE_PATH + "/hOFF.png")

        fig5, ax5 = plt.subplots()
        plt.plot(self.Lfh, label = r'$\mathcal{L}_f h_s$')
        plt.plot(self.LfhGP1off, label = r'$\mathcal{L}_f \mu_h$')
        plt.plot(self.LfhGP2off, label = r'$\mu_z$')
        plt.plot(self.LGPCoff, label = r'Constr')
        plt.legend()
        plt.title(r'Offline Comparison of Estimation of $\mathcal{L}_f h_s$ along followed trajectory')
        # fig5.savefig(FIGURE_PATH + "/LieOFF.png")

        fig6, ax6 = plt.subplots()
        plt.plot(self.LfhGP1on - self.Lfh, label = r'$\mathcal{L}_f \mu_h - \mathcal{L}_f h_s$')
        plt.plot(self.LfhGP2on - self.Lfh, label = r'$\mu_z - \mathcal{L}_f h_s$')
        plt.title('Online Gaussian Processes Errors')
        plt.legend()
        # fig6.savefig(FIGURE_PATH + "/errorsON.png")

        fig7, ax7 = plt.subplots()
        plt.plot(self.LfhGP1off - self.Lfh, label = r'$\mathcal{L}_f \mu_h - \mathcal{L}_f h_s$')
        plt.plot(self.LfhGP2off - self.Lfh, label = r'$\mu_z - \mathcal{L}_f h_s$')
        plt.title('Offline Gaussian Processes Errors')
        plt.legend()
        # fig7.savefig(FIGURE_PATH + "/errorsOFF.png")


        # Show plots
        plt.show()
        plt.close()
        
    def run(self):
        # Simulation time
        self.t = 0.0

        # Initialize Observer
        self.zAct[0] = self.dist_fun()
        self.zAct[1] = 0.0

        # Initialize GP h
        self.dist_fitter.add_sample(self.p, self.dist_fun() + self.noise(0, 0.01))
        self.dist_fitter.train()

        # Initialize GP L_f h
        self.d_dist_fitter.add_sample(self.xAct, self.zAct[1] + self.noise(0, 0.01))
        self.d_dist_fitter.train()

        # Initialize GP Constrained
        self.dist_fitter_constr.addSample(self.xAct, self.dist_fun() + self.noise(0, 0.01))
        self.dist_fitter_constr.addDerivativeSample(self.xAct, self.zAct[1] + self.noise(0, 0.01))
        self.dist_fitter_constr.trainConstraints()
        self.dist_fitter_constr.trainWithoutConstraints()

        
        while self.t < self.T:
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
                self.dist_fitter.add_sample(self.p, self.dist_fun() + self.noise(0, 0.01))
                self.dist_fitter.train()

            # Update second GP
            new = True
            for x in self.d_dist_fitter.params.gp_x.T:
                if np.linalg.norm(self.xAct - x.reshape((4,1))) < 0.2:
                    new = False
                    break
            if new:
                self.d_dist_fitter.add_sample(self.xAct, self.zAct[1] + self.noise(0, 0.01))
                self.d_dist_fitter.train()

            # Update Constrained GP
            new = True
            for x in self.dist_fitter_constr.data_x.T:
                if np.linalg.norm(self.xAct - x.reshape((4,1))) < 0.5:
                    new = False
                    break
            if new:
                self.dist_fitter_constr.addSample(self.xAct, self.dist_fun() + self.noise(0, 0.01))
                self.dist_fitter_constr.addDerivativeSample(self.xAct, self.zAct[1] + self.noise(0, 0.01))

                self.dist_fitter_constr.trainConstraints()
                self.dist_fitter_constr.trainWithoutConstraints()
        
            # new = True
            # for x in self.dist_fitter_constr.data_der_x.T:
            #     if np.linalg.norm(self.xAct - x.reshape((4,1))) < 0.5:
            #         new = False
            #         break
            # if new:
            #     self.dist_fitter_constr.addDerivativeSample(self.xAct, self.zAct[1] + self.noise(0, 0.01))
            #     # print(self.dist_fitter_constr.data_der_x)
            #     self.dist_fitter_constr.trainConstraints()
            
                
            
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

            self.hGPCon = np.append(self.hGPCon, self.dist_fitter_constr.posteriorMeanConstraints(self.xAct)[0])
            # self.hGPCon = np.append(self.hGPCon, self.dist_fitter_constr.posteriorMeanWithoutConstraints(self.xAct))


            
            self.t = self.t + self.dt
        
        self.x = np.delete(self.x, 0, 1)
        self.z = np.delete(self.z, 0, 1)

        for xx in self.x.T:
            self.LfhGP2off = np.append(self.LfhGP2off, self.d_dist_fitter.posterior_mean(xx))
        for tt in range(np.size(self.x, 1)):
            self.hGPoff = np.append(self.hGPoff, self.dist_fitter.posterior_mean(self.x[0:2, tt]))
            self.LfhGP1off = np.append(self.LfhGP1off, self.dist_fitter.posterior_dxmean(self.x[0:2, tt]).T @ self.x[2:, tt])
        
            self.hGPCoff = np.append(self.hGPCoff, self.dist_fitter_constr.posteriorMeanWithoutConstraints(self.x[:, tt]))
            self.LGPCoff = np.append(self.LGPCoff, self.dist_fitter_constr.posteriorMeanConstraints(self.x[:, tt])[1])

        self.plotResults()
        

def main():
    sim = Simulation()
    sim.run()

if __name__ == '__main__':
    main()