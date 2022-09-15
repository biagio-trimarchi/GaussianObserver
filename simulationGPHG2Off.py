#! python3

# LIBRARIES
from cmath import tau
import os
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gp import GaussianProcess as GP
from scipy.integrate import ode
from cvxopt import solvers, matrix
import smoothMaxMin

class Simulation:
    def __init__(self):
        
        ### State variables
        self.dim = 2                                # Space dimension
        self.rel_degree = 2                         # Relative degree
        self.x0 = np.array([-1, -1, 0, 0])          # Initial state
        self.x0 = self.x0.reshape((2*self.dim, 1))  # Reshape 
        self.z0 = np.zeros((self.rel_degree, 1))    # Initialize observer

        self.xAct = self.x0                         # Actual state
        self.zAct = self.z0                         # Actual observer state
        self.p = self.xAct[0:self.dim]              # Actual position
        self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity

        self.x = self.x0                            # State Trajectory
        self.z = self.z0
        self.T = 10                               # Total Time
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
        self.obstacles.append({'Position': np.array([0.0, 1.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([-0.5, -2.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([1.0, 1.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([-2.0, 0.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([0.5, 0.5]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([-2.0, 3.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstaclesNum = len(self.obstacles)

        # Define circles for plots
        self.circles = []
        for obs in self.obstacles:
            self.circles.append( plt.Circle(obs['Position'], obs['Radius'], color='r') )

        ### Parameters
        self.sm_a = 15.0   # Smooth min/max parameter
        self.safe_par = 0.1 

        self.alpha1 = 25.0     # "Class K" multiplier for psi_0
        self.alpha2 = 15.0     # "Class K" multiplier for psi_1
        self.k1 = self.alpha1 * self.alpha2   # "Class K" multiplier for h(x)
        self.k2 = self.alpha1 + self.alpha2   # "Class K" multiplier for L_f h(x)
        
        self.Kp = 2     # Position Error Gain
        self.Kv = 3     # Velocity Error Gain

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
        # self.dist_fun = lambda : self.distanceProduct()
        
        # Gaussian Process
        self.dist_fitter = GP(self.dim, 1)    # Fit h function
        self.dist_fitter.set_hyperparams(np.ones((self.dim,))*1)
        self.d_dist_fitter = GP(self.rel_degree*self.dim, 1)  # Fit L_f h function
        self.d_dist_fitter.set_hyperparams(np.ones((2*self.dim,))*1)
        # Usefult quantities
        self.psi0 = np.empty((0))
        self.psi1 = np.empty((0))
        self.uNomStory = np.zeros((self.dim, 1))
        self.uStory = np.zeros((self.dim, 1))

        self.Aopt1Story = np.empty((0))
        self.Aopt2Story = np.empty((0))
        self.BoptStory = np.empty((0))

        self.Aopt1StoryNom = np.empty((0))
        self.Aopt2StoryNom = np.empty((0))
        self.BoptStoryNom = np.empty((0))

        self.Lfh = self.z0[1]
        self.LfhGP = np.empty((0))

        self.gpSigma = np.empty((0))

        ### Settings
        solvers.options['show_progress'] = False # Turn off solver verbose
        np.random.seed(0)

    def distanceReadingsSmooth(self):
        # Compute the distance of x from obstacle
        self.d = []
        for obstacle in self.obstacles:
            self.d.append(np.linalg.norm(self.p-obstacle['Position'])**2 - (obstacle['Radius'] + self.safe_par)**2)

        self.d = np.array(self.d)
        self.h = self.shFun(self.d)

        return self.h
    
    def distanceReadingsSmoothPoint(self, p):
        # Compute the distance of x from obstacle
        self.dP = []
        for obstacle in self.obstacles:
            self.dP.append(np.linalg.norm(p-obstacle['Position'])**2 - (obstacle['Radius'] + self.safe_par)**2)

        self.dP = np.array(self.dP)
        self.hP = self.shFun(self.dP)

        return self.hP
    
    def distanceProduct(self):
        # Compute the distance of x from obstacle
        self.h = 1.0
        for obstacle in self.obstacles:
            self.h = 0.01*self.h * (np.linalg.norm(self.p-obstacle['Position'])**2 - (obstacle['Radius'])**2)
        return self.h
    
    def distanceReadings(self):
        # Compute the distance of x from obstacle
        self.d = []
        for obstacle in self.obstacles:
            self.d.append(np.linalg.norm(self.p-obstacle['Position'])**2 - (obstacle['Radius'])**2)

        self.d = np.array(self.d)

        return self.d
    
    def gradientDistanceReadings(self):
        self.dd = []
        for obstacle in self.obstacles:
            dh = 2*(self.p-obstacle['Position'].reshape((2,1)))
            self.dd.append(dh)
        
        return self.dd
    
    def gradientDistanceReadingsPoint(self, p):
        self.ddP = []
        for obstacle in self.obstacles:
            dh = 2*(p-obstacle['Position'].reshape((2,1)))
            self.ddP.append(dh)
        
        return self.ddP
    
    def hessianDistanceReadings(self):
        self.ddd = []
        for obstacle in self.obstacles:
            ddh = 2*np.diag(self.p.squeeze())
            self.ddd.append(ddh)
        
        return self.ddd


    def dynamics(self):
        self.dx = self.Adyn@self.xAct + self.Bdyn@self.u
        
        return self.dx
    
    def controller(self):
        # self.uNom = self.sat(-self.Kp * ( self.p - self.pGoal ) - self.Kv * self.v)
        self.uNom = -self.Kp * ( self.p - self.pGoal ) - self.Kv * (self.v - self.vGoal)

        self.dist_fun()
        # hGP = self.dist_fitter.posterior_mean(self.p) - self.safe_par
        # hGP = self.zAct[0]
        hGP = self.h
        LfHGP = self.d_dist_fitter.posterior_mean(self.xAct)
        gradLfHGP = self.d_dist_fitter.posterior_dxmean(self.xAct)

        self.LfhGP = np.append(self.LfhGP, LfHGP)

        # Quantities for plots
        self.psi0 = np.append(self.psi0, hGP)
        self.psi1 = np.append(self.psi1, LfHGP)

        # Setup QP 
        Aopt = - gradLfHGP.T @ self.Bdyn
        Bopt = gradLfHGP.T @ self.Adyn @ self.xAct + \
                self.alpha2 * LfHGP
        
        self.Aopt1Story = np.append(self.Aopt1Story, np.squeeze(Aopt)[0])
        self.Aopt2Story = np.append(self.Aopt2Story, np.squeeze(Aopt)[1])
        self.BoptStory = np.append(self.BoptStory, Bopt)
        
        
        Aopt = matrix(Aopt.astype('float'))       # A u <= B
        Bopt = matrix(Bopt.astype('float'))       # A u <= B

        H = matrix( np.eye(2).astype('float') )   # Quadratic Cost
        F = matrix( -self.uNom.astype('float') )  # Linear Cost
        
        # Solve QP
        sol = solvers.qp(H, F, Aopt, Bopt)        # Solve
        self.u = np.squeeze(np.array(sol['x']))   # Store solution

        # self.u = self.sat(self.u.reshape((self.dim, 1)))   # Reshape and return
        self.u = self.u.reshape((self.dim, 1))   # Reshape and return
        
        return self.u

    def sat(self, u_):
        u = u_
        max_a = 2.0
        if u[0] > max_a:
            u[0] = max_a

        if u[1] > max_a:
            u[1] = max_a

        if u[0] < -max_a:
            u[0] = -max_a

        if u[1] < -max_a:
            u[1] = -max_a
        return u
    
    def fullKnownController(self):
        
        # Compute derivative
        self.distanceReadings()             # Distance readings
        self.gradientDistanceReadings()     # Gradients distance readings
        self.hessianDistanceReadings()      # Hessian distance readings

        self.sh = self.shFun(self.d) - self.safe_par # Smooth min
        self.dSh = self.dShFun(self.d)      # Gradient smooth min
        self.ddSh = self.ddShFun(self.d)    # Hessian smooth min

        self.gradSh = np.zeros((self.dim, 1))
        for i in range(self.obstaclesNum):
            self.gradSh = self.gradSh + self.dd[i]*self.dSh[i]
        
        self.gradSh = np.block([ 
            [self.gradSh],
            [np.zeros((self.dim, 1))]
        ])

        hessSh = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                aux1 = 0
                for k in range(self.obstaclesNum):
                    aux1 = aux1 + (self.dSh[k]*self.ddd[k])[i][j]

                aux2 = 0
                for k in range(self.obstaclesNum):
                    for l in range(self.obstaclesNum):
                        aux2 = aux2 + self.ddSh[k][l]*(self.dd[k])[i]*(self.dd[l])[j]
                hessSh[i][j] = aux1 + aux2
        
        hessSh = np.block([
            [hessSh, np.zeros((self.dim, self.dim))],
            [np.zeros((self.dim, self.dim)), np.zeros((self.dim, self.dim))]
        ])

        # Setup QP 
        Aopt = - ( self.xAct.T @ self.Adyn.T @ hessSh @ self.Bdyn \
                    + self.gradSh.T @ self.Adyn @ self.Bdyn )
        Bopt = self.xAct.T @ self.Adyn.T @ hessSh @ self.Adyn @ self.xAct \
                + self.gradSh.T @ self.Adyn @ self.Adyn @ self.xAct \
                + self.k1 * self.sh \
                + self.k2 * self.gradSh.T @ self.Adyn @ self.xAct
        
        self.Aopt1StoryNom = np.append(self.Aopt1StoryNom, np.squeeze(Aopt)[0])
        self.Aopt2StoryNom = np.append(self.Aopt2StoryNom, np.squeeze(Aopt)[1])
        self.BoptStoryNom = np.append(self.BoptStoryNom, Bopt)

    def observerDynamics(self):
        self.dist_fun()
        self.dz = np.zeros((self.rel_degree, 1))
        self.dz[0] = self.zAct[1] + self.obsL*self.obsK1*(self.h - self.zAct[0])
        self.dz[1] = self.obsL**2*self.obsK2*(self.h - self.zAct[0])
        return self.dz

    def plotResults(self):
        # Define figure and axis for trajectory
        fig1, ax1 = plt.subplots()

        # Add obstacles to plot
        for circle in self.circles:
            ax1.add_patch(circle) 

        # Plot trajectory
        plt.plot(self.x[0, :], self.x[1, :], label='Trajectory')
        plt.plot(self.x0[0], self.x0[1], 'o', label='Starting Position')
        
        # Set plot limits
        ax1.set_xlim([-4, 4])
        ax1.set_ylim([-4, 4])
        plt.legend()

        # Plot psi
        fig2, ax2 = plt.subplots()
        plt.plot(self.psi0, label='Psi 0')
        plt.plot(self.psi1, label='Psi 1')
        plt.legend()

        # Plot psi
        fig3, ax3 = plt.subplots()
        plt.plot(self.uNomStory[0, :], label='u nom x')
        plt.plot(self.uStory[0, :], label='u x')
        plt.legend()

        fig4, ax4 = plt.subplots()
        plt.plot(self.uNomStory[1, :], label='u nom y')
        plt.plot(self.uStory[1, :], label='u y')
        plt.legend()

        fig5, ax5 = plt.subplots()
        plt.plot(self.z[0, :], label='z1')
        plt.plot(self.z[1, :], label='z2')
        plt.plot(self.Lfh, label='Lfh')
        # plt.plot(self.LfhGP, label='LfhGP')
        plt.legend()

        # fig6, ax6 = plt.subplots()
        # plt.plot(self.Aopt1Story, label='Aopt1')
        # plt.plot(self.Aopt1StoryNom, label='Aopt1 nom')
        # plt.legend()

        # fig7, ax7 = plt.subplots()
        # plt.plot(self.Aopt2Story, label='Aopt2')
        # plt.plot(self.Aopt2StoryNom, label='Aopt2 nom')
        # plt.legend()

        # fig8, ax8 = plt.subplots()
        # plt.plot(self.BoptStory, label='Bopt')
        # plt.plot(self.BoptStoryNom, label='Bopt nom')
        # plt.legend()

        fig8, ax8 = plt.subplots()
        plt.plot(self.x[2], label='Vel_x')
        plt.plot(self.x[3], label='Vel_y')
        plt.legend()

        fig8, ax8 = plt.subplots()
        plt.plot(self.gpSigma, label='Sigma')
        plt.legend()


        # Show plots
        plt.show()
        plt.close()

    def computeStateDerivative(self):
        self.controller()
        self.fullKnownController()
        self.dynamics()
        self.observerDynamics()

        return self.dx

    def findNextGoal(self):
        found = False
        while not found:
            self.pGoal = self.p + np.random.rand(self.dim, 1) * 0.5 - 0.25
            self.vGoal = self.v + np.random.rand(self.dim, 1) * 0.5 - 0.25
            self.xTry = np.block([[self.pGoal],
                                  [self.vGoal]])
            # print(self.xTry)
            # print(self.d_dist_fitter.posterior_mean(self.xTry))
            if self.d_dist_fitter.posterior_mean(self.xTry) > 0:# and self.d_dist_fitter.posterior_variance(self.xTry) < 0.2:# and self.hGG > 0:# 
                found = True
        # print(self.pGoal)
        return self.pGoal

    def run(self):
        t = 0.0
        # Initialize Observer
        self.zAct[0] = self.dist_fun()
        self.zAct[1] = 0.0
        # Initialize GP h
        self.dist_fitter.add_sample(self.p, self.dist_fun())
        self.dist_fitter.train()

        dom_x = np.arange(-4, 4.1, 0.5)
        dom_y = np.arange(-4, 4.1, 0.5)
        dom_vx = np.arange(-2, 2.1, 0.4)
        dom_vy = np.arange(-2, 2.1, 0.4)

        for idx_x in range(dom_x.shape[0]):
            for idx_y in range(dom_y.shape[0]):
                for idx_vx in range(dom_vx.shape[0]):
                    for idx_vy in range(dom_vy.shape[0]):
                        self.p = np.array([dom_x[idx_x], dom_y[idx_y]]).reshape((2,1))
                        self.v = np.array([dom_vx[idx_vx], dom_vy[idx_vy]]).reshape((2,1))

                        # Compute derivative
                        self.distanceReadings()             # Distance readings
                        self.gradientDistanceReadings()     # Gradients distance readings

                        self.sh = self.shFun(self.d) - self.safe_par # Smooth min
                        self.dSh = self.dShFun(self.d)      # Gradient smooth min

                        self.gradSh = np.zeros((self.dim, 1))
                        for i in range(self.obstaclesNum):
                            self.gradSh = self.gradSh + self.dd[i]*self.dSh[i]
                        
                        lfh = self.gradSh.T@self.v
                        self.d_dist_fitter.add_sample(np.array([dom_x[idx_x], dom_y[idx_y], dom_vx[idx_vx], dom_vy[idx_vy] ]), lfh + self.alpha1*self.h)
        self.d_dist_fitter.train()

        self.tExp = 0.0
        self.findNextGoal()
        taux = 0.0

        self.p = self.xAct[0:self.dim]              # Actual position
        self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity
        while t < self.T:
            # print(t)
            if  np.linalg.norm(self.p) > 4.5:
                break
            self.computeStateDerivative()

            self.xAct = self.xAct + self.dx*self.dt
            self.p = self.xAct[0:self.dim]              # Actual position
            self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity

            self.zAct = self.zAct + self.dz*self.dt
            
            self.gpSigma = np.append(self.gpSigma, self.d_dist_fitter.posterior_variance(self.xAct))
                
            lfh = self.gradSh[0:2].T@self.v
            self.x = np.append(self.x, (self.xAct).reshape((2*self.dim,1)), 1)
            self.z = np.append(self.z, (self.zAct).reshape((self.rel_degree,1)), 1)     # Add actual state to state trajectory
            self.Lfh = np.append(self.Lfh, lfh)
            
            t = t + self.dt
            self.tExp = self.tExp + self.dt
            taux = taux + self.dt
            
            if self.tExp > 2.0 or (np.linalg.norm(self.v - self.vGoal) < 0.01 and np.linalg.norm(self.p - self.pGoal) < 0.01 ):
                self.findNextGoal()
                print(self.xTry)
                self.tExp = 0.0

            # if self.tExp > 10 and np.linalg.norm(self.v) < 0.1:
            #     print("Error 3")
            #     break

            # Store quantities for plots
            self.uNomStory = np.append(self.uNomStory, self.uNom, 1)
            self.uStory = np.append(self.uStory, self.u, 1)

            # if self.psi0[-1] + 0.2 < 0:
            #     print("ERROR 1")
            #     break
            
            # if self.psi1[-1] + 0.2 < 0:
            #     print("ERROR 2")
            #     break

            
        self.plotResults()

def main():
    sim = Simulation()
    sim.run()


if __name__ == '__main__':
    main()