#! python3

# LIBRARIES
import os
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
        self.dim = 2                                # Space dimenstion
        self.x0 = np.array([-1, -1, 0, 0])          # Initial state
        self.x0 = self.x0.reshape((2*self.dim, 1))

        self.xAct = self.x0                         # Actual state
        self.p = self.xAct[0:self.dim]              # Actual position
        self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity

        self.x = self.x0                            # State Trajectory
        self.T = 20                                 # Total Time
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

        ### Goals
        self.goals = []
        self.goals.append(np.array([0, -1]).reshape((self.dim,1)))
        self.goals.append(np.array([1, 1]).reshape((self.dim,1)))
        self.goals.append(np.array([0, -2]).reshape((self.dim,1)))
        self.goals.append(np.array([0, 0]).reshape((self.dim,1)))
        self.goals.append(np.array([1, 2]).reshape((self.dim,1)))
        self.pGoal = np.array([3, 3.5]).reshape((self.dim,1))     # Actual goal

        ### Obstacles
        self.obstacles = []
        self.obstacles.append({'Position': np.array([0.0, 1.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([-0.5, -2.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([1.0, 1.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([-2.0, 0.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([0.0, 0.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstacles.append({'Position': np.array([-2.0, 3.0]).reshape((self.dim,1)), 'Radius': 0.5})
        self.obstaclesNum = len(self.obstacles)

        # Define circles for plots
        self.circles = []
        for obs in self.obstacles:
            self.circles.append( plt.Circle(obs['Position'], obs['Radius'], color='r') )

        ### Parameters
        self.sm_a = 8.0       # Smooth min/max parameter 
        self.safe_par = 0.1

        self.alpha1 = 25.0     # "Class K" multiplier for psi_0
        self.alpha2 = 15.0     # "Class K" multiplier for psi_1
        self.k1 = self.alpha1 * self.alpha2   # "Class K" multiplier for h(x)
        self.k2 = self.alpha1 + self.alpha2   # "Class K" multiplier for L_f h(x)
        
        self.Kp = 2     # Position Error Gain
        self.Kv = 3     # Velocity Error Gain

        ### Auxiliary functions
        self.shFun = lambda d: smoothMaxMin.fun(d, -self.sm_a)             # Smooth minimum
        self.dShFun = lambda d: smoothMaxMin.gradientFun(d, -self.sm_a)    # Gradient smooth minimum
        self.ddShFun = lambda d: smoothMaxMin.hessianFun(d, -self.sm_a)    # Hessin smooth minimum
        self.dist_fun = lambda : self.distanceReadingsSmooth()
        # self.dist_fun = lambda : self.distanceProduct()
        
        # Gaussian Process
        self.dist_fitter = GP(self.dim, 1)    # Fit h function
        self.dist_fitter.set_hyperparams(np.ones((self.dim,))*0.2)

        # Usefult quantities
        self.psi0 = np.empty((0))
        self.psi1 = np.empty((0))
        self.uNomStory = np.zeros((self.dim, 1))
        self.uStory = np.zeros((self.dim, 1))
        self.hStory = np.empty((0))

        self.Aopt1Story = np.empty((0))
        self.Aopt2Story = np.empty((0))
        self.BoptStory = np.empty((0))

        self.Aopt1StoryNom = np.empty((0))
        self.Aopt2StoryNom = np.empty((0))
        self.BoptStoryNom = np.empty((0))

        self.gpSigma = np.empty((0))

        ### Settings
        solvers.options['show_progress'] = False # Turn off solver verbose
        np.random.seed(5)
    
    def distanceReadingsSmooth(self):
        # Compute the distance of x from obstacle
        self.d = []
        for obstacle in self.obstacles:
            self.d.append(np.linalg.norm(self.p-obstacle['Position'])**2 - (obstacle['Radius'] + self.safe_par)**2)

        self.d = np.array(self.d)
        self.h = self.shFun(self.d)

        return self.h
    
    def distanceReadingsSmoothFromP(self, p):
        # Compute the distance of x from obstacle
        self.dd = []
        for obstacle in self.obstacles:
            self.dd.append(np.linalg.norm(p-obstacle['Position'])**2 - (obstacle['Radius'] + self.safe_par)**2)

        self.dd = np.array(self.dd)
        self.hh = self.shFun(self.dd)

        return self.hh
    
    def distanceProduct(self):
        # Compute the distance of x from obstacle
        self.h = 1
        for obstacle in self.obstacles:
            self.h = self.h * 0.01 * (np.linalg.norm(self.p-obstacle['Position'])**2 - (obstacle['Radius'])**2)
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
    
    def hessianDistanceReadings(self):
        self.ddd = []
        for obstacle in self.obstacles:
            ddh = 2*np.diag(self.p.squeeze())
            self.ddd.append(ddh)
        
        return self.ddd

    def findNextGoal(self):
        found = False
        while not found:
            self.pGoal = np.random.rand(self.dim, 1) * 8.0 - 4.0
            if self.dist_fitter.posterior_mean(self.pGoal) > 0.0: # and np.linalg.norm(self.pGoal - self.p) < 1.0:
                found = True
        print(self.pGoal)
        return self.pGoal

    def dynamics(self):
        self.dx = self.Adyn@self.xAct + self.Bdyn@self.u
        
        return self.dx
    
    def controller(self):
        self.uNom = -self.Kp * ( self.p - self.pGoal ) - self.Kv * self.v

        self.dist_fun()
        mu = 0.05
        self.hGP = self.dist_fitter.posterior_mean(self.p) - mu*self.dist_fitter.posterior_variance(self.p)
        gradGP = self.dist_fitter.posterior_dxmean(self.p) - mu*self.dist_fitter.posterior_dxvariance(self.p)
        hessGP = self.dist_fitter.posterior_ddxmean(self.p) - mu*self.dist_fitter.posterior_ddxvariance(self.p)

        gradGP = np.block([ 
                            [gradGP], 
                            [np.zeros((self.dim, 1))]])
        
        hessGP = np.block([ 
            [hessGP, np.zeros((self.dim, self.dim))], 
            [np.zeros((self.dim, self.dim)), np.zeros((self.dim, self.dim))]
         ])

        # Quantities for plots
        self.psi0 = np.append(self.psi0, self.hGP)
        self.psi1 = np.append(self.psi1, gradGP.T @ self.Adyn @ self.xAct + self.alpha1*self.hGP)
        self.hStory = np.append(self.hStory, self.h)

        # Setup QP 
        Aopt = - ( self.xAct.T @ self.Adyn.T @ hessGP @ self.Bdyn \
                    + gradGP.T @ self.Adyn @ self.Bdyn )
        Bopt = self.xAct.T @ self.Adyn.T @ hessGP @ self.Adyn @ self.xAct \
                + gradGP.T @ self.Adyn @ self.Adyn @ self.xAct \
                + self.k1 * self.hGP \
                + self.k2 * gradGP.T @ self.Adyn @ self.xAct
        
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

        self.u = self.u.reshape((self.dim, 1))    # Reshape and return
        return self.u
    
    def fullKnownController(self):
        
        # Compute derivative
        self.distanceReadings()             # Distance readings
        self.gradientDistanceReadings()     # Gradients distance readings
        self.hessianDistanceReadings()      # Hessian distance readings

        self.sh = self.shFun(self.d)        # Smooth min
        self.dSh = self.dShFun(self.d)      # Gradient smooth min
        self.ddSh = self.ddShFun(self.d)    # Hessian smooth min

        gradSh = np.zeros((self.dim, 1))
        for i in range(self.obstaclesNum):
            gradSh = gradSh + self.dd[i]*self.dSh[i]
        
        gradSh = np.block([ 
            [gradSh],
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
                    + gradSh.T @ self.Adyn @ self.Bdyn )
        Bopt = self.xAct.T @ self.Adyn.T @ hessSh @ self.Adyn @ self.xAct \
                + gradSh.T @ self.Adyn @ self.Adyn @ self.xAct \
                + self.k1 * self.sh \
                + self.k2 * gradSh.T @ self.Adyn @ self.xAct
        
        self.Aopt1StoryNom = np.append(self.Aopt1StoryNom, np.squeeze(Aopt)[0])
        self.Aopt2StoryNom = np.append(self.Aopt2StoryNom, np.squeeze(Aopt)[1])
        self.BoptStoryNom = np.append(self.BoptStoryNom, Bopt)
        
        

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
        plt.plot(self.Aopt1Story, label='Aopt1')
        plt.plot(self.Aopt1StoryNom, label='Aopt1 nom')
        plt.legend()

        # fig5, ax5 = plt.subplots()
        # plt.plot(self.Aopt2Story, label='Aopt2')
        # plt.plot(self.Aopt2StoryNom, label='Aopt2 nom')
        # plt.legend()

        # fig5, ax5 = plt.subplots()
        # plt.plot(self.BoptStory, label='Bopt')
        # plt.plot(self.BoptStoryNom, label='Bopt nom')
        # plt.legend()

        fig8, ax8 = plt.subplots()
        plt.plot(self.gpSigma, label='Sigma')
        plt.legend()

        # Plot psi
        fig9, ax9 = plt.subplots()
        plt.plot(self.psi0, label='Psi 0')
        plt.plot(self.hStory, label='h')
        plt.legend()


        # Show plots
        plt.show()
        plt.close()

    def plotSafetySet(self):
        dom_x = np.arange(-4, 4, 0.2)
        dom_y = np.arange(-4, 4, 0.2)
        xx, yy = np.meshgrid(dom_x, dom_y)
        zz = np.empty(xx.shape)
        zv = np.empty(xx.shape)
        zh = np.empty(xx.shape)
        for idx_x in range(dom_x.shape[0]):
            for idx_y in range(dom_y.shape[0]):
                zz[idx_x, idx_y] = self.dist_fitter.posterior_mean(np.array([dom_x[idx_x], dom_y[idx_y]]))
                zh[idx_x, idx_y] = self.distanceReadingsSmoothFromP(np.array([dom_x[idx_x], dom_y[idx_y]]).reshape((2, 1)))
                zv[idx_x, idx_y] = zh[idx_x, idx_y] - zz[idx_x, idx_y]
        
        fig1 = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.contour3D(xx, yy, zz, 25, cmap='viridis')
        ax1.contour3D(xx, yy, zh, 25, cmap='coolwarm')
        plt.grid()

        fig2 = plt.figure()
        ax2 = plt.axes(projection='3d')
        ax2.contour3D(xx, yy, zv, 25, cmap='viridis')
        plt.grid()

        plt.show()

    def computeStateDerivative(self):
        self.controller()
        self.fullKnownController()
        self.dynamics()

        return self.dx


    def run(self):
        t = 0.0

        # Initialize GP
        scale = 10.0
        self.dist_fitter.add_sample(self.p, self.dist_fun()/scale)
        self.dist_fitter.train()

        self.tExp = 0.0
        self.findNextGoal()
        taux = 0.0
        while t < self.T:
            self.gpSigma = np.append(self.gpSigma, self.dist_fitter.posterior_variance(self.p))
            self.computeStateDerivative()
            self.xAct = self.xAct + self.dx*self.dt
            self.p = self.xAct[0:self.dim]              # Actual position
            self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity
            self.x = np.append(self.x, (self.xAct).reshape((2*self.dim,1)), 1)     # Add actual state to state trajectory
            t = t + self.dt
            self.tExp = self.tExp + self.dt

            new = True
            for x in self.dist_fitter.params.gp_x.T:
                if np.linalg.norm(self.p - x.reshape((2,1))) < 0.08:
                    new = False
                    break
            if new:
                self.dist_fitter.add_sample(self.p, self.dist_fun()/scale)
                self.dist_fitter.train()

                # self.plotSafetySet()
            taux = taux + self.dt
            
            if self.tExp > 2.0 and np.linalg.norm(self.v) < 0.1:
                self.findNextGoal()
                self.tExp = 0.0

            # Store quantities for plots
            self.uNomStory = np.append(self.uNomStory, self.uNom, 1)
            self.uStory = np.append(self.uStory, self.u, 1)
        self.plotResults()
        self.plotSafetySet()

def main():
    sim = Simulation()
    sim.run()


if __name__ == '__main__':
    main()