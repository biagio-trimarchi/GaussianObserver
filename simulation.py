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
        self.x0 = np.array([-3, -3, 0, 0])          # Initial state
        self.x0 = self.x0.reshape((2*self.dim, 1))

        self.xAct = self.x0                         # Actual state
        self.p = self.xAct[0:self.dim]              # Actual position
        self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity

        self.x = self.x0                            # State Trajectory
        self.T = 50                                  # Total Time
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
        self.safe_par = 0.2

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

        # Usefult quantities
        self.psi0 = np.empty((0))
        self.psi1 = np.empty((0))

        ### Settings
        solvers.options['show_progress'] = False # Turn off solver verbose
        np.random.seed(10)
    
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
        self.uNom = -self.Kp * ( self.p - self.pGoal ) - self.Kv * self.v
        
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

        # Quantities for plots
        self.psi0 = np.append(self.psi0, self.sh)
        #print(self.psi0)
        self.psi1 = np.append(self.psi1, gradSh.T @ self.Adyn @ self.xAct + self.alpha1*self.sh)

        # Setup QP 
        Aopt = - ( self.xAct.T @ self.Adyn.T @ hessSh @ self.Bdyn \
                    + gradSh.T @ self.Adyn @ self.Bdyn )
        Bopt = self.xAct.T @ self.Adyn.T @ hessSh @ self.Adyn @ self.xAct \
                + gradSh.T @ self.Adyn @ self.Adyn @ self.xAct \
                + self.k1 * self.sh \
                + self.k2 * gradSh.T @ self.Adyn @ self.xAct
        
        
        
        Aopt = matrix(Aopt.astype('float'))       # A u <= B
        Bopt = matrix(Bopt.astype('float'))       # A u <= B

        H = matrix( np.eye(2).astype('float') )   # Quadratic Cost
        F = matrix( -self.uNom.astype('float') )  # Linear Cost
        
        # Solve QP
        sol = solvers.qp(H, F, Aopt, Bopt)        # Solve
        self.u = np.squeeze(np.array(sol['x']))   # Store solution

        self.u = self.u.reshape((self.dim, 1))    # Reshape and return
        return self.u

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



        # Show plots
        plt.show()
        plt.close()

    def computeStateDerivative(self):
        self.controller()
        self.dynamics()

        return self.dx

    def findNextGoal(self):
        self.pGoal = np.random.rand(self.dim, 1) * 8.0 - 4.0
        print(self.pGoal)
        return self.pGoal

    def run(self):
        t = 0.0

        self.tExp = 0.0
        self.findNextGoal()
        while t < self.T:
            self.computeStateDerivative()
            self.xAct = self.xAct + self.dx*self.dt
            self.p = self.xAct[0:self.dim]              # Actual position
            self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity
            self.x = np.append(self.x, (self.xAct).reshape((2*self.dim,1)), 1)     # Add actual state to state trajectory
            t = t + self.dt
            self.tExp = self.tExp + self.dt

            if self.tExp > 0.2 and np.linalg.norm(self.v) < 0.1:
                self.findNextGoal()
                self.tExp = 0.0
     
        self.plotResults()

def main():
    sim = Simulation()
    sim.run()


if __name__ == '__main__':
    main()