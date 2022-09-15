#! python3

# LIBRARIES
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
        ### Random Stuff
        random.seed(0)
        self.noise = random.gauss
        
        ### State variables
        self.dim = 2                                # Space dimenstion
        self.x0 = np.array([-2, -2, 0, 0])          # Initial state
        self.x0 = self.x0.reshape((2*self.dim, 1))

        self.xAct = self.x0                         # Actual state
        self.p = self.xAct[0:self.dim]              # Actual position
        self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity

        self.x = self.x0                            # State Trajectory
        self.T = 4.2                                # Total Time
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
        self.sm_a = 0.5       # Smooth min/max parameter 

        self.alpha1 = 2.0     # "Class K" multiplier for psi_0
        self.alpha2 = 1.2     # "Class K" multiplier for psi_1
        self.k1 = self.alpha1 * self.alpha2   # "Class K" multiplier for h(x)
        self.k2 = self.alpha1 + self.alpha2   # "Class K" multiplier for L_f h(x)
        
        self.Kp = 1     # Position Error Gain
        self.Kv = 2     # Velocity Error Gain

        ### Auxiliary functions
        self.shFun = lambda d: smoothMaxMin.fun(d, -self.sm_a)             # Smooth minimum
        self.dShFun = lambda d: smoothMaxMin.gradientFun(d, -self.sm_a)    # Gradient smooth minimum
        self.ddShFun = lambda d: smoothMaxMin.hessianFun(d, -self.sm_a)    # Hessin smooth minimum

        # Usefult quantities
        self.psi0 = np.array([0])
        self.psi1 = np.array([0])

        ### Settings
        solvers.options['show_progress'] = False # Turn off solver verbose

    
    def distanceProduct(self):
        # Compute the distance of x from obstacle
        self.d = 1
        for obstacle in self.obstacles:
            self.d = self.d * (np.linalg.norm(self.p-obstacle['Position'])**2 - (obstacle['Radius'])**2)
        return self.d
    
    def gradientDistanceProduct(self):
        self.dd = 0.0
        for i in range(self.obstaclesNum):
            aux = 1.0
            for j in range(self.obstaclesNum):
                if i != j:
                    aux = aux*(np.linalg.norm(self.p-self.obstacles[j]['Position'])**2 - (self.obstacles[j]['Radius'])**2)

            self.dd = self.dd + (self.p-self.obstacles[i]['Position'])*aux
            self.dd = 2.0*self.dd
        return self.dd
    
    def hessianDistanceProduct(self):
        self.ddd = 0.0
        aux_cum_sum_1 = 0.0
        for i in range(self.obstaclesNum):
            
            aux_cum_prod_1 = 1.0
            
            for j in range(self.obstaclesNum):
                if i != j:
                    aux_cum_prod_1 = aux_cum_prod_1 * (np.linalg.norm(self.p-self.obstacles[j]['Position'])**2 - (self.obstacles[j]['Radius'])**2)
            aux_cum_prod_1 = aux_cum_prod_1 * np.diag(np.squeeze(self.p))
            aux_cum_sum_1 = aux_cum_sum_1 + aux_cum_prod_1
        aux_cum_sum_1 = aux_cum_sum_1*2

        aux_cum_sum_2 = 0.0
        for i in range(self.obstaclesNum):
            aux_cum_sum_3 = 0.0
            for j in range(self.obstaclesNum):
                aux_cum_prod_2 = 1.0
                if i != j:
                    for k in range(self.obstaclesNum):
                        if k != j and k != i:
                            aux_cum_prod_2 = aux_cum_prod_2 * (np.linalg.norm(self.p-self.obstacles[k]['Position'])**2 - (self.obstacles[k]['Radius'])**2)
                    aux_cum_prod_2 = aux_cum_prod_2*(self.p-self.obstacles[j]['Position'])
                    aux_cum_sum_3 = aux_cum_sum_3 + aux_cum_prod_2
            aux_cum_sum_2 = aux_cum_sum_2 + aux_cum_sum_3*(self.p-self.obstacles[i]['Position']).T
        aux_cum_sum_2 = 4*aux_cum_sum_2

        print(aux_cum_sum_1)
        print(aux_cum_sum_2)
        
        self.ddd = aux_cum_sum_1 + aux_cum_sum_2
        return self.ddd

    def dynamics(self):
        self.dx = self.Adyn@self.xAct + self.Bdyn@self.u
        
        return self.dx
    
    def controller(self):
        self.uNom = -self.Kp * ( self.p - self.pGoal ) - self.Kv * self.v
        
        # Compute derivative
        self.distanceProduct()             # Distance readings
        self.gradientDistanceProduct()     # Gradients distance readings
        self.hessianDistanceProduct()      # Hessian distance readings
        
        gradD = np.block([ 
            [self.dd],
            [np.zeros((self.dim, 1))]
        ])
        
        hessD = np.block([
            [self.ddd, np.zeros((self.dim, self.dim))],
            [np.zeros((self.dim, self.dim)), np.zeros((self.dim, self.dim))]
        ])

        # Quantities for plots
        self.psi0 = np.append(self.psi0, self.d)
        self.psi1 = np.append(self.psi1, gradD.T @ self.Adyn @ self.xAct + self.alpha1*self.d)

        # Setup QP 
        Aopt = - ( self.xAct.T @ self.Adyn.T @ hessD @ self.Bdyn \
                    + gradD.T @ self.Adyn @ self.Bdyn )
        Bopt = self.xAct.T @ self.Adyn.T @ hessD @ self.Adyn @ self.xAct \
                + gradD.T @ self.Adyn @ self.Adyn @ self.xAct \
                + self.k1 * self.d \
                + self.k2 * gradD.T @ self.Adyn @ self.xAct
        
        
        
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


    def run(self):
        t = 0.0
        aux = 0
        while t < self.T:
            self.computeStateDerivative()
            self.xAct = self.xAct + self.dx*self.dt
            self.p = self.xAct[0:self.dim]              # Actual position
            self.v = self.xAct[self.dim:2*self.dim]     # Actual velocity
            self.x = np.append(self.x, (self.xAct).reshape((2*self.dim,1)), 1)     # Add actual state to state trajectory
            t = t + self.dt
            aux = aux+1

            # if aux > 200:
            #     self.plotResults()
            #     aux = 0        
        self.plotResults()

def main():
    sim = Simulation()
    sim.run()


if __name__ == '__main__':
    main()