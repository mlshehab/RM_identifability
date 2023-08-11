"""
Run linear programming inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import normalize
# print(sys.path)
# from irl.linear_irl import irl

# import irl.linear_irl as linear_irl
# import irl.mdp.gridworld as gridworld
# from irl.mdp.gridworld import Gridworld

import random

from cvxopt import matrix, solvers

import cvxpy as cp 

# import numpy as np
import numpy.random as rn

class Gridworld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, wind, discount):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.n_states - 1:
            return 1
        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1

    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                                    random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)))]

                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)



def binary_labeling_matrix(n_states, n_RM_states):
     

    out = np.zeros((n_RM_states+1, n_states))
    for state in range(n_states):
        if state == 0:
            out[0][state] = 1
        elif state == 4:
            out[1][state] = 1
        elif state == 24:
            out[2][state] = 1
        elif state == 20:
            out[3][state] = 1
        else:
            out[4][state] = 1
    return out

# UNIT TEST FOR :::binary_labeling_matrix:::
# print(binary_labeling_matrix(25,4))

def labeling_function(s_prime, n_states= 25, n_RM_states =4):
    out = binary_labeling_matrix(n_states,n_RM_states)
    return out.T[s_prime]

# UNIT TEST FOR :::labeling_function:::
# print(labeling_function(4))




def delta_u(u, s_prime):

    l = labeling_function(s_prime)

    if u == 0 and l[0] == 0:
        return 0 # i.e u_0
    elif u == 0 and l[0] == 1:
        return 1 # i.e u_1 
    elif u == 1 and l[1] == 0:
        return 1 # i.e. u_1
    elif u == 1 and l[1] == 1:
        return 2 # i.e. u_2
    elif u == 2 and l[2] == 0 :
        return 2 # i.e. u_2
    elif u == 2 and l[2] == 1:
        return 3 # i.e. u_3
    elif u == 3 and l[3] == 0:
        return 3 # i.e. u_3
    elif u == 3 and l[3] == 1:
        return 0 # i.e. u_0

# UNIT TEST FOR :::delta_u:::
# print(delta_u(0,1))


def get_su(state , n_states = 25):
    if state <= 24:
        return (state%n_states, 0 )
    elif state >= 25 and state <= 49:
        return (state%n_states, 1 )
    elif state >= 50 and state <= 74:
        return (state%n_states, 2 )
    else:
        return (state%n_states, 3 )
    

# UNIT TEST FOR :::get_u:::
# print("YOY:", get_u(28))


def product_transition_matrix(n_mdp_states = 25, n_rm_states = 4,n_actions = 4, transition_probability = None):
    
    new_state_dim = n_mdp_states*n_rm_states

    P = np.zeros((n_actions, new_state_dim, new_state_dim ))

    transition_probability = np.transpose(transition_probability, (1, 0, 2))

    for i in range(n_actions):
        for starting_state in range(new_state_dim):
            for end_state in range(new_state_dim):
                mdp_state_start, rm_state_start = get_su(starting_state)
                mdp_state_end, rm_state_end = get_su(end_state)

                if rm_state_end == delta_u(rm_state_start, mdp_state_end):
                    P[i][starting_state][end_state] = transition_probability[i][mdp_state_start][mdp_state_end]
    
    assert abs(P[0].sum(axis = 1).sum() - new_state_dim) <= 1e-4
    assert abs(P[1].sum(axis = 1).sum() - new_state_dim) <= 1e-4
    assert abs(P[2].sum(axis = 1).sum() - new_state_dim) <= 1e-4
    assert abs(P[3].sum(axis = 1).sum() - new_state_dim) <= 1e-4

    # print(P[1].sum(axis = 1))


# def optimal_policy_rm(product_state, gw, n_mdp_states = 25, n_rm_states = 4,n_actions = 4):
#     s,u = product_state

#     sx, sy = gw.int_to_point(s)
    
#     if u == 0:
#         if 




def main(grid_size, discount):
    """
    Run linear programming inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """

    wind = 0.3
    trajectory_length = 3*grid_size

    # gw = gridworld.Gridworld(grid_size, wind, discount)
    gw = Gridworld(grid_size, wind, discount)

    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)]

    


    # print(policy)
    # print(matrix_policy(policy, gw.n_states, gw.n_actions))
    # r = linear_irl.irl(gw.n_states, gw.n_actions, gw.transition_probability,
    #         policy, gw.discount, 1, 5)
    # r = irl(gw.n_states, gw.n_actions, gw.transition_probability,
    #         policy, gw.discount, 1, 5)

    

    # r = IRL_state_action_state(gw.n_states, gw.n_actions, gw.transition_probability,
    #         policy, gw.discount, 1, 5)

    # plt.subplot(1, 2, 1)
    # plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("Groundtruth reward")
    # plt.subplot(1, 2, 2)
    # plt.pcolor(r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("Recovered reward")
    # plt.show()
    # plt.savefig("mygraph.png")
    # print("learned r: ", r/np.linalg.norm(r) )
    # print("trueeee r: ", ground_r)
    print(gw.int_to_point(5))


def matrix_policy(policy, n_states, n_actions):
    
    out_policy = np.zeros(shape = (n_actions, n_states))

    for i in range(n_states):
        out_policy[policy[i]][i] = 1

    return out_policy




def IRL_state_action_state(n_states, n_actions, transition_probability, policy, discount, Rmax,
        l1):
    # optimal policy here should be transformed into a matrix of appropiate dimesion
    opt_policy = matrix_policy(policy, n_states, n_actions)
    # print("OPT:", opt_policy.T)
    # The transition policy convention is different here to the rest of the code
    # for legacy reasons; here, we reorder axes to fix this. We expect the
    # new probabilities to be of the shape (A, N, N).
    transition_probability = np.transpose(transition_probability, (1, 0, 2))

    # create the optimal transition matrix based on the optimal policy
    opt_transition_matrix = np.zeros(shape = (n_states, n_states))
    for i in range(n_states):
        opt_transition_matrix[i] = transition_probability[policy[i]][i]


    # the main optimization variable R is a (S x A x S) dimensional matrix
    R_bar_bar_bar = {}
    for i in range(n_states):
        R_bar_bar_bar[i] = cp.Variable((n_actions, n_states))

    # creating the auxiliary state-action reward variable
    # R_bar_bar = np.ones((n_states,n_actions))
    R_bar_bar = cp.Variable((n_states,n_actions))
    

    # print(R_bar_bar.shape)
    # print(R_bar_bar)
    # R_bar_bar = cp.Variable((n_states,n_actions))

    # print(R_bar_bar)
    R = cp.diag(R_bar_bar@opt_policy)
    # print(R)
    # v_star = cp.diag(R_bar_bar@opt_policy)

    A = set(range(n_actions))  # Set of actions to help manage reordering
                               # actions.

    constraints = []
    inv_matrix = np.linalg.inv(np.eye(n_states) -
                        discount*opt_transition_matrix)

    for s in range(n_states):
        for a in A - {policy[s]}:
            constraints += [R_bar_bar[s, policy[s]] - R_bar_bar[s,a] + \
            discount* (transition_probability[policy[s]][s] - transition_probability[a][s])@inv_matrix@R  >= 0 ]

    for s in range(n_states):
        for a in range(n_actions):
            v = 0
            for s_to in range(n_states):
                v += R_bar_bar_bar[s][a][s_to]*transition_probability[a,s,s_to]
            # print(v)
            constraints += [R_bar_bar[s][a] == v]

    def T(a, s):
        """
        Shorthand for a dot product used a lot in the LP formulation.
        """

        return np.dot(transition_probability[policy[s], s] -
                      transition_probability[a, s],
                      np.linalg.inv(np.eye(n_states) -
                        discount*transition_probability[policy[s]]))
    

    zero_stack1 = np.zeros((n_states*(n_actions-1), n_states))

    T_stack = np.vstack([
        -T(a, s)
        for s in range(n_states)
        for a in A - {policy[s]}
    ])
    I_stack1 = np.vstack([
        np.eye(1, n_states, s)
        for s in range(n_states)
        for a in A - {policy[s]}
    ])

    D = np.hstack([T_stack,I_stack1,zero_stack1])
    print(D.shape)

    u = cp.Variable(n_states)
    M = cp.Variable(n_states)
    

    constraints += [R <= u, -R<= u , D@cp.hstack([R,M,u]) <= 0, R <= 2, R>= -2]

    # constraints += [R <= u, R<= -u, D@cp.hstack([R,M,u]) <= 0]

    obj = cp.Maximize(cp.sum(M- l1*u) )

    prob = cp.Problem(obj, constraints)
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    x_normed = normalize(R_bar_bar.value, axis=1, norm='l1')
    print("R: ",np.round(x_normed,2))
    # print("M:", M.value)
    # print(cp.diag(R_bar_bar@opt_policy))
    actions = ['$a_0$','$a_1$','$a_2$','$a_3$']
    states = ['$s_{%.d}$'%(24-d) for d in range(n_states)]
    # print(states)
    fig, (ax1, ax2) = plt.subplots(figsize=(18, 7), dpi = 500,ncols=2)
    pos = ax1.imshow(x_normed,origin = 'upper', extent = [0,4,0,24], aspect = 1.0/5)
    ax1.set_title("Normalized Reward")
    fig.colorbar(pos,ax = ax1)

    ax1.set_xticks(np.arange(len(actions)))
    ax1.set_xticklabels(labels= actions)
    ax1.set_yticks(np.arange(len(states)))
    ax1.set_yticklabels(labels= states)

    pos2 = ax2.imshow(opt_policy.T,origin = 'upper', extent = [0,4,0,24], aspect = 1.0/5)
    ax2.set_title("Optimal Policy")
    fig.colorbar(pos2,ax = ax2)
    # fig.tight_layout()

    ax2.set_xticks(np.arange(len(actions)))
    ax2.set_xticklabels(labels= actions)
    ax2.set_yticks(np.arange(len(states)))
    ax2.set_yticklabels(labels= states)
    fig.suptitle("Comparison of the Recovered State-Action Rewards with the Optimal Policy", fontsize = 20)
    # fig.colorbar()
    plt.savefig("state_action_state_def_reward.png")
    # plt.show()

if __name__ == '__main__':
    # R is 3D array with dimensions S x A x S.
   
    main(5, 0.2)
