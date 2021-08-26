# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Finite state controller policy
"""
import numpy as np
import networkx as nx

# =============================================================================
# def SBPR(Z, sigma, rho, size_A, size_O, c = 0.1, d = 10**-6):
#     '''
#     Parameters
#     ----------
#     Z : int
#         truncation level of SB process (number of nodes).
#     sigma : float array
#         SB parameter.
#     rho : float vector (length = size of action set)
#         parameter of Dirichlet distribution.
#     size_A : int
#         size of action set.
#     size_O : int
#         size of observation set.
#     c : float, optional
#         parameter of Gamma. The default is 0.1.
#     d : float, optional
#         parameter of Gamma. The default is 10**-6.
#     Returns
#     -------
#     None.
#     '''
#     # if rho vector != size of action set, alarm
#     assert rho.shape[0] == size_A
#     # create action distribution at each node
#     action_prob = np.random.default_rng().dirichlet(rho, size = Z)
#     
#     # create node transition probabilities for each (action, observation) set
#     assert sigma.shape == (Z, Z)
#     W = []
#     for o in range(size_O):
#         W_temp = []
#         for a in range(size_A):
#             # generate eta for Beta distribution
#             eta = np.random.default_rng().gamma(c, d, size = (Z ,Z))
#             V = np.random.default_rng().beta(sigma, eta)  # Beta r.v. array
#             node_prob = np.empty_like(V)
#             node_prob[:, 0] = V[:, 0]
#             node_prob[:, 1:] = V[:, 1:] * (1 - V[:, :-1]).cumprod(axis = 1)
#             W_temp.append(node_prob)
#             
#         W.append(W_temp)
#     
#     # create FSC policy
#     policy = FSC_policy(action_prob, W)
#     
#     return policy
# =============================================================================


class uniform_policy(object):
    '''
    For collecting node info for behavior policy
    '''
    def __init__(self, act_set):
        self.act_setsize = len(act_set)
        self.action_prob = np.array([1/self.act_setsize] * self.act_setsize)
        
    def select_action(self, node: int):
        # the input argument is not functional, just to make class structure similar
        # to FSC policy
        act = np.random.default_rng().multinomial(1, self.action_prob, size = 1)[0]
        
        return np.where(act == 1)[0][0]
    
    def next_node(self, cur_node: int, act: int, obs: int):
        # the input argument is not functional, just to make class structure similar
        # to FSC policy
        return 0


# =============================================================================
# class behavior_policy(object):
#     def __init__(self, A, O, Z, epsilon):
#         self.A = A
#         self.O = O
#         self.Z = Z
#         self.epsilon = epsilon
#         
#         self.prob_table()
#         
#     
#     def prob_table(self):
#         # initial node distribution
#         self.eta = np.zeros(self.Z)
#         self.eta[0] = 1
#         
#         # initialize p(z_t|a_0, ..., a_{t-1}, o_0, ..., o_t)
#         self.pz = np.array(self.eta)
#         
#         # uniform node transition prob
#         self.node_prob = np.ones((self.A, self.O, self.Z, self.Z))
#         self.node_prob /= np.sum(self.node_prob, axis = 3)[...,np.newaxis]
#         
#         # uniform exploration action probabilities
#         self.action_prob = np.random.default_rng().dirichlet(np.ones(self.A), self.Z)
#     
#     
#     def select_action(self, node: int):
#         '''
#         Parameters
#         ----------
#         node : int
#             current node.
#         Returns
#         -------
#         output the index of action to take given node.
#         '''
#         # extract probability vector for given node
#         prob_set = self.action_prob[node, :]
#         # pick an action
#         act = np.random.default_rng().multinomial(1, prob_set, size = 1)[0]
#         
#         return np.where(act == 1)[0][0]
#         
#     
#     def next_node(self, cur_node: int, act: int, obs: int):
#         '''
#         Parameters
#         ----------
#         act : int
#             index of action taken.
#         obs : int
#             index of observation received after act is taken.
#         cur_node : int
#             index of current node.
#         Returns
#         -------
#         output node distribution for next step given action, obseration, and 
#         current node.
#         '''
#         # extract probability vector for given node, action, observation
#         prob_set = self.node_prob[act, obs, cur_node, :]
#         # pick a noode
#         node = np.random.default_rng().multinomial(1, prob_set, size = 1)[0]
#         
#         return np.where(node == 1)[0][0]
# =============================================================================
# =============================================================================
# class behavior_policy(object):
#     def __init__(self, A, O, Z, epsilon, phi = None, sigma = None, lambda_ = None):
#         self.A = A
#         self.O = O
#         self.Z = Z
#         self.epsilon = epsilon
#         # pi array
#         self.phi = phi
#         self.sigma = sigma
#         self.lambda_ = lambda_
#         
#         self.prob_table()
#         
#     
#     def prob_table(self):
#         # initial node distribution
#         self.eta = np.zeros(self.Z)
#         self.eta[0] = 1
#         
#         # uniform exploration action probabilities
#         explore_act = np.ones((self.Z, self.A))
#         explore_act /= np.sum(explore_act, axis = 1)[:, np.newaxis]
#         
#         if self.phi is None:
#             # initial exploitation action probabilities
#             exploit_act = np.array(explore_act)
#             # initial node transition prob
#             self.node_prob = np.ones((self.A, self.O, self.Z, self.Z))
#             self.node_prob /= np.sum(self.node_prob, axis = 3)[...,np.newaxis]
#             
#         else:
#             assert self.phi.shape == explore_act.shape
#             exploit_act = self.phi / np.sum(self.phi, axis = -1)[..., np.newaxis]
#             
#             v = self.sigma / (self.sigma + self.lambda_)
#             vv = (1 - v[..., :-1]).cumprod(axis = -1)
#             self.node_prob = np.empty_like(self.sigma)
#             self.node_prob[..., 0] = v[..., 0]
#             self.node_prob[..., 1:-1] = v[..., 1:-1] * vv[..., :-1]
#             self.node_prob[..., -1] = vv[..., -1]
#             self.node_prob /= np.sum(self.node_prob, axis = -1)[..., np.newaxis]
#             
#         self.action_prob = self.epsilon * explore_act + (1 - self.epsilon) * exploit_act
#             
#     
#     def refresh_prob(self):
#         # initialize p(z_t|a_0, ..., a_{t-1}, o_0, ..., o_t)
#         self.pz = np.array(self.eta)
#         self.t = 0  # indicator for initial state
#         
#     
#     def select_action(self, act_pre: int = -1, obv_pre: int = -1):
#         '''
#         Parameters
#         ----------
#         act_pre : int, optional
#             previous action index. The default is -1.
#         obv_pre : int, optional
#             previous observation index. The default is -1.
#         Returns
#         -------
#         TYPE
#             DESCRIPTION.
#         '''
#         if self.t > 0:
#             self.update_action(act_pre, obv_pre)
#         
#         self.t += 1
#         
#         # update action probability marginalizing nodes
#         # p(z_t) * p(a|z)
#         prob_set = self.pz[..., np.newaxis] * self.action_prob
#         # marginalize z
#         prob_set = np.sum(prob_set, axis = 0)
#         # normalize to valid probability
#         prob_set = prob_set / np.sum(prob_set)
#         
#         # pick an action
#         act = np.random.default_rng().multinomial(1, prob_set, size = 1)[0]
#         
#         return np.where(act == 1)[0][0]
#     
#     
#     def update_action(self, act, obv):
#         assert act >= 0 and obv >= 0
#         # p(z_t|z_{t-1}, a_{t-1}, o_t)
#         p_z_zao = self.node_prob[act, obv]
#         
#         # p(z_{t-1}) * p(a_{t-1}|z)
#         p_az = self.pz * self.action_prob[:, act]
#         # p(z_t, z_{t-1}, a_{t-1}|o_t)
#         p_azz_o = p_az[..., np.newaxis] * p_z_zao
#         # marginalize z_{t-1}, get p(z_t, a_{t-1}|o_t)
#         p_az_o = np.sum(p_azz_o, axis = 0)
#         # normalize, get p(z_t|a_{t-1}, o_t)
#         self.pz = p_az_o / np.sum(p_az_o)
# =============================================================================

    
###############################################################################
class behavior_policy2:
    def __init__(self, O, Z, epsilon, policy_parameters):
        self.A = 7
        self.O = O + 1  # size of observation set
        self.Z = Z
        self.epsilon = epsilon  # exploration factor
        
        # policy parameters = (delta, mu, phi, sigma, lambda)
        self.delta = policy_parameters[0]   # for generating p(z_0)
        self.mu = policy_parameters[1]      # for generating p(z_0)
        self.phi = policy_parameters[2]     # for generating p(a|z)
        self.sigma = policy_parameters[3]   # for generating p(z|z, a, o)
        self.lambda_ = policy_parameters[4] # for generating p(z|z, a, o)
        
        # build probability tables using above parameters
        self.gen_probability_tables()
        
    
    def gen_probability_tables(self):
        
        # if parameter is None, it is the very 1st iteration (so only exploration)
        if self.delta is None:
            # uniform initial node distribution
            self.eta = np.ones(self.Z) / self.Z
            
            # uniform action probabilities for exploration
            self.explore_act = np.ones((self.Z, self.A)) / self.A
            # action probabilities for exploitation for 1st iteration
            self.exploit_act = np.array(self.explore_act)
            
            # node transition probabilities for 1st iteration
            self.node_prob = np.ones((self.A, self.O, self.Z, self.Z)) / self.Z
            
        # have learned parameters from previous learning iteration, use them to
        # generate probability tables
        else:
            # compute initial node distribution
            u = self.delta / (self.delta + self.mu)
            uu = (1 - u[: -1]).cumprod(axis = -1)
            self.eta = np.empty_like(self.delta)
            self.eta[0] = u[0]
            self.eta[1: -1] = u[..., 1:-1] * uu[..., :-1]
            self.eta[-1] = uu[-1]
            self.eta /= np.sum(self.eta)
            
            # uniform action probabilities for exploration
            self.explore_act = np.ones((len(self.eta), self.A)) / self.A
            # compute action distribution given nodes
            assert self.phi.shape == self.explore_act.shape
            self.exploit_act = self.phi / np.sum(self.phi, axis = -1)[..., np.newaxis]
            
            # compute node distribution given previous node, action, observation
            v = self.sigma / (self.sigma + self.lambda_)
            vv = (1 - v[..., :-1]).cumprod(axis = -1)
            self.node_prob = np.empty_like(self.sigma)
            self.node_prob[..., 0] = v[..., 0]
            self.node_prob[..., 1:-1] = v[..., 1:-1] * vv[..., :-1]
            self.node_prob[..., -1] = vv[..., -1]
            self.node_prob /= np.sum(self.node_prob, axis = -1)[..., np.newaxis]
            
        # for variational inference use
        self.action_prob = self.epsilon * self.explore_act + (1 - self.epsilon) * self.exploit_act
            
    
    def reset_policy(self):
        # initialize p(z_t|a_{0:t-1}, o_{1:t})
        self.p_z = np.array(self.eta)
        self.not_1st_state = False  # flag for initial state
        
    
    def select_action(self, act_pre: int = -1, obv_pre: int = -1):
        
        # update p(z) if it is not initial state
        if self.not_1st_state:
            self.update_action(act_pre, obv_pre, self.u)
        # is initial state, set flag for future update
        else:
            self.not_1st_state = True
        
        # exploration or exploitation
        self.u = np.random.default_rng().uniform(0, 1)
        
        
        if self.u > self.epsilon:
            # update action probability by marginalizing nodes
            # joint p(a, z) = p(z_t) * p(a|z)
            p_az = self.p_z[..., np.newaxis] * self.exploit_act
            # get p(a) by marginalize z
            p_a = np.sum(p_az, axis = 0)
            # normalize to valid probability
            p_a = p_a / np.sum(p_a)
        
        else:
            p_a = self.explore_act[0]
        
        # sample an action
        act = np.random.default_rng().multinomial(1, p_a, size = 1)[0]
        
        return np.where(act == 1)[0][0]
    
    
    def update_action(self, act, obv, u):
        assert act >= 0 and obv >= 0
        # p(z_t|z_{t-1}, a_{t-1}, o_t)
        p_z_zao = self.node_prob[act, obv]
        
        if u > self.epsilon:
            act_prob = self.exploit_act[:, act]
        else:
            act_prob = self.explore_act[:, act]
        
        # p(z_{t-1}) * p(a_{t-1}|z)
        p_az = self.p_z * act_prob
        # p(z_t, z_{t-1}, a_{t-1}|o_t)
        p_azz_o = p_az[..., np.newaxis] * p_z_zao
        # marginalize z_{t-1}, get p(z_t, a_{t-1}|o_t)
        p_az_o = np.sum(p_azz_o, axis = 0)
        # normalize, get p(z_t|a_{t-1}, o_t)
        self.p_z = p_az_o / np.sum(p_az_o)
        
        
###############################################################################
class learned_policy:
    def __init__(self, policy_parameters):
        # policy parameters = (delta, mu, phi, sigma, lambda)
        delta = policy_parameters[0]   # for generating p(z_0)
        mu = policy_parameters[1]      # for generating p(z_0)
        phi = policy_parameters[2]     # for generating p(a|z)
        theta = policy_parameters[3]   # for generating p(a|z)
        sigma = policy_parameters[4]   # for generating p(z|z, a, o)
        lambda_ = policy_parameters[5] # for generating p(z|z, a, o)
        
        # find nodes that are assigned non-positive rewards
        removed_nodes = np.sum(phi - theta, axis = -1)
        removed_nodes = tuple(np.where(removed_nodes <= 0)[0])
        
        # if there is only 1 node left, this policy is stateless
        # then there is no node transition
        if len(removed_nodes) < len(delta) - 1:
            self.stateless = False
        else:
            self.stateless = True
        
        # initial node & node transition probabilities exist "only"
        # in policy with |Z| > 1
        if self.stateless == False:
            # remove nodes in delta & mu arrays
            delta = np.delete(delta, removed_nodes)
            mu = np.delete(mu, removed_nodes)
            # compute initial node distribution
            u = delta / (delta + mu)
            self.eta = np.empty_like(u)
            self.eta[0] = u[0]
            self.eta[1: ] = u[1: ] * (1 - u[: -1]).cumprod()
        
            # remove nodes in sigma & lambda arrays
            sigma = np.delete(sigma, removed_nodes, axis = 2)
            sigma = np.delete(sigma, removed_nodes, axis = 3)
            lambda_ = np.delete(lambda_, removed_nodes, axis = 2)
            lambda_ = np.delete(lambda_, removed_nodes, axis = 3)
            # compute node distribution given previous node, action, observation
            v = sigma / (sigma + lambda_)
            self.node_prob = np.empty_like(v)
            self.node_prob[..., 0] = v[..., 0]
            self.node_prob[..., 1:] = v[..., 1:] * (1 - v[..., :-1]).cumprod(axis = -1)
       
        # remove nodes in phi array
        phi = np.delete(phi, removed_nodes, axis = 0)
        # compute action distribution given nodes
        self.action_prob = phi / np.sum(phi, axis = -1)[..., None]
       
        
    def reset_policy(self):
        # for stateless policy, always stay at node 0
        if self.stateless == True:
            self.current_node = 0
        
        # initial node is meanful only when |Z| > 1
        else:
            # sample the initial node
            node = np.random.default_rng().multinomial(1, self.eta, size = 1)[0]
            self.current_node = np.where(node == 1)[0][0]
        
            self.not_1st_state = False  # flag for initial state
        
    
    def select_action(self, act_pre: int = -1, obv_pre: int = -1):
        # for stateless policy, always stay at node 0
        if self.stateless == True:
            self.current_node = 0
        
        # not stateless, sample next node
        else:
            # sample next node if it is not initial state
            if self.not_1st_state:
                self.current_node = self.next_node(self.current_node, act_pre, obv_pre)
            # is initial state, set flag for future update
            else:
                self.not_1st_state = True
        
        # extract action distribution for current node
        action_dist = self.action_prob[self.current_node, :]
        # pick an action given node
        act = np.random.default_rng().multinomial(1, action_dist, size = 1)[0]
        
        return np.where(act == 1)[0][0]
        
    
    def next_node(self, current_node: int, act_pre: int, obv_pre: int):
        # extract probability vector for given node, action, observation
        node_dist = self.node_prob[act_pre, obv_pre, current_node, :]
        # pick a noode
        node = np.random.default_rng().multinomial(1, node_dist, size = 1)[0]
        
        return np.where(node == 1)[0][0]
    


# =============================================================================
# # for test 
# class behavior_policy2(object):
#     def __init__(self, A, O, Z, epsilon, phi = None, sigma = None, lambda_ = None):
#         self.A = A
#         self.O = O
#         self.Z = Z
#         self.epsilon = epsilon
#         self.test = 3
#         
#         self.prob_table()
#         
#     
#     def prob_table(self):
#         # initial node distribution
#         self.eta = np.zeros(self.Z)
#         self.eta[0: self.test] += 1
#         self.eta /= self.test
#         
#         self.action_prob = np.random.dirichlet([1.] * self.A, self.Z)
#         
#         # initial node transition prob
#         v = np.random.beta(1, 1, (self.A, self.O, self.Z, self.test))
#         vv = (1 - v[..., :-1]).cumprod(axis = -1)
#         tt = np.zeros((self.A, self.O, self.Z, self.test))
#         tt[..., 0] = v[..., 0]
#         tt[..., 1:-1] = v[..., 1:-1] * vv[..., :-1]
#         tt[...,-1] = vv[...,-1]
#         tt /= np.sum(tt, axis = -1)[..., np.newaxis]
#         
#         self.node_prob = np.zeros((self.A, self.O, self.Z, self.Z))
#         self.node_prob[..., 0: 3] = tt
#         
#             
#     def refresh_prob(self):
#         self.t = 0
#         node = np.random.default_rng().multinomial(1, self.eta, size = 1)[0]
#         self.cur_node = np.where(node == 1)[0][0]
#         
#         
#     
#     def select_action(self, act_pre: int = -1, obv_pre: int = -1):
#         '''
#         Parameters
#         ----------
#         act_pre : int, optional
#             previous action index. The default is -1.
#         obv_pre : int, optional
#             previous observation index. The default is -1.
#         Returns
#         -------
#         TYPE
#             DESCRIPTION.
#         '''
#         if self.t > 0:
#             next_node_prob = self.node_prob[act_pre, obv_pre, self.cur_node]
#             node = np.random.default_rng().multinomial(1, next_node_prob, size = 1)[0]
#             self.cur_node = np.where(node == 1)[0][0]
#             
#         self.t += 1
#         
#         act_prob = self.action_prob[self.cur_node]
#         
#         # pick an action
#         act = np.random.default_rng().multinomial(1, act_prob, size = 1)[0]
#         
#         return np.where(act == 1)[0][0]
# =============================================================================
    
    

def initializeFSCs(N, episode, data):
    '''
    Parameters
    ----------
    N : int
        number of agents.
    episode : int
        number of episodes.
    data : list
        trajectories of (act, obv, reward) histories.

    Returns
    -------
    policies : list
        list of initialized FSC policies.
    '''
    # list for initial FSC policies of all agents
    policies = []
    
    for n in range(N):
        graph = nx.DiGraph()
        graph.add_node(0, data = None)  # root node (initial node)
        edges = dict()
        index_counter = 1
    
        # build initial tree from episodes
        for ep in range(episode):
            act = data[ep][0][n, :]
            index = np.where(act >= 0)[0]
        
            act = data[ep][0][n, tuple(index)]
            obv = data[ep][1][n, tuple(index)]
            rwd = data[ep][2][index]
    
            # pointer to current parent node
            # 1st (a,o,r) in each episode always connects to root node
            ptr = 0
    
            for i in range(len(act)):
                aor_pre = graph.nodes[ptr]["data"]
                aor_cur = (act[i], obv[i], rwd[i])
        
                if (aor_pre, aor_cur) in edges:
                    graph = mergeNode(graph, edges[(aor_pre, aor_cur)][0], ptr)
            
                    for key in edges:
                        if key[1] == aor_pre and edges[key][1] == ptr:
                            edges[key] = (edges[key][0], edges[(aor_pre, aor_cur)][0])
                            break
                    ptr = edges[(aor_pre, aor_cur)][1]
                else:
                    graph.add_node(index_counter, data = aor_cur)
                    graph.add_edge(ptr, index_counter)
                    edges[(aor_pre, aor_cur)] = (ptr, index_counter)
                    ptr = index_counter
                    index_counter += 1
                
        
        # search terminal nodes, connect them to root node
        for n in graph.nodes():
            if list(graph.successors(n)) == []:
                graph.add_edge(n, 0)
        
        # reindex nodes with consecutive numbers (initial node = 0)
        graph = nx.convert_node_labels_to_integers(graph, first_label = 0, ordering = "sorted")
        
        policies.append(graph)
    
    # compute the controller size for all agents
    policy_size = np.fromiter(map(lambda g: g.number_of_nodes(), policies), dtype = np.int)
    
    return policies, policy_size


def mergeNode(G, n1, n2):
    '''
    Parameters
    ----------
    G : networkx graph object
        the graph to be processed.
    n1 : networkx node object
        the node to merge the other one.
    n2 : networkx node object
        the node to be merged.
    Returns
    -------
    G : networkx graph object
        the graph after node merging.
    '''
    # Get all predecessors and successors of the 2 nodes to be merged
    pre = set(G.predecessors(n1)) | set(G.predecessors(n2))
    suc = set(G.successors(n1)) | set(G.successors(n2))
    v = G.nodes[n1]["data"]
    # Remove old nodes
    G.remove_nodes_from([n1, n2])
    # Create new node the same as n1 (the merging node)
    G.add_node(n1, data = v)
    # Add predecessors and successors edges
    # We have DiGraph so there should be one edge per nodes pair
    G.add_edges_from([(p, n1) for p in pre])
    G.add_edges_from([(n1, s) for s in suc])
    
    return G