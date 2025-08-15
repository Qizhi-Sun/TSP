import numpy as np

"""
compute routing length
"""
def compute_length(graph_instance:np.ndarray, action_seq:np.ndarray) -> np.ndarray:
    dis_mat = np.zeros([graph_instance.shape[0], graph_instance.shape[1], graph_instance.shape[1]]) # shape as [batch. n_nodes, n_nodes]
    rewards = np.zeros([graph_instance.shape[0]])
    for i in range(graph_instance.shape[1]):
        for j in range(i+1):
            dis_mat[:, i, j] = np.sqrt(np.sum(np.square(graph_instance[:, i, :] - graph_instance[:, j, :]), axis=1))
            dis_mat[:, j, i] = dis_mat[:, i, j]
    for i in range(action_seq.shape[1]):
        rewards += dis_mat[:, action_seq[:, i], action_seq[:, i+1]]
    return rewards
