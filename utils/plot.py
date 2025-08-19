import matplotlib.pyplot as plt
import numpy as np
from jmespath.ast import projection

"""
plotting based on the planning results
"""
def plot_results(graph_instance: np.ndarray, planning_results: np.ndarray) -> None:
    """
    :param graph_instance: da graph instance
    :param planning_results: sequential action on da graph
    :return: None
    """
    for batch in range(graph_instance.shape[0]):
        graph = graph_instance[batch]
        action = planning_results[batch]
        graph = graph[action, :]
        fig = plt.figure(batch)
        ax = fig.add_subplot(111, projection='3d')
        x = graph[:, 0]
        y = graph[:, 1]
        z = graph[:, 2]
        ax.scatter(x, y, z)
        fig.savefig(f"./images/origin/{batch}.png")


"""
plotting based on the original data 
"""
def plot_original(graph_instance: np.ndarray) -> None:
    """
    :param graph_instance: da graph instance
    :return: None
    """
    for batch in range(graph_instance.shape[0]):
        graph = graph_instance[batch]
        fig = plt.figure(batch)
        ax = fig.add_subplot(111, projection='3d')
        x = graph[:, 0]
        y = graph[:, 1]
        z = graph[:, 2]
        ax.scatter(x, y, z)
        fig.savefig(f"./images/origin/{batch}.png")
