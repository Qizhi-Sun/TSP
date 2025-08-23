import json

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
        x = graph[:, 0]
        y = graph[:, 1]
        plt.plot(x, y,'*-')
        fig.savefig(f"./images/results/{batch}.png")
        plt.close(fig)


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
        x = graph[:, 0]
        y = graph[:, 1]
        plt.scatter(x, y, marker='*')
        fig.savefig(f"./images/origin/{batch}.png")
        plt.close(fig)

"""
plotting based on the log 
"""
def plot_log(log_path: str) -> None:
    """
    :param log_path: log of the planning process
    :return: None
    """
    a_loss_log = []
    c_loss_log = []
    r_mean_log = []
    with open(log_path, "r") as f:
        log = json.load(f)
    for batch in range(len(log)):
        a_loss_log.append(log[f'record{batch+1}']["actor_loss"])
        c_loss_log.append(log[f'record{batch+1}']["critic_loss"])
        r_mean_log.append(log[f'record{batch+1}']["reward_mean"])
    fig = plt.figure()
    plt.plot(a_loss_log, label="a_loss")
    fig.savefig("./images/log/a_loss.png")
    plt.close(fig)
    fig = plt.figure()
    plt.plot(c_loss_log, label="c_loss")
    fig.savefig("./images/log/c_loss.png")
    plt.close(fig)
    fig = plt.figure()
    plt.plot(r_mean_log, label="r_mean")
    fig.savefig("./images/log/r_mean.png")
    plt.close(fig)

