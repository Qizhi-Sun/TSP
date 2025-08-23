import torch
import numpy as np
import json

"""
generate
"""
def generate_data(batch_size:int, n_nodes:int, grid_edge:int, call_mode:bool) -> np.ndarray:
    """
    :param batch_size: num of each batch
    :param n_nodes: num of nodes in graph
    :param grid_edge: num of edge range
    :param call_mode: mode of train or eval which decides save json or not
    :return: data batch with shape [batch_size, n_nodes, 3]
    """
    graph_instance = np.random.uniform(0, 10, size=(batch_size, n_nodes, 2)) * grid_edge
    # save as json
    if not call_mode:
        t2j = {}
        for i in range (batch_size):
            t2j[f"batch{i}"] = {f"node{j}": {"x":graph_instance[i][j][0], "y":graph_instance[i][j][1]} for j in range(n_nodes)}
        graph_instance_json = json.dumps(t2j, indent=4)
        with open("./datalib/graph_instance_json.json", "w") as f:
            f.write(graph_instance_json)
    return graph_instance

"""
useful tool for convertion graph-data from json format to numpy 
"""
def json_to_np(file_name:str) -> np.ndarray:
    """
    :param file_name: name of json file waited for converting
    :return: numpy array data converting from json
    """
    with open(file_name, "r") as f:
        graph_instance_dict= json.load(f)
    batch_size = len(graph_instance_dict)
    n_nodes = len(graph_instance_dict["batch0"])
    graph_instance = np.zeros(shape=(batch_size, n_nodes, 2))
    for i in range(batch_size):
        for j in range(n_nodes):
            graph_instance[i][j][0] = graph_instance_dict[f"batch{i}"][f"node{j}"]["x"]
            graph_instance[i][j][1] = graph_instance_dict[f"batch{i}"][f"node{j}"]["y"]
            graph_instance[i][j][2] = graph_instance_dict[f"batch{i}"][f"node{j}"]["z"]
    return graph_instance

