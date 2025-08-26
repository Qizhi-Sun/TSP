import numpy as np
import json
from typing import Dict, List, Tuple, Optional

"""
This func takes json cities and return dict as 
 {
     0: {"index": 0, "coord": np.array([x1, y1]), "name": "City 0"},
     1: {"index": 1, "coord": np.array([x2, y2]), "name": "City 1"},
     ...
 }
"""
def cities_form_transfer(json_cities: str, is_3d: bool = False) -> Dict:
    """
    :param is_3d: is 3d or not
    :param json_cities: file_name
    :return: dict cities
    """
    cities_dict = {}
    cities_dict_batch = {}
    with open(json_cities) as f:
        cities = json.load(f)
        for i in range(len(cities)):
            cities_batch = cities[f"batch{i}"]
            for j in range(len(cities_batch)):
                cities_dict[j] = {
                    "index": j,
                    "coord": np.array([cities_batch[f"node{j}"]["x"],
                                       cities_batch[f"node{j}"]["y"]]
                                       + ([cities_batch[f"node{j}"]["z"]] if is_3d else [])
                                     ),
                    "name": f"City {j}"
                }
            cities_dict_batch[f"batch{i}"] = cities_dict
    return cities_dict_batch
