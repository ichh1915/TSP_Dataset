from __future__ import print_function

import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
import math

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp




def create_data_model(D):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = D  # yapf: disable
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    # print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    route_list = []
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        route_list.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    # print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)
    route_list.append(0)
    return route_list,route_distance

def TSP(D):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(D)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    # Tabu Search 
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.solution_limit = 50

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        route_list,route_distance = print_solution(manager, routing, solution)
        print('solved TSP')

    return route_list, route_distance

def draw_Routes(pos,A):
    G = nx.from_numpy_matrix(A,create_using=nx.DiGraph())
    """
    NoStp = np.shape(A)[0]
    for i in range(NoStp):
            G.nodes[i]['pos'] = (pos[i][0], pos[i][1])
    """
    nx.draw_networkx_nodes(G,pos,node_size=500,node_color='r')
    nx.draw_networkx_edges(G,pos,width=8,arrows=True,edge_color='r')           
    nx.draw_networkx_labels(G,pos,font_size=25)
