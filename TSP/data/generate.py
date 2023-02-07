import time
import argparse
import pprint as pp
import os

import pandas as pd
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

def gendata():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=10)
    parser.add_argument("--node_dim", type=int, default=2)
    parser.add_argument("--filename", type=str, default=None)
    opts = parser.parse_args()
    
    if opts.filename is None:
        opts.filename = f"tsp{opts.num_nodes}.txt"
    
    # Pretty print the run args
    pp.pprint(vars(opts))
    
    set_nodes_coord = np.random.random([opts.num_samples, opts.num_nodes, opts.node_dim])
    with open(opts.filename, "w") as f:
        start_time = time.time()
        for nodes_coord in set_nodes_coord:
            solution,cost = solve_tsp_dynamic_programming(distance_matrix(nodes_coord))
            f.write( " ".join( str(x)+str(" ")+str(y) for x,y in nodes_coord) )
            f.write( str(" ") + str('output') + str(" ") )
            f.write( str(" ").join( str(node_idx) for node_idx in solution) )
            f.write(" " + str(cost))
            # f.write( str(" ") + str(solution.tour[0]+1) + str(" ") )
            f.write( "\n" )
        end_time = time.time() - start_time
    
    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.num_nodes}.")
    print(f"Total time: {end_time/60:.1f}mins")
    print(f"Average time: {(end_time/60)/opts.num_samples:.1f}mins")

def distance_matrix(nodes_coord):
    # set_nodes_coord = np.random.random([opts.num_samples, opts.num_nodes, opts.node_dim])
    distance_matrix = np.zeros((nodes_coord.shape[0], nodes_coord.shape[0]))
    for i in range(nodes_coord.shape[0]-1):
        for j in range(i+1,nodes_coord.shape[0]):
            distance_matrix[i][j] = np.linalg.norm(nodes_coord[i]-nodes_coord[j])
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix
    
if __name__ == "__main__":
    gendata()
    
    
