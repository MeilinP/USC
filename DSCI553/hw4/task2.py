import sys
from pyspark import SparkContext
from collections import defaultdict
from copy import deepcopy

def compute_edge_betweenness(adjacency_map, node_list):
    edge_betweenness_scores = defaultdict(float)
    
    for source_node in node_list:
        predecessor_map = defaultdict(set)
        distance_map = {}
        shortest_path_count = defaultdict(float)
        traversal_order = []
        bfs_queue = []
        
        bfs_queue.append(source_node)
        explored_nodes = set()
        explored_nodes.add(source_node)
        distance_map[source_node] = 0
        shortest_path_count[source_node] = 1
        
        while len(bfs_queue) > 0:
            current_node = bfs_queue.pop(0)
            traversal_order.append(current_node)
            
            for adjacent_node in adjacency_map[current_node]:
                if adjacent_node not in explored_nodes:
                    bfs_queue.append(adjacent_node)
                    explored_nodes.add(adjacent_node)
                    
                    if adjacent_node not in predecessor_map.keys():
                        predecessor_map[adjacent_node] = set()
                        predecessor_map[adjacent_node].add(current_node)
                    else:
                        predecessor_map[adjacent_node].add(current_node)
                    
                    shortest_path_count[adjacent_node] += shortest_path_count[current_node]
                    distance_map[adjacent_node] = distance_map[current_node] + 1
                    
                elif distance_map[adjacent_node] == distance_map[current_node] + 1:
                    if adjacent_node not in predecessor_map.keys():
                        predecessor_map[adjacent_node] = set()
                        predecessor_map[adjacent_node].add(current_node)
                    else:
                        predecessor_map[adjacent_node].add(current_node)
                    
                    shortest_path_count[adjacent_node] += shortest_path_count[current_node]
        
        node_weight_map = {}
        for node in traversal_order:
            node_weight_map[node] = 1
        
        edge_contribution = defaultdict(float)
        
        for node in reversed(traversal_order):
            for predecessor in predecessor_map[node]:
                contribution = node_weight_map[node] * (shortest_path_count[predecessor] / shortest_path_count[node])
                node_weight_map[predecessor] += contribution
                
                edge_key = tuple(sorted([node, predecessor]))
                if edge_key not in edge_contribution.keys():
                    edge_contribution[edge_key] = contribution
                else:
                    edge_contribution[edge_key] += contribution
        
        for edge, contribution_value in edge_contribution.items():
            if edge not in edge_betweenness_scores.keys():
                edge_betweenness_scores[edge] = contribution_value / 2
            else:
                edge_betweenness_scores[edge] += contribution_value / 2
    
    sorted_betweenness = sorted(edge_betweenness_scores.items(), key=lambda x: (-x[1], x[0]))
    return sorted_betweenness


if __name__ == '__main__':
    threshold_value = sys.argv[1]
    data_input_path = sys.argv[2]
    betweenness_file_path = sys.argv[3]
    community_file_path = sys.argv[4]
    
    spark_context = SparkContext('local[*]', 'task2')
    spark_context.setLogLevel('ERROR')
    
    data_lines = spark_context.textFile(data_input_path)
    header_line = data_lines.first()
    data_lines = data_lines.filter(lambda row: row != header_line).map(lambda row: row.split(","))
    
    user_to_businesses = data_lines.groupByKey().mapValues(set)
    business_set_by_user = {}
    for uid, bids in user_to_businesses.collect():
        business_set_by_user[uid] = bids
    
    all_users = data_lines.map(lambda row: row[0]).distinct()
    user_pairs = []
    user_collection = all_users.collect()
    
    for i in range(len(user_collection)):
        for j in range(i + 1, len(user_collection)):
            user_pairs.append((user_collection[i], user_collection[j]))
            user_pairs.append((user_collection[j], user_collection[i]))
    
    edge_list = []
    vertex_collection = set()
    for user_a, user_b in user_pairs:
        common_businesses = business_set_by_user[user_a] & business_set_by_user[user_b]
        if len(common_businesses) >= int(threshold_value):
            edge_list.append((user_a, user_b))
            vertex_collection.add(user_a)
    
    vertices = list(vertex_collection)
    
    adjacency_structure = {}
    for user_x, user_y in edge_list:
        if user_x not in adjacency_structure.keys():
            adjacency_structure[user_x] = set()
            adjacency_structure[user_x].add(user_y)
        else:
            adjacency_structure[user_x].add(user_y)
        
        if user_y not in adjacency_structure.keys():
            adjacency_structure[user_y] = set()
            adjacency_structure[user_y].add(user_x)
        else:
            adjacency_structure[user_y].add(user_x)
    
    betweenness_results = compute_edge_betweenness(adjacency_structure, vertices)
    
    with open(betweenness_file_path, "w") as output_file:
        for edge_tuple, score in betweenness_results:
            edge_str = str(edge_tuple)
            output_line = edge_str + "," + str(round(score, 5)) + "\n"
            output_file.write(output_line)
    
    working_graph = deepcopy(adjacency_structure)
    total_edges = len(betweenness_results)
    
    node_degrees = {node: len(adjacency_structure[node]) for node in adjacency_structure}
    
    best_modularity_score = -float('inf')
    optimal_communities = []
    
    while len(betweenness_results) > 0:
        component_list = []
        remaining_vertices = vertices.copy()
        
        while len(remaining_vertices) > 0:
            start_vertex = remaining_vertices.pop()
            component_queue = []
            component_queue.append(start_vertex)
            visited_set = set()
            visited_set.add(start_vertex)
            
            while len(component_queue) > 0:
                vertex = component_queue.pop(0)
                for neighbor in working_graph[vertex]:
                    if neighbor not in visited_set:
                        remaining_vertices.remove(neighbor)
                        component_queue.append(neighbor)
                        visited_set.add(neighbor)
            
            component_members = sorted(list(visited_set))
            component_list.append(component_members)
        
        modularity_value = 0.0
        for component in component_list:
            for node_i in component:
                for node_j in component:
                    has_edge = 1.0 if node_j in adjacency_structure[node_i] else 0.0
                    modularity_value += has_edge - (node_degrees[node_i] * node_degrees[node_j]) / (2.0 * total_edges)
        
        modularity_value /= (2 * total_edges)
        
        if modularity_value > best_modularity_score:
            best_modularity_score = modularity_value
            optimal_communities = deepcopy(component_list)
        
        max_betweenness = betweenness_results[0][1]
        for edge, betweenness_val in betweenness_results:
            if betweenness_val >= max_betweenness:
                working_graph[edge[0]].remove(edge[1])
                working_graph[edge[1]].remove(edge[0])
            
        betweenness_results = compute_edge_betweenness(working_graph, vertices)
    
    optimal_communities = sorted(optimal_communities, key=lambda x: (len(x), x[0]))
    
    with open(community_file_path, "w") as output_file:
        for community in optimal_communities:
            community_str = str(community)
            output_line = community_str[1:-1] + "\n"
            output_file.write(output_line)
    
    spark_context.stop()