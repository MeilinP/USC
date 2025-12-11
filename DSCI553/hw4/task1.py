import sys
from pyspark import SparkContext
from pyspark.sql import SQLContext
from graphframes import GraphFrame

def main():
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    community_output_file_path = sys.argv[3]
    
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('ERROR')
    sqlContext = SQLContext(sc)
    
    rdd = sc.textFile(input_file_path)
    header = rdd.first()
    data = rdd.filter(lambda line: line != header).map(lambda line: line.split(','))
    
    user_business = data.map(lambda x: (x[0], x[1]))
    
    user_businesses = user_business.groupByKey().mapValues(set).collectAsMap()
    
    all_users = set(user_businesses.keys())
    
    edges = []
    users_list = sorted(all_users)
    
    for i in range(len(users_list)):
        for j in range(i + 1, len(users_list)):
            user1 = users_list[i]
            user2 = users_list[j]
            
            if user1 in user_businesses and user2 in user_businesses:
                common_businesses = len(user_businesses[user1] & user_businesses[user2])
                
                if common_businesses >= filter_threshold:
                    edges.append((user1, user2))
                    edges.append((user2, user1))
    
    nodes_with_edges = set()
    for edge in edges:
        nodes_with_edges.add(edge[0])
        nodes_with_edges.add(edge[1])
    
    vertices = sqlContext.createDataFrame([(user,) for user in nodes_with_edges], ['id'])
    edges_df = sqlContext.createDataFrame(edges, ['src', 'dst'])
    
    graph = GraphFrame(vertices, edges_df)
    
    result = graph.labelPropagation(maxIter=5)
    
    communities_rdd = result.rdd.map(lambda row: (row['label'], row['id']))
    communities = communities_rdd.groupByKey().mapValues(list).map(lambda x: sorted(x[1])).collect()
    
    communities.sort(key=lambda x: (len(x), x[0]))
    
    with open(community_output_file_path, 'w') as f:
        for community in communities:
            f.write("'" + "', '".join(community) + "'\n")
    
    sc.stop()

if __name__ == '__main__':
    main()