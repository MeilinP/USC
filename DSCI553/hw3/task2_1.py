import sys
from pyspark import SparkContext
import time

def pearson_similarity(ratings_a, ratings_b):
    n = len(ratings_a)
    if n == 0:
        return 0
    
    mean_a = sum(ratings_a) / n
    mean_b = sum(ratings_b) / n
    
    numerator = sum((ratings_a[i] - mean_a) * (ratings_b[i] - mean_b) for i in range(n))
    
    sum_sq_a = sum((ratings_a[i] - mean_a) ** 2 for i in range(n))
    sum_sq_b = sum((ratings_b[i] - mean_b) ** 2 for i in range(n))
    
    denominator = (sum_sq_a ** 0.5) * (sum_sq_b ** 0.5)
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

def calculate_similarity(biz_a, biz_b):
    common_users = biz_user_map[biz_a] & biz_user_map[biz_b]
    
    co_rated_count = len(common_users)
    
    if co_rated_count <= 1:
        diff = abs(biz_avg_map[biz_a] - biz_avg_map[biz_b])
        return (5.0 - diff) / 5.0
    
    elif co_rated_count == 2:
        users_list = list(common_users)
        diff1 = abs(float(biz_user_rating_map[biz_a][users_list[0]]) - 
                    float(biz_user_rating_map[biz_a][users_list[1]]))
        diff2 = abs(float(biz_user_rating_map[biz_b][users_list[0]]) - 
                    float(biz_user_rating_map[biz_b][users_list[1]]))
        sim1 = (5.0 - diff1) / 5.0
        sim2 = (5.0 - diff2) / 5.0
        return (sim1 + sim2) / 2
    
    else:
        ratings_a = []
        ratings_b = []
        for u in common_users:
            ratings_a.append(float(biz_user_rating_map[biz_a][u]))
            ratings_b.append(float(biz_user_rating_map[biz_b][u]))
        
        return pearson_similarity(ratings_a, ratings_b)

def predict_rating(biz_id, user_id):
    if user_id not in user_biz_map:
        return 3.5
    
    if biz_id not in biz_user_map:
        return user_avg_map[user_id]
    
    user_rated_biz = user_biz_map[user_id]
    
    similarity_rating_pairs = []
    
    for rated_biz in user_rated_biz:
        pair_key = tuple(sorted([rated_biz, biz_id]))
        
        if pair_key in similarity_cache:
            sim = similarity_cache[pair_key]
        else:
            sim = calculate_similarity(biz_id, rated_biz)
            similarity_cache[pair_key] = sim
        
        user_rating = float(biz_user_rating_map[rated_biz][user_id])
        similarity_rating_pairs.append((sim, user_rating))
    
    similarity_rating_pairs.sort(key=lambda x: -x[0])
    top_neighbors = similarity_rating_pairs[:15]
    
    numerator = sum(sim * rating for sim, rating in top_neighbors)
    denominator = sum(abs(sim) for sim, rating in top_neighbors)
    
    if denominator == 0:
        return 3.5
    
    return numerator / denominator

if __name__ == '__main__':
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    
    start_time = time.time()
    
    sc = SparkContext(appName='task2_1')
    sc.setLogLevel('ERROR')
    
    train_rdd = sc.textFile(train_file_path)
    train_header = train_rdd.first()
    train_data = train_rdd.filter(lambda x: x != train_header) \
                          .map(lambda x: x.split(',')) \
                          .map(lambda x: (x[1], x[0], x[2])) 
    
    biz_user_rdd = train_data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set)
    biz_user_map = {}
    for biz, users in biz_user_rdd.collect():
        biz_user_map[biz] = users
    
    user_biz_rdd = train_data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)
    user_biz_map = {}
    for user, businesses in user_biz_rdd.collect():
        user_biz_map[user] = businesses
    
    biz_avg_rdd = train_data.map(lambda x: (x[0], float(x[2]))) \
                            .groupByKey() \
                            .mapValues(list) \
                            .map(lambda x: (x[0], sum(x[1]) / len(x[1])))
    biz_avg_map = {}
    for biz, avg in biz_avg_rdd.collect():
        biz_avg_map[biz] = avg
    
    user_avg_rdd = train_data.map(lambda x: (x[1], float(x[2]))) \
                             .groupByKey() \
                             .mapValues(list) \
                             .map(lambda x: (x[0], sum(x[1]) / len(x[1])))
    user_avg_map = {}
    for user, avg in user_avg_rdd.collect():
        user_avg_map[user] = avg
    
    biz_user_rating_rdd = train_data.map(lambda x: (x[0], (x[1], x[2]))) \
                                    .groupByKey() \
                                    .mapValues(set)
    biz_user_rating_map = {}
    for biz, user_rating_set in biz_user_rating_rdd.collect():
        rating_dict = {}
        for user, rating in user_rating_set:
            rating_dict[user] = rating
        biz_user_rating_map[biz] = rating_dict
    
    test_rdd = sc.textFile(test_file_path)
    test_header = test_rdd.first()
    test_data = test_rdd.filter(lambda x: x != test_header) \
                        .map(lambda x: x.split(',')) \
                        .map(lambda x: (x[1], x[0]))  
    
    similarity_cache = {}
    
    output_lines = "user_id, business_id, prediction\n"
    for biz, user in test_data.collect():
        pred = predict_rating(biz, user)
        output_lines += f"{user},{biz},{pred}\n"
    
    with open(output_file_path, 'w') as f:
        f.write(output_lines)
    
    sc.stop()
    
    end_time = time.time()
    print('Duration:', end_time - start_time)