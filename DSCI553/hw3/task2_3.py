from pyspark import SparkContext
import json
import sys
import time
from xgboost import XGBRegressor


def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except:
        return default


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
                    float(biz_user_rating_map[biz_b][users_list[0]]))
        diff2 = abs(float(biz_user_rating_map[biz_a][users_list[1]]) - 
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


def predict_item_based(biz_id, user_id):
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
    folder_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    
    start_time = time.time()
    
    sc = SparkContext(appName="task2_3")
    sc.setLogLevel('ERROR')
    
    train_rdd = sc.textFile(folder_path + '/yelp_train.csv')
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
    
    similarity_cache = {}
    
    # Review features
    review_rdd = sc.textFile(folder_path + '/review_train.json') \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], 
                       (safe_float(x.get('useful', 0)), 
                        safe_float(x.get('funny', 0)), 
                        safe_float(x.get('cool', 0))))) \
        .groupByKey() \
        .mapValues(list)
    
    review_dict = {}
    for bus_id, items in review_rdd.collect():
        if len(items) > 0:
            useful_sum = sum(item[0] for item in items)
            funny_sum = sum(item[1] for item in items)
            cool_sum = sum(item[2] for item in items)
            count = len(items)
            review_dict[bus_id] = (useful_sum / count, funny_sum / count, cool_sum / count)
    
    if review_dict:
        all_useful = [v[0] for v in review_dict.values()]
        all_funny = [v[1] for v in review_dict.values()]
        all_cool = [v[2] for v in review_dict.values()]
        default_useful = sum(all_useful) / len(all_useful)
        default_funny = sum(all_funny) / len(all_funny)
        default_cool = sum(all_cool) / len(all_cool)
    else:
        default_useful = default_funny = default_cool = 0.0
    
    user_rdd = sc.textFile(folder_path + '/user.json') \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], 
                       (safe_float(x.get('average_stars', 0)), 
                        safe_float(x.get('review_count', 0)), 
                        safe_float(x.get('fans', 0)))))
    
    user_dict = dict(user_rdd.collect())
    
    if user_dict:
        all_user_stars = [v[0] for v in user_dict.values()]
        all_user_reviews = [v[1] for v in user_dict.values()]
        all_user_fans = [v[2] for v in user_dict.values()]
        default_user_stars = sum(all_user_stars) / len(all_user_stars)
        default_user_reviews = sum(all_user_reviews) / len(all_user_reviews)
        default_user_fans = sum(all_user_fans) / len(all_user_fans)
    else:
        default_user_stars = 3.5
        default_user_reviews = default_user_fans = 0.0
    
    bus_rdd = sc.textFile(folder_path + '/business.json') \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], 
                       (safe_float(x.get('stars', 0)), 
                        safe_float(x.get('review_count', 0)))))
    
    bus_dict = dict(bus_rdd.collect())
    
    if bus_dict:
        all_bus_stars = [v[0] for v in bus_dict.values()]
        all_bus_reviews = [v[1] for v in bus_dict.values()]
        default_bus_stars = sum(all_bus_stars) / len(all_bus_stars)
        default_bus_reviews = sum(all_bus_reviews) / len(all_bus_reviews)
    else:
        default_bus_stars = 3.5
        default_bus_reviews = 0.0
    
    X_train = []
    y_train = []
    
    train_list = train_data.collect()
    for biz_id, user_id, rating in train_list:
        if biz_id in review_dict:
            useful, funny, cool = review_dict[biz_id]
        else:
            useful, funny, cool = default_useful, default_funny, default_cool
        
        if user_id in user_dict:
            user_stars, user_reviews, user_fans = user_dict[user_id]
        else:
            user_stars, user_reviews, user_fans = default_user_stars, default_user_reviews, default_user_fans
        
        if biz_id in bus_dict:
            bus_stars, bus_reviews = bus_dict[biz_id]
        else:
            bus_stars, bus_reviews = default_bus_stars, default_bus_reviews
        
        X_train.append([useful, funny, cool, user_stars, user_reviews, user_fans, bus_stars, bus_reviews])
        y_train.append(float(rating))
    
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    xgb_model.fit(X_train, y_train)
    
    test_rdd = sc.textFile(test_path)
    test_header = test_rdd.first()
    test_data = test_rdd.filter(lambda x: x != test_header) \
                        .map(lambda x: x.split(',')) \
                        .map(lambda x: (x[1], x[0]))
    
    test_pairs = test_data.collect()
    
    output_lines = "user_id, business_id, prediction\n"
    
    for biz_id, user_id in test_pairs:
        cf_pred = predict_item_based(biz_id, user_id)
        
        if biz_id in review_dict:
            useful, funny, cool = review_dict[biz_id]
        else:
            useful, funny, cool = default_useful, default_funny, default_cool
        
        if user_id in user_dict:
            user_stars, user_reviews, user_fans = user_dict[user_id]
        else:
            user_stars, user_reviews, user_fans = default_user_stars, default_user_reviews, default_user_fans
        
        if biz_id in bus_dict:
            bus_stars, bus_reviews = bus_dict[biz_id]
        else:
            bus_stars, bus_reviews = default_bus_stars, default_bus_reviews
        
        X_test = [[useful, funny, cool, user_stars, user_reviews, user_fans, bus_stars, bus_reviews]]
        model_pred = xgb_model.predict(X_test)[0]
        
        alpha = 0.12
        final_pred = alpha * cf_pred + (1 - alpha) * model_pred
        
        output_lines += f"{user_id},{biz_id},{final_pred}\n"
    
    with open(output_path, 'w') as f:
        f.write(output_lines)
    
    end_time = time.time()
    print(f'Duration: {end_time - start_time}')
    
    sc.stop()