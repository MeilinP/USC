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

if __name__ == '__main__':
    folder_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    
    start_time = time.time()
    
    sc = SparkContext(appName="task2_2")
    sc.setLogLevel('ERROR')
    
    train_rdd = sc.textFile(folder_path + '/yelp_train.csv')
    train_header = train_rdd.first()
    train_data = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))
    
    review_rdd = sc.textFile(folder_path + '/review_train.json') \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], (safe_float(x.get('useful', 0)), 
                                            safe_float(x.get('funny', 0)), 
                                            safe_float(x.get('cool', 0))))) \
        .groupByKey() \
        .mapValues(lambda vals: list(vals))
    
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
        .map(lambda x: (x['user_id'], (safe_float(x.get('average_stars', 0)), 
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
        .map(lambda x: (x['business_id'], (safe_float(x.get('stars', 0)), 
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
    
    for row in train_data.collect():
        user_id = row[0]
        bus_id = row[1]
        rating = safe_float(row[2], 3.5)
        
        if bus_id in review_dict:
            useful, funny, cool = review_dict[bus_id]
        else:
            useful, funny, cool = default_useful, default_funny, default_cool
        
        if user_id in user_dict:
            user_stars, user_reviews, user_fans = user_dict[user_id]
        else:
            user_stars, user_reviews, user_fans = default_user_stars, default_user_reviews, default_user_fans
        
        if bus_id in bus_dict:
            bus_stars, bus_reviews = bus_dict[bus_id]
        else:
            bus_stars, bus_reviews = default_bus_stars, default_bus_reviews
        
        X_train.append([useful, funny, cool, user_stars, user_reviews, user_fans, bus_stars, bus_reviews])
        y_train.append(rating)
    
    test_rdd = sc.textFile(test_path)
    test_header = test_rdd.first()
    test_data = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))
    
    X_test = []
    test_pairs = []
    
    for row in test_data.collect():
        user_id = row[0]
        bus_id = row[1]
        test_pairs.append((user_id, bus_id))
        
        if bus_id in review_dict:
            useful, funny, cool = review_dict[bus_id]
        else:
            useful, funny, cool = default_useful, default_funny, default_cool
        
        if user_id in user_dict:
            user_stars, user_reviews, user_fans = user_dict[user_id]
        else:
            user_stars, user_reviews, user_fans = default_user_stars, default_user_reviews, default_user_fans
        
        if bus_id in bus_dict:
            bus_stars, bus_reviews = bus_dict[bus_id]
        else:
            bus_stars, bus_reviews = default_bus_stars, default_bus_reviews
        
        X_test.append([useful, funny, cool, user_stars, user_reviews, user_fans, bus_stars, bus_reviews])
    
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    
    with open(output_path, 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for i in range(len(y_pred)):
            user_id, bus_id = test_pairs[i]
            prediction = y_pred[i]
            f.write(f'{user_id},{bus_id},{prediction}\n')
    
    end_time = time.time()
    print(f'Duration: {end_time - start_time}')
    
    sc.stop()