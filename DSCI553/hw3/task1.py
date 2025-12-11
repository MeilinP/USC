import sys
from pyspark import SparkContext
import time
import random
from itertools import combinations

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    
    start_time = time.time()
    
    sc = SparkContext(appName='task1_lsh')
    sc.setLogLevel('ERROR')
    
    data_lines = sc.textFile(input_file_path)
    header = data_lines.first()
    data_lines = data_lines.filter(lambda line: line != header).map(lambda line: line.split(','))
    
    business_users_rdd = data_lines.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set)
    
    business_users_dict = {}
    for biz, users in business_users_rdd.collect():
        business_users_dict[biz] = users
    
    all_users = data_lines.map(lambda row: row[0]).distinct()
    user_to_index = {}
    index = 0
    for user in all_users.collect():
        user_to_index[user] = index
        index += 1
    
    num_hashes = 60
    num_users = index
    large_prime = int(1e9 + 7)
    
    random.seed(42)
    coeff_a = random.sample(range(1, num_users), num_hashes)
    coeff_b = random.sample(range(1, num_users), num_hashes)
    
    signatures = {}
    for biz, user_set in business_users_rdd.collect():
        signature_list = []
        for hash_idx in range(num_hashes):
            min_hash_value = float('inf')
            for user in user_set:
                hash_result = ((coeff_a[hash_idx] * user_to_index[user] + coeff_b[hash_idx]) % large_prime) % num_users
                min_hash_value = min(min_hash_value, hash_result)
            signature_list.append(int(min_hash_value))
        signatures[biz] = signature_list
    
    rows_per_band = 2
    num_bands = num_hashes // rows_per_band
    
    buckets = {}
    for biz, sig in signatures.items():
        for band_idx in range(num_bands):
            band_signature = tuple(sig[band_idx * rows_per_band : (band_idx + 1) * rows_per_band])
            bucket_id = (band_idx, band_signature)
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(biz)
    
    filtered_buckets = {}
    for bucket_id, biz_list in buckets.items():
        if len(biz_list) > 1:
            filtered_buckets[bucket_id] = biz_list
    
    candidate_pairs = set()
    for biz_list in filtered_buckets.values():
        sorted_list = sorted(biz_list)
        for pair in combinations(sorted_list, 2):
            candidate_pairs.add(pair)
    
    results = {}
    for biz1, biz2 in candidate_pairs:
        users1 = business_users_dict[biz1]
        users2 = business_users_dict[biz2]
        intersection = len(users1 & users2)
        union = len(users1 | users2)
        jaccard_sim = intersection / union
        
        if jaccard_sim >= 0.5:
            key = f"{biz1},{biz2}"
            results[key] = jaccard_sim
    
    results = dict(sorted(results.items()))
    
    output_content = "business_id_1, business_id_2, similarity\n"
    for key, sim in results.items():
        output_content += f"{key},{sim}\n"
    
    with open(output_file_path, 'w') as f:
        f.write(output_content)
    
    sc.stop()
    
    end_time = time.time()
    print('Duration:', end_time - start_time)