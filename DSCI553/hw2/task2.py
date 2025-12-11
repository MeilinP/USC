import sys
import time
from pyspark import SparkContext
from itertools import combinations

def pcy_frequent_itemsets(basket_iterator, total_baskets, min_support, hash_buckets):
    baskets = list(basket_iterator)
    if not baskets:
        return []
    
    local_ratio = len(baskets) / total_baskets
    local_threshold = local_ratio * min_support
    
    item_frequency = {}
    bucket_counts = [0] * hash_buckets
    
    for basket in baskets:
        for item in basket:
            item_frequency[item] = item_frequency.get(item, 0) + 1
        
        basket_items = list(basket)
        for i in range(len(basket_items)):
            for j in range(i + 1, len(basket_items)):
                bucket_id = (hash(basket_items[i]) ^ hash(basket_items[j])) % hash_buckets
                bucket_counts[bucket_id] += 1
    
    frequent_buckets = set()
    for bucket_id, count in enumerate(bucket_counts):
        if count >= local_threshold:
            frequent_buckets.add(bucket_id)
    
    frequent_singles = set()
    result_itemsets = []
    
    for item, freq in item_frequency.items():
        if freq >= local_threshold:
            frequent_singles.add(item)
            result_itemsets.append(tuple([item]))
    
    if len(frequent_singles) < 2:
        return result_itemsets
    
    sorted_singles = sorted(frequent_singles)
    candidate_pairs = {}
    
    for i in range(len(sorted_singles)):
        for j in range(i + 1, len(sorted_singles)):
            item1, item2 = sorted_singles[i], sorted_singles[j]
            bucket_id = (hash(item1) ^ hash(item2)) % hash_buckets
            if bucket_id in frequent_buckets:
                candidate_pairs[(item1, item2)] = 0
    
    for basket in baskets:
        basket_set = set(basket)
        for pair in candidate_pairs.keys():
            if pair[0] in basket_set and pair[1] in basket_set:
                candidate_pairs[pair] += 1
    
    frequent_pairs = set()
    for pair, count in candidate_pairs.items():
        if count >= local_threshold:
            frequent_pairs.add(pair)
            result_itemsets.append(pair)
    
    if not frequent_pairs:
        return result_itemsets
    
    filtered_baskets = []
    for basket in baskets:
        filtered = [item for item in basket if item in frequent_singles]
        if len(filtered) >= 3:
            filtered_baskets.append(filtered)
    
    current_frequent = frequent_pairs
    k = 3
    
    while current_frequent:
        all_items = set()
        for itemset in current_frequent:
            all_items.update(itemset)
        
        candidate_k = list(combinations(sorted(all_items), k))
        if not candidate_k:
            break
        
        candidate_counts = {cand: 0 for cand in candidate_k}
        
        for basket in filtered_baskets:
            if len(basket) < k:
                continue
            basket_set = set(basket)
            for candidate in candidate_k:
                if all(item in basket_set for item in candidate):
                    candidate_counts[candidate] += 1
        
        next_frequent = set()
        for candidate, count in candidate_counts.items():
            if count >= local_threshold:
                next_frequent.add(candidate)
                result_itemsets.append(candidate)
        
        if not next_frequent:
            break
        
        current_frequent = next_frequent
        k += 1
    
    return result_itemsets


def count_global_support(basket_iterator, candidate_itemsets):

    baskets = list(basket_iterator)
    support_counts = {}
    
    for basket in baskets:
        basket_set = set(basket)
        for itemset in candidate_itemsets:
            if all(item in basket_set for item in itemset):
                support_counts[itemset] = support_counts.get(itemset, 0) + 1
    
    return support_counts.items()


def format_output(itemsets):
    grouped = {}
    for itemset in itemsets:
        size = len(itemset)
        if size not in grouped:
            grouped[size] = []
        grouped[size].append(tuple(sorted(itemset)))
    
    for size in grouped:
        grouped[size] = sorted(set(grouped[size]))
    
    output_lines = []
    for size in sorted(grouped.keys()):
        items = grouped[size]
        if size == 1:
            formatted = ','.join([f"('{item[0]}')" for item in items])
        else:
            formatted = ','.join([str(item) for item in items])
        output_lines.append(formatted)
    
    return '\n\n'.join(output_lines)


if __name__ == '__main__':
    filter_threshold = int(sys.argv[1])
    support_threshold = int(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    
    start_time = time.time()
    
    sc = SparkContext(appName='task2_pcy')
    sc.setLogLevel('ERROR')
    
    raw_data = sc.textFile(input_path)
    header = raw_data.first()
    
    def extract_transaction(line):
        fields = line.replace('"', '').split(',')
        if len(fields) < 6:
            return None
        
        date = fields[0]
        customer = fields[1]
        product = fields[5]
        
        basket_id = f"{date}-{customer}"
        
        try:
            product = str(int(float(product)))
        except:
            return None
        
        return (basket_id, product)
    
    transactions = raw_data.filter(lambda x: x != header) \
                           .map(extract_transaction) \
                           .filter(lambda x: x is not None)
    
    baskets = transactions.groupByKey() \
                         .mapValues(lambda products: list(set(products))) \
                         .filter(lambda x: len(x[1]) > filter_threshold) \
                         .values()
    
    baskets.cache()
    
    total_baskets = baskets.count()
    num_partitions = baskets.getNumPartitions()
    hash_buckets = 2000  
    candidates = baskets.mapPartitions(
        lambda partition: pcy_frequent_itemsets(
            partition, 
            total_baskets, 
            support_threshold, 
            hash_buckets
        )
    ).distinct().collect()
    
    candidates_output = format_output(candidates)
    
    candidate_set = set(candidates)
    candidate_broadcast = sc.broadcast(candidate_set)
    
    def verify_support(partition):
        return count_global_support(partition, candidate_broadcast.value)
    
    frequent = baskets.mapPartitions(verify_support) \
                     .reduceByKey(lambda a, b: a + b) \
                     .filter(lambda x: x[1] >= support_threshold) \
                     .keys() \
                     .collect()
    
    frequent_output = format_output(frequent)
    
    with open(output_path, 'w') as file:
        file.write("Candidates:\n")
        file.write(candidates_output)
        file.write("\n\n")
        file.write("Frequent Itemsets:\n")
        file.write(frequent_output)
    
    baskets.unpersist()
    sc.stop()
    
    elapsed = int(time.time() - start_time)
    print(f"Duration: {elapsed}")