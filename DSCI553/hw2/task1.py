import sys
import time
from pyspark import SparkContext
from itertools import combinations

def apriori(baskets, support):
    basket_list = list(baskets)
    if not basket_list:
        return []
    
    item_counts = {}
    for basket in basket_list:
        for item in basket:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    frequent = {}
    k = 1
    frequent[k] = set()
    for item, count in item_counts.items():
        if count >= support:
            frequent[k].add((item,))
    
    while frequent[k]:
        k += 1
        candidates = set()
        freq_list = list(frequent[k-1])
        for i in range(len(freq_list)):
            for j in range(i+1, len(freq_list)):
                union = tuple(sorted(set(freq_list[i]) | set(freq_list[j])))
                if len(union) == k:
                    valid = True
                    for subset in combinations(union, k-1):
                        if subset not in frequent[k-1]:
                            valid = False
                            break
                    if valid:
                        candidates.add(union)
        
        candidate_counts = {}
        for basket in basket_list:
            basket_set = set(basket)
            for candidate in candidates:
                if set(candidate).issubset(basket_set):
                    candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1
    
        frequent[k] = set()
        for candidate, count in candidate_counts.items():
            if count >= support:
                frequent[k].add(candidate)
        
        if not frequent[k]:
            break
    
    result = []
    for k in sorted(frequent.keys()):
        for itemset in frequent[k]:
            result.append(itemset)
    
    return result


if __name__ == '__main__':

    case_num = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    
    start_time = time.time()

    sc = SparkContext(appName='task1')
    sc.setLogLevel('ERROR')
    
    data = sc.textFile(input_file)
    header = data.first()
    data = data.filter(lambda x: x != header)
    
    pairs = data.map(lambda x: x.split(','))
    
    if case_num == 1:
        baskets = pairs.map(lambda x: (x[0], x[1])) \
                       .groupByKey() \
                       .map(lambda x: (x[0], set(x[1]))) \
                       .map(lambda x: x[1])
    else:
        baskets = pairs.map(lambda x: (x[1], x[0])) \
                       .groupByKey() \
                       .map(lambda x: (x[0], set(x[1]))) \
                       .map(lambda x: x[1])
    
    num_partitions = baskets.getNumPartitions()
    
    partition_support = support / num_partitions
    
    candidates = baskets.mapPartitions(lambda part: apriori(part, partition_support)) \
                        .distinct() \
                        .collect()
    
    candidates_by_size = {}
    for candidate in candidates:
        size = len(candidate)
        if size not in candidates_by_size:
            candidates_by_size[size] = []
        candidates_by_size[size].append(candidate)
    
    for size in candidates_by_size:
        candidates_by_size[size].sort()
    
    candidates_set = set(candidates)
    
    def count_candidates(baskets):
        basket_list = list(baskets)
        counts = {}
        for basket in basket_list:
            basket_set = set(basket)
            for candidate in candidates_set:
                if set(candidate).issubset(basket_set):
                    counts[candidate] = counts.get(candidate, 0) + 1
        return counts.items()
    
    candidate_counts = baskets.mapPartitions(count_candidates) \
                              .reduceByKey(lambda a, b: a + b) \
                              .filter(lambda x: x[1] >= support) \
                              .map(lambda x: x[0]) \
                              .collect()
    
    frequent_by_size = {}
    for itemset in candidate_counts:
        size = len(itemset)
        if size not in frequent_by_size:
            frequent_by_size[size] = []
        frequent_by_size[size].append(itemset)
    
    for size in frequent_by_size:
        frequent_by_size[size].sort()
    
    with open(output_file, 'w') as f:
        f.write("Candidates:\n")
        for size in sorted(candidates_by_size.keys()):
            items = candidates_by_size[size]
            if size == 1:
                result = ','.join([f"('{item[0]}')" for item in items])
            else:
                result = ','.join([str(item) for item in items])
            f.write(result + "\n\n")
        
        f.write("Frequent Itemsets:\n")
        for size in sorted(frequent_by_size.keys()):
            items = frequent_by_size[size]
            if size == 1:
                result = ','.join([f"('{item[0]}')" for item in items])
            else:
                result = ','.join([str(item) for item in items])
            f.write(result + "\n\n")
    
    sc.stop()
    
    end_time = time.time()
    duration = int(end_time - start_time)
    print(f"Duration: {duration}")
