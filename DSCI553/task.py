import sys
import numpy as np
from sklearn.cluster import KMeans

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            idx = int(parts[0])
            features = np.array([float(x) for x in parts[2:]])
            data.append((idx, features))
    return data

def mahalanobis_distance(point, cluster_stats, d):
    N, SUM, SUMSQ = cluster_stats
    if N == 0:
        return float('inf')
    
    centroid = SUM / N
    variance = SUMSQ / N - (SUM / N) ** 2
    variance = np.maximum(variance, 1e-10)
    
    diff = point - centroid
    distance = np.sqrt(np.sum(diff ** 2 / variance))
    return distance

def merge_clusters(cluster1, cluster2):
    N1, SUM1, SUMSQ1 = cluster1
    N2, SUM2, SUMSQ2 = cluster2
    return (N1 + N2, SUM1 + SUM2, SUMSQ1 + SUMSQ2)

def main(input_file, n_cluster, output_file):
    all_data = load_data(input_file)
    n_points = len(all_data)
    d = len(all_data[0][1])
    
    np.random.seed(42)
    indices = np.random.permutation(n_points)
    
    DS = {}
    CS = {}
    RS = {}
    CS_points = {}
    point_to_ds = {}
    
    intermediate_results = []
    chunk_size = n_points // 5
    
    first_chunk_indices = indices[:chunk_size]
    first_chunk = [all_data[i] for i in first_chunk_indices]
    first_chunk_features = np.array([item[1] for item in first_chunk])
    
    large_k = 5 * n_cluster
    kmeans_large = KMeans(n_clusters=large_k, random_state=42, n_init=10)
    labels_large = kmeans_large.fit_predict(first_chunk_features)
    
    label_counts = {}
    for label in labels_large:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    remaining_points = []
    remaining_features = []
    for i, (idx, features) in enumerate(first_chunk):
        if label_counts[labels_large[i]] == 1:
            RS[idx] = features
        else:
            remaining_points.append((idx, features))
            remaining_features.append(features)
    
    if len(remaining_features) > 0:
        remaining_features = np.array(remaining_features)
        kmeans_n = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
        labels_n = kmeans_n.fit_predict(remaining_features)
        
        for cluster_id in range(n_cluster):
            DS[cluster_id] = (0, np.zeros(d), np.zeros(d))
        
        for i, (idx, features) in enumerate(remaining_points):
            cluster_id = labels_n[i]
            point_to_ds[idx] = cluster_id
            N, SUM, SUMSQ = DS[cluster_id]
            DS[cluster_id] = (N + 1, SUM + features, SUMSQ + features ** 2)
    
    if len(RS) > 1:
        rs_indices = list(RS.keys())
        rs_features = np.array([RS[idx] for idx in rs_indices])
        
        k_rs = min(5 * n_cluster, len(rs_features))
        kmeans_rs = KMeans(n_clusters=k_rs, random_state=42, n_init=10)
        labels_rs = kmeans_rs.fit_predict(rs_features)
        
        rs_label_counts = {}
        for label in labels_rs:
            rs_label_counts[label] = rs_label_counts.get(label, 0) + 1
        
        new_RS = {}
        label_to_cs = {}
        next_cs_id = 0
        
        for i, idx in enumerate(rs_indices):
            if rs_label_counts[labels_rs[i]] == 1:
                new_RS[idx] = RS[idx]
            else:
                label = labels_rs[i]
                if label not in label_to_cs:
                    label_to_cs[label] = next_cs_id
                    CS[next_cs_id] = (0, np.zeros(d), np.zeros(d))
                    next_cs_id += 1
                
                cs_id = label_to_cs[label]
                CS_points[idx] = cs_id
                N, SUM, SUMSQ = CS[cs_id]
                CS[cs_id] = (N + 1, SUM + RS[idx], SUMSQ + RS[idx] ** 2)
        
        RS = new_RS
    
    num_discard = sum(DS[c][0] for c in DS)
    num_cs_clusters = len(CS)
    num_compression = sum(CS[c][0] for c in CS)
    num_rs = len(RS)
    intermediate_results.append((num_discard, num_cs_clusters, num_compression, num_rs))
    
    for round_num in range(1, 5):
        start_idx = round_num * chunk_size
        end_idx = min((round_num + 1) * chunk_size, n_points)
        chunk_indices = indices[start_idx:end_idx]
        
        new_points = [(all_data[i][0], all_data[i][1]) for i in chunk_indices]
        
        unassigned = []
        for idx, features in new_points:
            best_distance = float('inf')
            best_cluster = None
            
            for cluster_id in DS:
                dist = mahalanobis_distance(features, DS[cluster_id], d)
                if dist < best_distance:
                    best_distance = dist
                    best_cluster = cluster_id
            
            if best_distance < 2 * np.sqrt(d):
                point_to_ds[idx] = best_cluster
                N, SUM, SUMSQ = DS[best_cluster]
                DS[best_cluster] = (N + 1, SUM + features, SUMSQ + features ** 2)
            else:
                unassigned.append((idx, features))
        
        unassigned2 = []
        for idx, features in unassigned:
            best_distance = float('inf')
            best_cluster = None
            
            for cluster_id in CS:
                dist = mahalanobis_distance(features, CS[cluster_id], d)
                if dist < best_distance:
                    best_distance = dist
                    best_cluster = cluster_id
            
            if best_distance < 2 * np.sqrt(d):
                CS_points[idx] = best_cluster
                N, SUM, SUMSQ = CS[best_cluster]
                CS[best_cluster] = (N + 1, SUM + features, SUMSQ + features ** 2)
            else:
                unassigned2.append((idx, features))
        
        for idx, features in unassigned2:
            RS[idx] = features
        
        if len(RS) > 1:
            rs_indices = list(RS.keys())
            rs_features = np.array([RS[idx] for idx in rs_indices])
            
            k_rs = min(5 * n_cluster, len(rs_features))
            kmeans_rs = KMeans(n_clusters=k_rs, random_state=42, n_init=10)
            labels_rs = kmeans_rs.fit_predict(rs_features)
            
            rs_label_counts = {}
            for label in labels_rs:
                rs_label_counts[label] = rs_label_counts.get(label, 0) + 1
            
            new_RS = {}
            label_to_cs = {}
            next_cs_id = max(CS.keys()) + 1 if CS else 0
            
            for i, idx in enumerate(rs_indices):
                if rs_label_counts[labels_rs[i]] == 1:
                    new_RS[idx] = RS[idx]
                else:
                    label = labels_rs[i]
                    if label not in label_to_cs:
                        label_to_cs[label] = next_cs_id
                        CS[next_cs_id] = (0, np.zeros(d), np.zeros(d))
                        next_cs_id += 1
                    
                    cs_id = label_to_cs[label]
                    CS_points[idx] = cs_id
                    N, SUM, SUMSQ = CS[cs_id]
                    CS[cs_id] = (N + 1, SUM + RS[idx], SUMSQ + RS[idx] ** 2)
            
            RS = new_RS
        
        merged = set()
        cs_list = list(CS.keys())
        for i in range(len(cs_list)):
            for j in range(i + 1, len(cs_list)):
                c1, c2 = cs_list[i], cs_list[j]
                if c1 in merged or c2 in merged:
                    continue
                
                N1, SUM1, SUMSQ1 = CS[c1]
                N2, SUM2, SUMSQ2 = CS[c2]
                centroid1 = SUM1 / N1
                centroid2 = SUM2 / N2
                variance1 = SUMSQ1 / N1 - centroid1 ** 2
                variance1 = np.maximum(variance1, 1e-10)
                
                diff = centroid2 - centroid1
                dist = np.sqrt(np.sum(diff ** 2 / variance1))
                
                if dist < 2 * np.sqrt(d):
                    CS[c1] = merge_clusters(CS[c1], CS[c2])
                    for pt_idx in CS_points:
                        if CS_points[pt_idx] == c2:
                            CS_points[pt_idx] = c1
                    del CS[c2]
                    merged.add(c2)
        
        if round_num == 4:
            for cs_id in list(CS.keys()):
                best_distance = float('inf')
                best_ds = None
                
                N_cs, SUM_cs, SUMSQ_cs = CS[cs_id]
                centroid_cs = SUM_cs / N_cs
                
                for ds_id in DS:
                    dist = mahalanobis_distance(centroid_cs, DS[ds_id], d)
                    if dist < best_distance:
                        best_distance = dist
                        best_ds = ds_id
                
                if best_distance < 2 * np.sqrt(d):
                    DS[best_ds] = merge_clusters(DS[best_ds], CS[cs_id])
                    for pt_idx in CS_points:
                        if CS_points[pt_idx] == cs_id:
                            point_to_ds[pt_idx] = best_ds
                            del CS_points[pt_idx]
                    del CS[cs_id]
        
        num_discard = sum(DS[c][0] for c in DS)
        num_cs_clusters = len(CS)
        num_compression = sum(CS[c][0] for c in CS)
        num_rs = len(RS)
        intermediate_results.append((num_discard, num_cs_clusters, num_compression, num_rs))
    
    final_assignments = {}
    for idx in point_to_ds:
        final_assignments[idx] = point_to_ds[idx]
    for idx in RS:
        final_assignments[idx] = -1
    for idx in CS_points:
        final_assignments[idx] = -1
    
    with open(output_file, 'w') as f:
        f.write("The intermediate results:\n")
        for i, (nd, ncc, nc, nr) in enumerate(intermediate_results):
            f.write(f"Round {i+1}: {nd},{ncc},{nc},{nr}\n")
        f.write("\n")
        f.write("The clustering results:\n")
        for idx in sorted(final_assignments.keys()):
            f.write(f"{idx},{final_assignments[idx]}\n")

if __name__ == "__main__":
    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]
    
    main(input_file, n_cluster, output_file)