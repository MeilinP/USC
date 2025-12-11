import sys
import binascii
from blackbox import BlackBox

def myhashs(s):
    result = []
    user_int = int(binascii.hexlify(s.encode('utf8')), 16)
    
    hash_params = []
    a_values = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 
                73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 
                139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                199, 211, 223, 227, 229, 233, 239, 241, 251, 257]
    b_values = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 
                67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 
                131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
                193, 197, 199, 211, 223, 227, 229, 233, 239, 241]
    
    for i in range(50):
        a = a_values[i]
        b = b_values[i]
        hash_val = (a * user_int + b) % 10000000
        result.append(hash_val)
    
    return result

def trailing_zeros(num):
    if num == 0:
        return 0
    binary = bin(num)[2:]
    return len(binary) - len(binary.rstrip('0'))

def estimate_unique_users(hash_results):
    max_trailing_zeros = [0] * len(hash_results[0])
    
    for hash_vals in hash_results:
        for i, h in enumerate(hash_vals):
            tz = trailing_zeros(h)
            max_trailing_zeros[i] = max(max_trailing_zeros[i], tz)
    
    estimates = [2 ** r for r in max_trailing_zeros]
    
    num_groups = 5
    group_size = len(estimates) // num_groups
    group_averages = []
    
    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        group_avg = sum(estimates[start:end]) / group_size
        group_averages.append(group_avg)
    
    group_averages.sort()
    median = group_averages[len(group_averages) // 2]
    
    return int(median)

if __name__ == "__main__":
    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]
    
    bx = BlackBox()
    
    with open(output_file, 'w') as f:
        f.write("Time,Ground Truth,Estimation\n")
        
        for i in range(num_of_asks):
            stream_users = bx.ask(input_file, stream_size)
            
            unique_users = len(set(stream_users))
            
            hash_results = []
            for user in stream_users:
                hash_results.append(myhashs(user))
            
            estimation = estimate_unique_users(hash_results)
            
            f.write(f"{i},{unique_users},{estimation}\n")