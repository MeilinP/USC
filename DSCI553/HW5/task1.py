import sys
import binascii
from blackbox import BlackBox

def myhashs(s):
    result = []
    user_int = int(binascii.hexlify(s.encode('utf8')), 16)
    hash_params = [
        (23, 97), (31, 103), (37, 107), (41, 113), (43, 127),
        (47, 131), (53, 137), (59, 139), (61, 149), (67, 151),
        (71, 157), (73, 163), (79, 167), (83, 173), (89, 179)
    ]
    for a, b in hash_params:
        hash_val = (a * user_int + b) % 69997
        result.append(hash_val)
    return result

def calculate_fpr(bloom_filter, previous_users, stream_users):
    false_positives = 0
    for user in stream_users:
        if user not in previous_users:
            hash_values = myhashs(user)
            if all(bloom_filter[h] == 1 for h in hash_values):
                false_positives += 1
    
    new_users = len([u for u in stream_users if u not in previous_users])
    if new_users == 0:
        return 0.0
    return false_positives / new_users

if __name__ == "__main__":
    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]
    
    bloom_filter = [0] * 69997
    previous_users = set()
    
    bx = BlackBox()
    
    with open(output_file, 'w') as f:
        f.write("Time,FPR\n")
        
        for i in range(num_of_asks):
            stream_users = bx.ask(input_file, stream_size)
            
            fpr = calculate_fpr(bloom_filter, previous_users, stream_users)
            f.write(f"{i},{fpr}\n")
            
            for user in stream_users:
                hash_values = myhashs(user)
                for h in hash_values:
                    bloom_filter[h] = 1
                previous_users.add(user)
