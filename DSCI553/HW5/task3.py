import sys
import random
from blackbox import BlackBox

if __name__ == "__main__":
    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]
    
    random.seed(553)
    
    bx = BlackBox()
    
    reservoir = []
    seqnum = 0
    
    with open(output_file, 'w') as f:
        f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
        
        for ask_num in range(num_of_asks):
            stream_users = bx.ask(input_file, stream_size)
            
            for user in stream_users:
                seqnum += 1
                
                if len(reservoir) < 100:
                    reservoir.append(user)
                else:
                    prob = random.random()
                    if prob < 100 / seqnum:
                        replace_idx = random.randint(0, 99)
                        reservoir[replace_idx] = user
                
                if seqnum % 100 == 0:
                    f.write(f"{seqnum},{reservoir[0]},{reservoir[20]},{reservoir[40]},{reservoir[60]},{reservoir[80]}\n")
