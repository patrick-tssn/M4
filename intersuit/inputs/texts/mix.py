import json
import random

# original
with open("llava-next-sub-10k.json") as jp:
    llava = json.load(jp)
# rev
with open("llava-next-sub-10k-rev.json") as jp:
    rev = json.load(jp)
# noise 
with open("llava-next-sub-10k-noise.json") as jp:
    noise = json.load(jp)
# stop
with open("llava-next-sub-10k-stop.json") as jp:
    stop = json.load(jp)
    
# mix: all-40k
mix = llava + rev + noise + stop
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-40k-ORNS.json", "w") as jp:
    json.dump(mix, jp, indent=4)

# mix: special-30k
mix = rev + noise + stop
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-30k-RNS.json", "w") as jp:
    json.dump(mix, jp, indent=4)
    
# mix: special-10rev-5noise-5stop
random.seed(43)
rev_half = random.sample(rev, len(rev)//2) 
random.seed(43)
stop_half = random.sample(stop, len(stop)//2)
mix = rev_half + stop_half + noise
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-20k-RNS121.json", "w") as jp:
    json.dump(mix, jp, indent=4)
    
# mix: special-10rev-5noise-5stop
random.seed(43)
noise_half = random.sample(noise, len(noise)//2) 
random.seed(43)
stop_half = random.sample(stop, len(stop)//2)
mix = rev + stop_half + noise_half
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-20k-RNS211.json", "w") as jp:
    json.dump(mix, jp, indent=4)

# mix: special-10rev-5noise-5stop
random.seed(43)
rev_half = random.sample(rev, len(rev)//2) 
random.seed(43)
stop_half = random.sample(stop, len(stop)//2)
mix = rev_half + stop_half + noise + llava
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-30k-ORNS2121.json", "w") as jp:
    json.dump(mix, jp, indent=4)
    
    
# mix: special-10rev-5noise-5stop
random.seed(43)
noise_half = random.sample(noise, len(noise)//2) 
random.seed(43)
stop_half = random.sample(stop, len(stop)//2)
mix = stop_half + noise_half + llava
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-20k-ONS211.json", "w") as jp:
    json.dump(mix, jp, indent=4)


# mix: special-10rev-5noise-5stop
random.seed(43)
noise_half = random.sample(noise, len(noise)//2) 
random.seed(43)
stop_half = random.sample(stop, len(stop)//2)
random.seed(43)
rev_half = random.sample(rev, len(rev)//2) 
mix = stop_half + noise_half + llava + rev_half
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-20k-ORNS2111.json", "w") as jp:
    json.dump(mix, jp, indent=4)
    
    
# mix: special-10rev-5noise-5stop
random.seed(43)
noise_half = random.sample(noise, len(noise)//4) 
random.seed(43)
stop_half = random.sample(stop, len(stop)//4)
random.seed(43)
rev_half = random.sample(rev, len(rev)//4) 
random.seed(43)
ori_half = random.sample(llava, len(llava)//4) 
mix = stop_half + noise_half + ori_half + rev_half
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-10k-ORNS1111.json", "w") as jp:
    json.dump(mix, jp, indent=4)


# mix: special-10rev-5noise-5stop
random.seed(43)
noise_half = random.sample(noise, len(noise)//4) 
random.seed(43)
stop_half = random.sample(stop, len(stop)//4)
random.seed(43)
rev_half = random.sample(rev, len(rev)//2) 
random.seed(43)
ori_half = random.sample(llava, len(llava)//4) 
mix = stop_half + noise_half + rev_half
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-10k-RNS211.json", "w") as jp:
    json.dump(mix, jp, indent=4)
    
    
# mix: special-10rev-5noise-5stop
random.seed(43)
noise_half = random.sample(noise, len(noise)//2) 
random.seed(43)
stop_half = random.sample(stop, len(stop)//4)
random.seed(43)
rev_half = random.sample(rev, len(rev)//4) 
random.seed(43)
ori_half = random.sample(llava, len(llava)//4) 
mix = stop_half + noise_half + rev_half
random.seed(43)
random.shuffle(mix)
with open("llava-next-sub-10k-NRS211.json", "w") as jp:
    json.dump(mix, jp, indent=4)