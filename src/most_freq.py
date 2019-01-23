from collections import Counter
from functools import reduce
from time import time 


filename = 'data/truven_5w'
testfile = 'data/truven_2w_test'
#topk = 50
topk = 30 
t1 = time()
lines = open(filename, 'r').readlines()
f1 = lambda x:' '.join(x.rstrip().split(';'))
lines = list(map(f1, lines))
lines = ' '.join(lines)
clinical_variable_lst = [int(i) for i in lines.split()]
counter = Counter(clinical_variable_lst)

common_variable = counter.most_common(topk)
common_variable = [i[0] for i in common_variable]


lines = open(testfile, 'r').readlines()
f2 = lambda x:list(set([int(i) for i in x.rstrip().split(';')[-1].split()]))
lines = list(map(f2, lines))
all_code = reduce(lambda x,y:x+y, lines)
right_code = list(filter(lambda x: x in common_variable, all_code))

print('top {} recall: {}, cost {} seconds'.format(topk, len(right_code) / len(all_code), int(time() - t1)))





