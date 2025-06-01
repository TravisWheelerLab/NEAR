import numpy as np
import sys
import struct


with np.load(sys.argv[1]) as data:
    rloc = data['threshold']
    rscale =  data['scale']
    rshape = data['xi']

    loc = []
    scale = []
    shape = []
    for i in range(128):
        if i > 64:
            i = i + 1
        loc.append([])
        scale.append([])
        shape.append([])
        for j in range(128):
            if j > 64:
                j = j + 1
            loc[-1].append(rloc[i,j])
            scale[-1].append(rscale[i,j])
            shape[-1].append(rshape[i,j])

loc = np.array(loc).astype(np.float64)
scale = np.array(scale).astype(np.float64)
shape = np.array(shape).astype(np.float64)

angle_divergence = np.load(sys.argv[2]).astype(np.float64)

with np.load(sys.argv[3], allow_pickle=True) as data:
    scores=data['scores']
    queries=data['qids']
    targets=data['tids']
    tnames=list(data['tnames'])
    qnames=list(data['qnames'])

scores = scores.astype(np.float64)


query_lengths = np.load(sys.argv[4])
target_lengths = np.load(sys.argv[5])

queries = queries[:,0]

query_seq_names = ('\0'.join(qnames) + '\0').encode('utf-8')
target_seq_names = ('\0'.join(tnames) + '\0').encode('utf-8')


#print(type(loc), loc.dtype, loc.size, loc.shape)
#print(type(scale), scale.dtype, scale.size, scale.shape)
#print(type(shape), shape.dtype, shape.size, shape.shape)
#print("--")
#print(type(angle_divergence), angle_divergence.dtype, angle_divergence.size, angle_divergence.shape)
#print("--")
#print(len(qnames))
#print(len(query_seq_names))
#print(len(query_lengths))
#print("--")
#print(len(tnames))
#print(len(target_seq_names))
#print(len(target_lengths))
#print("--")
#print(queries.shape, queries.dtype, queries.size)
#print(targets.shape, targets.dtype, targets.size)
#print(scores.shape, scores.dtype, scores.size)

#exit(0)


sys.stdout.buffer.write(loc.flatten().tobytes())        # send loc
sys.stdout.buffer.write(scale.flatten().tobytes())      # send scale
sys.stdout.buffer.write(shape.flatten().tobytes())      # send shape

sys.stdout.buffer.write(angle_divergence.tobytes())     # send angle divergence (127 double)
sys.stdout.buffer.write(struct.pack('Q', int(0))) # send 1 extra double

sys.stdout.buffer.write(struct.pack('Q', len(qnames)))
sys.stdout.buffer.write(struct.pack('Q', len(query_seq_names)))
sys.stdout.buffer.write(query_seq_names)
sys.stdout.buffer.write(query_lengths.tobytes())

sys.stdout.buffer.write(struct.pack('Q', len(tnames)))
sys.stdout.buffer.write(struct.pack('Q', len(target_seq_names)))
sys.stdout.buffer.write(target_seq_names)
sys.stdout.buffer.write(target_lengths.tobytes())


sys.stdout.buffer.write(struct.pack('Q', len(queries)))
sys.stdout.buffer.write(queries.tobytes())
sys.stdout.buffer.write(targets.tobytes())
sys.stdout.buffer.write(scores.tobytes())