import numpy as np
import sys
import struct
from scipy.stats import norm, expon, laplace
def log1mexp(x: np.ndarray) -> np.ndarray:
    """
    Compute log(1 - exp(x)) in a numerically stable way for x <= 0.
    """
    x = x.astype(np.float64)
    thresh = -0.6931471805599453  # ln(0.5)
    out = np.empty_like(x)

    # 1) x == 0 => log(1 - 1) = -inf
    zero_mask = (x == 0)
    out[zero_mask] = -np.inf

    # 2) x <= ln(0.5): safe to use log1p
    mask1 = (x <= thresh) & ~zero_mask
    out[mask1] = np.log1p(-np.exp(x[mask1]))

    # 3) ln(0.5) < x < 0: use expm1
    mask2 = (x > thresh) & (x < 0)
    out[mask2] = np.log(-np.expm1(x[mask2]))

    return out

with np.load('/home/danielolson/gpd_dist_0.001.npz') as data:
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

angle_divergence = np.load('/home/danielolson/expected_angle_deviation.npy')

with np.load(sys.argv[1], allow_pickle=True) as data:
    scores=data['scores'][:100000]
    queries=data['qids'][:100000]
    targets=data['tids'][:100000]
    tnames=list(data['tnames'])
    qnames=list(data['qnames'])


qpos = queries & 0x7F
tpos = targets & 0x7F

scores = scores.astype(np.float64)


query_lengths = np.load(sys.argv[2])
target_lengths = np.load(sys.argv[3])

sys.stdout.buffer.write(loc.tobytes())
sys.stdout.buffer.write(scale.tobytes())
sys.stdout.buffer.write(shape.tobytes())

sys.stdout.buffer.write(angle_divergence.tobytes())
sys.stdout.buffer.write(struct.pack('Q', 0))

queries = queries[:,0]

query_seq_names = ('\0'.join(qnames) + '\0').encode('utf-8')
target_seq_names = '\0'.join(tnames).encode('utf-8')


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