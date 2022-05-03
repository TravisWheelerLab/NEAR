import matplotlib.pyplot as plt
import pandas as pd

thresholds = [1, 2, 3, 5, 10, 20, 100, 200]
pid20 = [0.564, 0.632, 0.663, 0.707, 0.757, 0.804, 0.905, 0.950]
ds20 = [(1 - i / 559) for i in thresholds]

pid25 = [0.688, 0.734, 0.756, 0.781, 0.815, 0.843, 0.918, 0.948]
ds25 = [(1 - i / 962) for i in thresholds]

pid30 = [0.803, 0.836, 0.852, 0.869, 0.888, 0.907, 0.950, 0.966]
ds30 = [(1 - i / 1191) for i in thresholds]

pid35 = [0.848, 0.871, 0.883, 0.896, 0.910, 0.924, 0.955, 0.969]
ds35 = [(1 - i / 1292) for i in thresholds]

pid40 = [0.916, 0.933, 0.940, 0.947, 0.956, 0.963, 0.979, 0.986]
ds40 = [(1 - i / 1309) for i in thresholds]

pid50 = [0.957, 0.964, 0.969, 0.974, 0.979, 0.983, 0.989, 0.993]
ds50 = [(1 - i / 1185) for i in thresholds]

pids = [pid20, pid25, pid30, pid35, pid40, pid50]
dss = [ds20, ds25, ds30, ds35, ds40, ds50]

top1 = [p[0] for p in pids]
top5 = [p[3] for p in pids]
top20 = [p[5] for p in pids]
top200 = [p[-1] for p in pids]

dstop1 = [p[0] for p in dss]
dstop5 = [p[3] for p in dss]
dstop20 = [p[5] for p in dss]
dstop200 = [p[-1] for p in dss]

pid_vals = [20, 25, 30, 35, 40, 50]
fig, ax = plt.subplots()

ax.plot(pid_vals, top1, ".-", label=f"top1 - min filtration rate: {min(dstop1):.3f}")
ax.plot(pid_vals, top5, ".-", label=f"top5 - min filtration rate: {min(dstop5):.3f}")
ax.plot(pid_vals, top20, ".-", label=f"top20 - min filtration rate: {min(dstop20):.3f}")
ax.plot(
    pid_vals, top200, ".-", label=f"top200 - min filtration rate: {min(dstop200):.3f}"
)
ax.legend()
ax.invert_xaxis()

ax.set_xlabel("max percent ID between target and query")
ax.set_ylabel("fraction of true homologs detected")
ax.set_title("prefilter performance at different %ids")

ax.plot([20, 50], [0.5, 0.5], "b--")
ax.set_ylim(0.0, 1.0)
ax.set_xlim(50, 0)
plt.show()
