import numpy as np
import math
import time
def pull_data(dir_name, file_name, separator=';'):
    f = open(dir_name + '/' + file_name + '.csv')
    xs = []
    ys = []
    zs = []
    rs = []
    timestamps = []
    f.readline() # ignore separator declaration
    f.readline() # ignore header
    for line in f:
        value = line.split(separator)
        if len(value) > 3:
            t=value[3].strip()
            timestamps.append(t)
            x = float(value[0])
            y = float(value[1])
            z = float(value[2])
            r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            rs.append(r)
    return np.array(xs), np.array(ys), np.array(zs), np.array(rs), np.array(timestamps)