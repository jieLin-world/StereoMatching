import numpy as np
import re 
file = open('/mnt/cephfs/dataset/stereo_matching/sceneflow/disparity/TRAIN/C/0457/left/0014.pfm', 'rb')
header = file.readline().decode('utf-8').rstrip()
if header == 'PF':
    color = True
elif header == 'Pf':
    color = False
else:
    raise Exception('Not a PFM file.')

dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
if dim_match:
    width, height = map(int, dim_match.groups())
else:
    raise Exception('Malformed PFM header.')

scale = float(file.readline().rstrip())
if scale < 0:  # little-endian
    endian = '<'
    scale = -scale
else:
    endian = '>'  # big-endian

data = np.fromfile(file, endian + 'f')
shape = (height, width, 3) if color else (height, width)

data_reshape = np.reshape(data, shape)
data_flipud = np.flipud(data_reshape)

print(file)
print(header)
print(dim_match)
print(width, height)
print(scale)
print(data)
print(data_reshape)
print(data_flipud)
print(data)