# -*- coding: utf-8 -*-
"""
It seems the file "explore_panc_ocm.py" does not run correctly because of the memory.
This file tries to find the end of run with less memory.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

out_list = []

'''
Note: file name run1, run2 and run3 means: before, shortly after and 10 minutes after water, respectively.
      This run name is confusing because we also use only three OCM out of four in this study. 
'''

# Jihun Local
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy")  # Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy")  # After water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run3.npy")  # 10min After water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run3.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run3.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run3.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run3.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run3.npy")

# these are where the runs end in each OCM file
# rep_list = [8769, 8769, 8769, 8767, 8767, 8767, 7506, 7506, 7506]
num_subject = 6  # This number has to be the number of total run (number of subjects * number of runs)
rep_list = [8196, 8196, 8196, 8192, 8192, 8192, 6932, 6932, 6932, 3690, 3690, 3690, 3401, 3401, 3401, 3690, 3690, 3690]  # 3124 3401 3200

# Define fidx here (instead of for loop)
fidx = 15
in_filename = out_list[fidx]
ocm = np.load(in_filename)

# crop data
ocm = ocm[250:700, :]  # Original code.

# s=# of samples per trace
# t=# of total traces
s, t = np.shape(ocm)

# filter the data
hptr = np.ones([s, t])  # high pass filter
lptr = np.ones([s, t])  # low pass filter
lptra = np.ones([s, t])
f1 = np.ones([5])
f2 = np.ones([10])
for p in range(0, t):
    # high pass then low pass filter the data
    tr1 = ocm[:, p]
    hptr[:, p] = np.convolve(tr1, [1, -1], 'same')
    tr2 = hptr[:, p]
    lptra[:, p] = np.convolve(tr2, f1, 'same')
    tr3 = lptra[:, p]
    # square and envelope detect
    lptr[:, p] = np.convolve(np.sqrt(np.square(tr3)), f2, 'same')
    # normalize
    lptr[:, p] = np.divide(lptr[:, p], np.max(lptr[:, p]))

ocm = lptr

b = np.linspace(0, t - 1, t)
'''
b0 = np.mod(b, 4) == 0
ocm0 = ocm[:, b0]
'''
b1 = np.mod(b, 4) == 1
ocm1 = ocm[:, b1]
'''
b2 = np.mod(b, 4) == 2
ocm2 = ocm[:, b2]

b3 = np.mod(b, 4) == 3
ocm3 = ocm[:, b3]
'''

# =============================================================================

# plot the data
'''
fig, ax = plt.subplots()
ax.imshow(np.abs(ocm0[100:600, :]), aspect="auto", vmin=0, vmax=1)
'''
fig, ax = plt.subplots()
ax.imshow(np.abs(ocm1[100:600, :]), aspect="auto", vmin=0, vmax=1)
'''
fig, ax = plt.subplots()
ax.imshow(np.abs(ocm2[100:600, :]), aspect="auto", vmin=0, vmax=1)
fig, ax = plt.subplots()
ax.imshow(np.abs(ocm3[100:600, :]), aspect="auto", vmin=0, vmax=1)
'''
# =============================================================================
