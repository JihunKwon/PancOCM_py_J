import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.close('all')

out_list = []

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
num_subject = 6  # This number has to be the number of total run (number of subjects * number of runs)
rep_list = [8196, 8196, 8196, 8192, 8192, 8192, 6932, 6932, 6932, 3690, 3690, 3690, 3401, 3401, 3401, 3690, 3690, 3690]

# these store data for each transducer, 5 breath holds, 15 runs
t0 = np.zeros([300, 5, np.size(rep_list)])
t1 = np.zeros([300, 5, np.size(rep_list)])
t2 = np.zeros([300, 5, np.size(rep_list)])
t3 = np.zeros([300, 5, np.size(rep_list)])

# stores mean squared difference
d0 = np.zeros([5, np.size(rep_list)])
d1 = np.zeros([5, np.size(rep_list)])
d2 = np.zeros([5, np.size(rep_list)])
d3 = np.zeros([5, np.size(rep_list)])

for fidx in range(0, np.size(rep_list)):
    # fidx = 16
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    # crop data
    ocm = ocm[300:600, :]  # Original code.

    # s=# of samples per trace
    # t=# of total traces
    s, t = np.shape(ocm)

    # ============================Moving Average=====================================
    # filter the data
    offset = np.ones([s, t])  # offset correction
    hptr = np.ones([s, t])  # high pass filter
    lptr = np.ones([s, t])  # low pass filter
    lptra = np.ones([s, t])
    lptr_norm = np.ones([s, t])  # Normalized
    f1 = np.ones([5])
    f2 = np.ones([10])
    max_p = 0

    # My variables
    offset_my = np.ones([s, t])  # offset correction
    lptr_my = np.ones([s, t])  # low pass filter
    lptr_env_my = np.ones([s, t])  # low pass filter
    f1_my1 = np.ones([5])
    f2_my = np.ones([10])  # Envelop
    for p in range(0, t):

        # high pass then low pass filter the data
        tr1 = ocm[:, p]
        
        offset = signal.detrend(tr1)
        #hptr[:, p] = np.convolve(offset, [1, -1], 'same')
        #tr2 = hptr[:, p]
        lptra[:, p] = np.convolve(offset, f1, 'same')
        tr3 = lptra[:, p]
        # square and envelope detect
        lptr[:, p] = np.convolve(np.sqrt(np.square(tr3)), f2, 'same')
        # normalize
        max_temp = np.max(lptr[:, p])
        if max_p < max_temp:
            max_p = max_temp

        lptr_norm[:, p] = np.divide(lptr[:, p], np.max(lptr[:, p]))

    ocm = lptr_norm

    b = np.linspace(0, t - 1, t)
    b0 = np.mod(b, 4) == 0
    ocm0 = ocm[:, b0]
    b1 = np.mod(b, 4) == 1
    ocm1 = ocm[:, b1]
    b2 = np.mod(b, 4) == 2
    ocm2 = ocm[:, b2]

    ##################### Start Signal Processing here ##########################
    s, c0 = np.shape(ocm0)
    s, c1 = np.shape(ocm1)
    s, c2 = np.shape(ocm2)

    ocm0d = np.zeros([s, c0]) #Offset Correction (detrend)
    ocm1d = np.zeros([s, c1])
    ocm2d = np.zeros([s, c2])
    ocm0mov = np.zeros([s, c0]) #moving average
    ocm1mov = np.zeros([s, c1])
    ocm2mov = np.zeros([s, c2])

    #Moving average
    N = 5
    for p in range(c0):
        #ocm0d[:,p] = signal.detrend(ocm0[:,p])
        #ocm1d[:,p] = signal.detrend(ocm1[:,p])
        #ocm2d[:,p] = signal.detrend(ocm2[:,p])

        ocm0mov[:,p] = np.convolve(ocm0[:,p], np.ones(N)/N, mode='same')
        ocm1mov[:,p] = np.convolve(ocm1[:,p], np.ones(N)/N, mode='same')
        ocm2mov[:,p] = np.convolve(ocm2[:,p], np.ones(N)/N, mode='same')

    # compute mean of the breath hold, there are 5 breath holds per run, 3 runs per subject
    ocm0m = np.ones([s, 5])
    ocm1m = np.ones([s, 5])
    ocm2m = np.ones([s, 5])

    for i in range(0, 5):  # Distribute ocm signal from end to start
        ocm0m[:, i] = np.mean(np.abs(ocm0mov[:, ocm0mov.shape[1] - rep_list[fidx] * (i + 1) - 1:ocm0mov.shape[1] - rep_list[fidx] * i - 1]), 1)
        ocm1m[:, i] = np.mean(np.abs(ocm1mov[:, ocm1mov.shape[1] - rep_list[fidx] * (i + 1) - 1:ocm1mov.shape[1] - rep_list[fidx] * i - 1]), 1)
        ocm2m[:, i] = np.mean(np.abs(ocm2mov[:, ocm2mov.shape[1] - rep_list[fidx] * (i + 1) - 1:ocm2mov.shape[1] - rep_list[fidx] * i - 1]), 1)

    # collect all the data so far
    t0[:, :, fidx] = ocm0m
    t1[:, :, fidx] = ocm1m
    t2[:, :, fidx] = ocm2m

sample_rate_MHz = 10
us_per_sample = 1 / sample_rate_MHz

# in cm
little_t = np.linspace(2.3, 6.2, s)
for fidx in range(0, np.size(rep_list)):
    if (fidx % 3) == 2:  # If three sets (before, after, 10min) finishes:
        depth = np.linspace(0, s - 1, s)

        fig = plt.figure(figsize=(20, 16))

        ax0 = fig.add_subplot(331)
        a0 = ax0.plot(depth, t0[:, 0, fidx - 2], label="line 1")
        a1 = ax0.plot(depth, t0[:, 1, fidx - 2], label="line 2")
        a2 = ax0.plot(depth, t0[:, 2, fidx - 2], label="line 3")
        a3 = ax0.plot(depth, t0[:, 3, fidx - 2], label="line 4")
        a4 = ax0.plot(depth, t0[:, 4, fidx - 2], label="line 5")
        ax0.set_title('OCM0, Before')

        ax1 = fig.add_subplot(332)
        b0 = ax1.plot(depth, t1[:, 0, fidx - 2], label="line 1")
        b1 = ax1.plot(depth, t1[:, 1, fidx - 2], label="line 2")
        b2 = ax1.plot(depth, t1[:, 2, fidx - 2], label="line 3")
        b3 = ax1.plot(depth, t1[:, 3, fidx - 2], label="line 4")
        b4 = ax1.plot(depth, t1[:, 4, fidx - 2], label="line 5")
        ax1.set_title('OCM1, Before')

        ax2 = fig.add_subplot(333)
        c0 = ax2.plot(depth, t2[:, 0, fidx - 2], label="line 1")
        c1 = ax2.plot(depth, t2[:, 1, fidx - 2], label="line 2")
        c2 = ax2.plot(depth, t2[:, 2, fidx - 2], label="line 3")
        c3 = ax2.plot(depth, t2[:, 3, fidx - 2], label="line 4")
        c4 = ax2.plot(depth, t2[:, 4, fidx - 2], label="line 5")
        ax2.set_title('OCM2, Before')

        # After
        ax0 = fig.add_subplot(334)
        a0 = ax0.plot(depth, t0[:, 0, fidx - 1], label="line 6")
        a1 = ax0.plot(depth, t0[:, 1, fidx - 1], label="line 7")
        a2 = ax0.plot(depth, t0[:, 2, fidx - 1], label="line 8")
        a3 = ax0.plot(depth, t0[:, 3, fidx - 1], label="line 9")
        a4 = ax0.plot(depth, t0[:, 4, fidx - 1], label="line 10")
        ax0.set_title('OCM0, After')

        ax1 = fig.add_subplot(335)
        b0 = ax1.plot(depth, t1[:, 0, fidx - 1], label="line 6")
        b1 = ax1.plot(depth, t1[:, 1, fidx - 1], label="line 7")
        b2 = ax1.plot(depth, t1[:, 2, fidx - 1], label="line 8")
        b3 = ax1.plot(depth, t1[:, 3, fidx - 1], label="line 9")
        b4 = ax1.plot(depth, t1[:, 4, fidx - 1], label="line 10")
        ax1.set_title('OCM1, After')

        ax2 = fig.add_subplot(336)
        c0 = ax2.plot(depth, t2[:, 0, fidx - 1], label="line 6")
        c1 = ax2.plot(depth, t2[:, 1, fidx - 1], label="line 7")
        c2 = ax2.plot(depth, t2[:, 2, fidx - 1], label="line 8")
        c3 = ax2.plot(depth, t2[:, 3, fidx - 1], label="line 9")
        c4 = ax2.plot(depth, t2[:, 4, fidx - 1], label="line 10")
        ax2.set_title('OCM2, After')

        # 10 min
        ax0 = fig.add_subplot(337)
        a0 = ax0.plot(depth, t0[:, 0, fidx], label="line 11")
        a1 = ax0.plot(depth, t0[:, 1, fidx], label="line 12")
        a2 = ax0.plot(depth, t0[:, 2, fidx], label="line 13")
        a3 = ax0.plot(depth, t0[:, 3, fidx], label="line 14")
        a4 = ax0.plot(depth, t0[:, 4, fidx], label="line 15")
        ax0.set_title('OCM0, 10min')

        ax1 = fig.add_subplot(338)
        b0 = ax1.plot(depth, t1[:, 0, fidx], label="line 11")
        b1 = ax1.plot(depth, t1[:, 1, fidx], label="line 12")
        b2 = ax1.plot(depth, t1[:, 2, fidx], label="line 13")
        b3 = ax1.plot(depth, t1[:, 3, fidx], label="line 14")
        b4 = ax1.plot(depth, t1[:, 4, fidx], label="line 15")
        ax1.set_title('OCM1, 10min')

        ax2 = fig.add_subplot(339)
        c0 = ax2.plot(depth, t2[:, 0, fidx], label="line 11")
        c1 = ax2.plot(depth, t2[:, 1, fidx], label="line 12")
        c2 = ax2.plot(depth, t2[:, 2, fidx], label="line 13")
        c3 = ax2.plot(depth, t2[:, 3, fidx], label="line 14")
        c4 = ax2.plot(depth, t2[:, 4, fidx], label="line 15")
        ax2.set_title('OCM2, 10min')

        fig.tight_layout()
        plt.savefig('t012_{fidx}_MA.png'.format(fidx=fidx))

'''Method 2: rm0 is only from fidx=0,3,6,9, which means signal from first-run (before water) is used for subtraction
'''

# loop through subjects
for sub in range(0, num_subject):
    # mean for this run across all breath holds
    rm0 = np.mean(t0[:, :, sub * 3], 1)
    rm1 = np.mean(t1[:, :, sub * 3], 1)
    rm2 = np.mean(t2[:, :, sub * 3], 1)
    # loop through runs (before water, after water, 10min after water)
    for run in range(0, 3):
        fidx = run + sub * 3  # file number
        # loop through breath holds
        for bh in range(0, 5):
            d0[bh, fidx] = np.mean(np.square(np.subtract(rm0, t0[:, bh, fidx])), 0)
            d1[bh, fidx] = np.mean(np.square(np.subtract(rm1, t1[:, bh, fidx])), 0)
            d2[bh, fidx] = np.mean(np.square(np.subtract(rm2, t2[:, bh, fidx])), 0)

out_txt = []

# Jihun Local
out_txt.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\transducer0_ma.txt")
out_txt.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\transducer1_ma.txt")
out_txt.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\transducer2_ma.txt")

np.savetxt(out_txt[0], d0, fmt='%0.08f', delimiter=' ', newline='\n')
np.savetxt(out_txt[1], d1, fmt='%0.08f', delimiter=' ', newline='\n')
np.savetxt(out_txt[2], d2, fmt='%0.08f', delimiter=' ', newline='\n')
