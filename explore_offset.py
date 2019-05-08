import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

out_list = []
#Jihun Local
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy") #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy") #After water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run3.npy") #10min After water
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
num_subject = 6
rep_list = [8196, 8196, 8196, 8192, 8192, 8192, 6932, 6932, 6932, 3690, 3690, 3690, 3401, 3401, 3401, 3690, 3690, 3690]
#rep_list = [3690, 3690, 3690]

# these store data for each transducer, 5 breath holds, 15 runs
t0 = np.zeros([700,5,np.size(rep_list)])
t1 = np.zeros([700,5,np.size(rep_list)])
t2 = np.zeros([700,5,np.size(rep_list)])
t3 = np.zeros([700,5,np.size(rep_list)])

#stores mean squared difference
d0 = np.zeros([5,np.size(rep_list)])
d1 = np.zeros([5,np.size(rep_list)])
d2 = np.zeros([5,np.size(rep_list)])
d3 = np.zeros([5,np.size(rep_list)])

for fidx in range(0,np.size(rep_list)):
    #fidx = 2
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    print("Before crop ocm")
    print(np.shape(ocm))
    #crop data
    ocm = ocm[0:700,:]

    #s=# of samples per trace
    #t=# of total traces
    s, t = np.shape(ocm)
    print("s: ")
    print(s)
    print("t: ")
    print(t)
    print(np.shape(ocm[:,:]))

    b = np.linspace(0,t-1,t)
    b0 = np.mod(b,4)==0
    ocm0 = ocm[:,b0]
    b1 = np.mod(b,4)==1
    ocm1 = ocm[:,b1]
    b2 = np.mod(b,4)==2
    ocm2 = ocm[:,b2]
    b3 = np.mod(b,4)==3
    ocm3 = ocm[:,b3]

    s, c0 = np.shape(ocm0)
    s, c1 = np.shape(ocm1)
    s, c2 = np.shape(ocm2)
    s, c3 = np.shape(ocm3)

    #Offset correction
    ocm0d = signal.detrend(ocm0)
    ocm1d = signal.detrend(ocm1)
    ocm2d = signal.detrend(ocm2)
    ocm3d = signal.detrend(ocm3)

    np.shape(ocm0d)
    ocm0mov = np.zeros([s, c0])

    #Moving average
    N = 8
    for col in range(c0):
        #ocm0mov[x,:c0-N+1] = np.convolve(ocm0d[x,:], np.ones(N)/N, mode='same')
        ocm0mov[:,col] = np.convolve(ocm0d[:,col], np.ones(N)/N, mode='same')

    # Compare detrend and moving average
    x = np.linspace(0, s - 1, s)
    fig = plt.figure(figsize=(12, 8))

    #ax1 = fig.add_subplot(211)
    #a1 = ax1.plot(x, ocm0d[:, 0])
    #ax1.set_title('Raw, OCM0')

    ax2 = fig.add_subplot(211)
    a2, a3 = ax2.plot(x, ocm0d[:, 0], x, ocm0mov[:, 0])
    ax2.set_title('Moving Average, ')
    # ax2.set_ylim(-100, 100)

    fig.tight_layout()
    plt.savefig('trace_{fidx}_ma.png'.format(fidx=fidx))