from lzc import lzc
import scipy.io as sio
import numpy as np
import os

# Path to data
data_path = '/Volumes/dataSets/restEEGHealthySubjects/rawData/'

# The below code will load each file from the folder data_path, and run a lzc on it.
# The results will be saved in the folder data_path/lzc_results
ext_npy = ".npy"
ext_mat = ".mat"
lz_all_npy = []
for filename in os.listdir(data_path):
    if filename.endswith(ext_npy):
        f = os.path.join(data_path, filename)
        file_npy = np.load(f)
        file_npy = file_npy.transpose(1,0,2).reshape(46, -1)
        LZc_single_subject = lzc.LZc(file_npy)
        lz_all_npy.append(LZc_single_subject)

    elif filename.endswith(ext_mat):
        f = os.path.join(data_path, filename)
        file_mat = sio.loadmat(f)
        file_mat = file_mat['EEG']
        LZc_single_subject = lzc.LZc(file_mat)
        lz_all_npy.append(LZc_single_subject)
    


results_path = 'preprocessedData/lzc_results/'
np.savetxt(results_path + "lzc_all_raw.csv", lz_all_npy, delimiter="," )
