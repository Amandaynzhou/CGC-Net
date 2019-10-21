import sys
sys.path.append('../')
import numpy as np
import os

# from setting import DataSetting
from common.utils import mkdirs
from multiprocessing.pool import Pool

def euc_dist(name):
    # f = os.path.join(setting.root, 'proto', 'coordinate', name)
    arr = np.load(name)
    arr_x = (arr[:,0,np.newaxis].T - arr[:,0,np.newaxis])**2
    arr_y = (arr[:,1,np.newaxis].T - arr[:,1,np.newaxis])**2
    arr = np.sqrt(arr_x + arr_y)

    np.save(name.replace('coordinate', 'distance'), arr, )
    return 0

if __name__ == '__main__':
    # calculate point to point distance and save to proto/distance
    # setting = DataSetting()

    # for name in  ['tialab','extra']:
    #     for split in ['train', 'valid']:
            name = 'extra'
            split = 'valid'
            pr = Pool(processes=8)
            savepath = os.path.join('/media/amanda/HDD2T_1/warwick-research/data', 'ICIAR', 'distance', split)
            files = os.listdir(os.path.join('/media/amanda/HDD2T_1/warwick-research/data', 'ICIAR', 'coordinate', split))
            mkdirs(savepath)
            files = [os.path.join('/media/amanda/HDD2T_1/warwick-research/data', 'ICIAR', 'coordinate',split,f) for f in files]
            arr = pr.map(euc_dist, files)
            pr.close()
            pr.join()