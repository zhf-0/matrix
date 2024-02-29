import argparse
import os
import shutil
import scipy
import ssgetpy

from Parameters import Parameter
from Utility import WriteMatAndVec

def GenerateMat(ss_id, mat_type=None, mat_path=None):
    download_dir = os.path.join(os.path.dirname(__file__),'suitesparse')
    res = ssgetpy.fetch(ss_id,'MM',download_dir)
    tmp_mat_dir = os.path.join(download_dir,res[0].name)
    tmp_mat_path = os.path.join(tmp_mat_dir,res[0].name+'.mtx')
    A = scipy.io.mmread(tmp_mat_path)
    shutil.rmtree(tmp_mat_dir)

    WriteMatAndVec(A,None,mat_type,mat_path,False)

class Para(Parameter):
    def __init__(self):
        super().__init__()

    def AddParas(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ss_id', default='1', type=int, dest='ss_id')
    parser.add_argument('--mat_type', default=None, type=str, dest='mat_type')
    parser.add_argument('--mat_path', default=None, type=str, dest='mat_path')

    args = parser.parse_args()
    ss_id = args.ss_id
    mat_type = args.mat_type
    mat_path = args.mat_path
    GenerateMat(ss_id,mat_type,mat_path)
