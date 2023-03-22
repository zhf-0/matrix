import os
import sys
import scipy
from .Parameters import Parameter
from .Utility import WriteMatAndVec
import shutil
import ssgetpy

def GenerateMat(ss_id,ss_mat_path, mat_type=None, mat_path=None):
    if ss_mat_path is None:
        download_dir = os.path.join(os.path.dirname(__file__),'suitesparse')
        res = ssgetpy.fetch(ss_id,'MM',download_dir)
        tmp_mat_dir = os.path.join(download_dir,res[0].name)
        tmp_mat_path = os.path.join(tmp_mat_dir,res[0].name+'.mtx')
        A = scipy.io.mmread(tmp_mat_path)
        shutil.rmtree(tmp_mat_dir)
    else:
        if ss_mat_path.endswith('mtx'):
            A = scipy.io.mmread(ss_mat_path)
        else:
            print('other formats, matlab and RB, are not fully supported by scipy')
            print('details can be found in https://docs.scipy.org/doc/scipy/reference/io.html')
            sys.exit()

    WriteMatAndVec(A,None,mat_type,mat_path,False)

    row_num, col_num = A.shape
    nnz = A.nnz
    return row_num, col_num, nnz


class Para(Parameter):
    def __init__(self):
        super().__init__()

    def AddParas(self):
        pass
