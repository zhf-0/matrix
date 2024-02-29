import os
import sys
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import coo_matrix, csr_matrix 

def WriteCOO2TXT(mat_path, scicoo): # type(scicoo): scipy.sparse.coo_matrix 
    coo_i = scicoo.row
    coo_j = scicoo.col
    coo_val = scicoo.data
    row_num, col_num = scicoo.shape
    nnz = scicoo.nnz

    output = [None]*(nnz+1)
    template = "{} {} {} \n"
    output[0] = template.format(row_num,col_num,nnz)
    for i in range(nnz):
        output[i+1] = template.format(coo_i[i], coo_j[i], coo_val[i])

    with open(mat_path,'w') as f:
        f.writelines(output)
        

def ReadCOOFromTXT(mat_path):
    with open(mat_path,'r') as f:
        contents = f.readlines()
        
    len_con = len(contents)
    first = contents[0]
    list_val = first.strip().split()
    row_num = int(list_val[0])
    col_num = int(list_val[1])
    nnz = int(list_val[2])
    
    coo_i = np.zeros(nnz,dtype=np.int)
    coo_j = np.zeros(nnz,dtype=np.int)
    coo_val = np.zeros(nnz,dtype=np.float64)
    for i in range(1,len_con):
        tmplist = contents[i].split()
        coo_i[i-1] = int(tmplist[0])  
        coo_j[i-1] = int(tmplist[1])  
        coo_val[i-1] = float(tmplist[2])  

    scicoo = coo_matrix((coo_val, (coo_i, coo_j)), shape=(row_num, col_num))
    return scicoo


def WriteVec2TXT(rhs_path, rhs): # type(rhs): numpy 
    row_num = rhs.shape[0]

    output = [None]*(row_num+1)
    template = "{} \n"
    output[0] = template.format(row_num)
    for i in range(row_num):
        output[i+1] = template.format(rhs[i])

    with open(rhs_path,'w') as f:
        f.writelines(output)
        

def ReadVecFromTXT(rhs_path):
    with open(rhs_path,'r') as f:
        contents = f.readlines()
        
    len_con = len(contents)
    row_num = int(contents[0])
    
    rhs = np.zeros(row_num,dtype=np.float64)
    for i in range(1,len_con):
        rhs[i-1] = float(contents[i])  
        
    return rhs

def WriteMatAndVec(A,F,mat_type,mat_path,need_rhs):
    if isinstance(A,sparse.coo.coo_matrix):
        if mat_type is None:
            pass
        elif mat_type == 'SciCSR':
            csr_A = csr_matrix(A)
            sparse.save_npz(mat_path, csr_A)
        elif mat_type =='SciCOO': 
            sparse.save_npz(mat_path, A)
        elif mat_type =='COO': 
            WriteCOO2TXT(mat_path, A)
        else:
            raise Exception("the matrix type is wrong")           
    elif isinstance(A,sparse.csr.csr_matrix):
        if mat_type is None:
            pass
        elif mat_type == 'SciCSR':
            sparse.save_npz(mat_path, A)
        elif mat_type =='SciCOO': 
            coo_A = coo_matrix(A)
            sparse.save_npz(mat_path,coo_A)
        elif mat_type =='COO': 
            WriteCOO2TXT(mat_path, A.tocoo())
        else:
            raise Exception("the matrix type is wrong")           
    else:
        raise Exception("the type of input matrix A is not scipy.sparse")           

    if F is not None and need_rhs:
        mat_dir = os.path.dirname(mat_path)
        mat_name = os.path.basename(mat_path)
        if mat_type in ('SciCSR', 'SciCOO'):  
            idx_str = mat_name[9:-4]
            rhs_name = f'rhs{idx_str}.npy'
            rhs_path = os.path.join(mat_dir,rhs_name)
            np.save(rhs_path, F)
        elif mat_type == 'COO':
            idx_str = mat_name[3:-4]
            rhs_name = f'rhs{idx_str}.txt'
            rhs_path = os.path.join(mat_dir,rhs_name)
            WriteVec2TXT(rhs_path, F)

if __name__ == '__main__':
    a = np.array([1.0,2.0,3.0,4.0])
    print(a)
    WriteVec2TXT('./b.txt', a)

    b = ReadVecFromTXT('./b.txt')
    print(b)


