import PDEs.SuiteSparse as pde0
import PDEs.PoissonFEM2d as pde1
import PDEs.ConvectionDiffusionReactionFEMwithDirichletBC2d as pde2
import PDEs.LinearElasticityFEM2D as pde3
import PDEs.FourthOrderEllipticDG2D as pde4
import PetscSolvers
import json
import os

def TestDemo():
    batch_size = 3
    json_dir = './JsonFiles/'
    mat_type = 'SciCSR'
    mat_dir = './MatData/'
    os.makedirs(mat_dir,exist_ok=True)
    need_rhs = 0

    # solver = PetscSolvers.SolveAndAnalysis(json_dir,batch_size,'summary.json')
    solver = PetscSolvers.ParaSolveAndAnalysis(json_dir,batch_size,'summary.json',num_cpu=2)

    i = 0
    if mat_type == 'SciCSR':
        mat_path = os.path.join(mat_dir,f'scipy_csr{i}.npz')
    elif mat_type =='SciCOO': 
        mat_path = os.path.join(mat_dir,f'scipy_coo{i}.npz')
    elif mat_type =='COO': 
        mat_path = os.path.join(mat_dir,f'coo{i}.txt')

    parameter = pde1.Para()
    parameter.DefineFixPara('mat_type',mat_type)
    parameter.DefineFixPara('mat_path',mat_path)
    parameter.DefineFixPara('need_rhs',need_rhs)
    parameter.DefineFixPara('seed',i)
    parameter.others['PDE_type'] = 1

    if not os.path.exists(mat_path):
        row_num, col_num, nnz = pde1.GenerateMat(**parameter.para)
        parameter.others['row_num'] = row_num
        parameter.others['col_num'] = col_num
        parameter.others['nnz'] = nnz

    solver.Process(i,mat_path,parameter,need_rhs)
    
def TestPDE():
    batch_size = 3
    json_dir = './JsonFiles/'
    mat_type = 'SciCSR'
    mat_dir = './MatData/'
    os.makedirs(mat_dir,exist_ok=True)
    need_rhs = 0
    # need_rhs = 1
    
    solver = PetscSolvers.ParaSolveAndAnalysis(json_dir,batch_size,'summary.json',num_cpu=2)

    for i in range(4000):
        if mat_type == 'SciCSR':
            mat_path = os.path.join(mat_dir,f'scipy_csr{i}.npz')
        elif mat_type =='SciCOO': 
            mat_path = os.path.join(mat_dir,f'scipy_coo{i}.npz')
        elif mat_type =='COO': 
            mat_path = os.path.join(mat_dir,f'coo{i}.txt')
        
        if i < 1000:
            parameter = pde1.Para()
            parameter.DefineFixPara('mat_type',mat_type)
            parameter.DefineFixPara('mat_path',mat_path)
            parameter.DefineFixPara('need_rhs',need_rhs)
            parameter.DefineFixPara('seed',i)
            parameter.others['PDE_type'] = 1

            if not os.path.exists(mat_path):
                row_num, col_num, nnz = pde1.GenerateMat(**parameter.para)
                parameter.others['row_num'] = row_num
                parameter.others['col_num'] = col_num
                parameter.others['nnz'] = nnz

            solver.Process(i,mat_path,parameter,need_rhs)
            
        elif 1000 <= i < 2000:
            parameter = pde2.Para()
            parameter.DefineFixPara('mat_type',mat_type)
            parameter.DefineFixPara('mat_path',mat_path)
            parameter.DefineFixPara('need_rhs',need_rhs)
            parameter.DefineFixPara('seed',i)
            parameter.others['PDE_type'] = 2

            if not os.path.exists(mat_path):
                row_num, col_num, nnz = pde1.GenerateMat(**parameter.para)
                parameter.others['row_num'] = row_num
                parameter.others['col_num'] = col_num
                parameter.others['nnz'] = nnz

            solver.Process(i,mat_path,parameter,need_rhs)

        elif 2000 <= i < 3000:
            parameter = pde3.Para()
            parameter.DefineFixPara('mat_type',mat_type)
            parameter.DefineFixPara('mat_path',mat_path)
            parameter.DefineFixPara('need_rhs',need_rhs)
            parameter.others['PDE_type'] = 3

            if not os.path.exists(mat_path):
                row_num, col_num, nnz = pde3.GenerateMat(**parameter.para)
                parameter.others['row_num'] = row_num
                parameter.others['col_num'] = col_num
                parameter.others['nnz'] = nnz

            solver.Process(i,mat_path,parameter,need_rhs)
            
        elif 3000 <= i < 4000:
            parameter = pde4.Para()
            parameter.DefineFixPara('mat_type',mat_type)
            parameter.DefineFixPara('mat_path',mat_path)
            parameter.DefineFixPara('need_rhs',need_rhs)
            parameter.others['PDE_type'] = 4
            
            if not os.path.exists(mat_path):
                row_num, col_num, nnz = pde4.GenerateMat(**parameter.para)
                parameter.others['row_num'] = row_num
                parameter.others['col_num'] = col_num
                parameter.others['nnz'] = nnz

            solver.Process(i,mat_path,parameter,need_rhs)

    solver.SortSummaryByNum()

def TestSuiteSparse():
    '''
    If all or partial suite sparse matrixes have been downloaded
    We can not use customized filter in ssgetpy, therefore we create a json file containing all the infomation of all matrixes in suite sparse matrix set. 
    The path of the json file is ./PDEs/suitesparse/meta.json
    '''
    batch_size = 3
    json_dir = './JsonFiles/'
    mat_type = 'SciCSR'
    if mat_type == 'SciCSR':
        name_template = 'scipy_csr{}.npz'
    elif mat_type =='SciCOO': 
        name_template = 'scipy_coo{}.npz'
    elif mat_type =='COO': 
        name_template = 'coo{}.txt'
        
    mat_dir = './MatData/'
    os.makedirs(mat_dir,exist_ok=True)

    solver = PetscSolvers.ParaSolveAndAnalysis(json_dir,batch_size,'summary.json',num_cpu=8)
    
    select_idx = []
    with open('./PDEs/suitesparse/meta.json','r') as f:
        meta_list = json.load(f)

    for i,item in enumerate(meta_list):
        if item["rows"] == item["cols"] and 1000 <= item["rows"] <= 10000 and item['nonzeros']<=200000 and item['real'] == True:
            select_idx.append(i)

    print(f'number of selected matrix = {len(select_idx)}')

    with open('./PDEs/suitesparse/paths.json','r') as f:
        path_list = json.load(f)

    id_dict = {}
    for item in path_list:
        id_dict[item['id']] = item['path']

    for idx in select_idx:
        ss_id = meta_list[idx]['id']
        mat_path = os.path.join(mat_dir,name_template.format(ss_id))
        parameter = pde0.Para()
        parameter.DefineFixPara('ss_id',ss_id)
        parameter.DefineFixPara('mat_type',mat_type)
        parameter.DefineFixPara('mat_path',mat_path)

        if ss_id in id_dict:
            parameter.DefineFixPara('ss_mat_path',id_dict[ss_id])
        else:
            print(f'matrix {ss_id} does not exist, need download')
            parameter.DefineFixPara('ss_mat_path',None)

        row_num, col_num, nnz = pde0.GenerateMat(**parameter.para)
        
        parameter.others['PDE_type'] = 0
        parameter.others['row_num'] = row_num
        parameter.others['col_num'] = col_num
        parameter.others['nnz'] = nnz

        # 'id' is already added to the dict by parameter.DefineFixPara('ss_id',ss_id)   
        meta_list[idx].pop('id')
        parameter.others.update(meta_list[idx])

        solver.Process(ss_id,mat_path,parameter)

    solver.SortSummaryByNum()


def TestSuiteSparse2():
    '''
    If no suite sparse matrix has been downloaded.
    We can not use customized filter in ssgetpy, therefore we create a json file containing all the infomation of all matrixes in suite sparse matrix set. 
    The path of the json file is ./PDEs/suitesparse/meta.json
    '''
    batch_size = 3
    json_dir = './JsonFiles/'
    mat_type = 'SciCSR'
    if mat_type == 'SciCSR':
        name_template = 'scipy_csr{}.npz'
    elif mat_type =='SciCOO': 
        name_template = 'scipy_coo{}.npz'
    elif mat_type =='COO': 
        name_template = 'coo{}.txt'
        
    mat_dir = './MatData/'
    os.makedirs(mat_dir,exist_ok=True)

    solver = PetscSolvers.ParaSolveAndAnalysis(json_dir,batch_size,'summary.json',num_cpu=8)
    
    select_idx = []
    with open('./PDEs/suitesparse/meta.json','r') as f:
        meta_list = json.load(f)

    for i,item in enumerate(meta_list):
        if item["rows"] == item["cols"] and 1000 <= item["rows"] <= 10000 and item['nonzeros']<=200000 and item['real'] == True:
            select_idx.append(i)

    print(f'number of selected matrix = {len(select_idx)}')

    for idx in select_idx:
        ss_id = meta_list[idx]['id']
        mat_path = os.path.join(mat_dir,name_template.format(ss_id))
        parameter = pde0.Para()
        parameter.DefineFixPara('ss_id',ss_id)
        parameter.DefineFixPara('mat_type',mat_type)
        parameter.DefineFixPara('mat_path',mat_path)
        parameter.DefineFixPara('ss_mat_path',None)

        row_num, col_num, nnz = pde0.GenerateMat(**parameter.para)
        
        parameter.others['PDE_type'] = 0
        parameter.others['row_num'] = row_num
        parameter.others['col_num'] = col_num
        parameter.others['nnz'] = nnz

        # 'id' is already added to the dict by parameter.DefineFixPara('ss_id',ss_id)   
        meta_list[idx].pop('id')
        parameter.others.update(meta_list[idx])

        solver.Process(ss_id,mat_path,parameter)

    solver.SortSummaryByNum()

def TestSuiteSparse3():
    '''
    If all suite sparse matrixes have been downloaded and transformed to csr format
    '''
    batch_size = 3
    json_dir = './JsonFiles/'
    mat_type = 'SciCSR'
    if mat_type == 'SciCSR':
        name_template = 'scipy_csr{}.npz'
    elif mat_type =='SciCOO': 
        name_template = 'scipy_coo{}.npz'
    elif mat_type =='COO': 
        name_template = 'coo{}.txt'
        
    mat_dir = './MatData/'
    os.makedirs(mat_dir,exist_ok=True)

    solver = PetscSolvers.ParaSolveAndAnalysis4(json_dir,batch_size,'summary.json',num_cpu=8)
    
    select_idx = []
    with open('./PDEs/suitesparse/meta.json','r') as f:
        meta_list = json.load(f)

    for i,item in enumerate(meta_list):
        if item["rows"] == item["cols"] and 1000 <= item["rows"] <= 10000 and item['nonzeros']<=200000 and item['real'] == True:
            select_idx.append(i)

    print(f'number of selected matrix = {len(select_idx)}')

    for idx in select_idx:
        ss_id = meta_list[idx]['id']
        mat_path = os.path.join(mat_dir,name_template.format(ss_id))
        parameter = pde0.Para()
        parameter.DefineFixPara('ss_id',ss_id)
        parameter.DefineFixPara('mat_type',mat_type)
        parameter.DefineFixPara('mat_path',mat_path)
        parameter.DefineFixPara('ss_mat_path',None)

        parameter.others['PDE_type'] = 0
        parameter.others['row_num'] = 0
        parameter.others['col_num'] = 0
        parameter.others['nnz'] = 0

        # 'id' is already added to the dict by parameter.DefineFixPara('ss_id',ss_id)   
        meta_list[idx].pop('id')
        parameter.others.update(meta_list[idx])

        solver.Process(ss_id,mat_path,parameter)

    solver.SortSummaryByNum()

if __name__ == '__main__':
    # TestPDE()
    # TestSuiteSparse3()
    TestDemo()
