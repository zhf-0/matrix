import os
import sys
import json

sys.path.append('./PDEs/')
import SuiteSparse as pde0
import poisson_lfem_mixedbc_2d as pde1
import diffusion_convection_reaction_lfem_mixedbc_2d as pde2
import helmholtz_robinbc_2d as pde3
import linear_elasticity_lfem_2d as pde4
import maxwell_nedelec_3d as pde5

def GenerateScript(script_path, header, footer):
    json_dir = './JsonFiles/'
    os.makedirs(json_dir,exist_ok=True)
    mat_type = 'SciCSR'
    mat_dir = './MatData/'
    os.makedirs(mat_dir,exist_ok=True)
    need_rhs = True
    
    contents = []
    for i in range(6):
        
        if mat_type == 'SciCSR':
            mat_path = os.path.join(mat_dir,f'scipy_csr{i}.npz')
        elif mat_type =='SciCOO': 
            mat_path = os.path.join(mat_dir,f'scipy_coo{i}.npz')
        elif mat_type =='COO': 
            mat_path = os.path.join(mat_dir,f'coo{i}.txt')


        if i==0:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde0.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                
                # the matrix with ss_id = 1 is used as the example 
                # the ss_id begin with 1
                # if ss_id = 0, the program will download all matrix in SuiteSparse
                para.DefineFixPara('ss_id',1)

                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 0
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/SuiteSparse.py' + line + '\n'
                contents.append(one_command)

                # if the matrix doesn't exist, regardless whether the json file exists or not,
                # just write json to the json file
                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)

        if i == 1:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde1.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 1
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/poisson_lfem_mixedbc_2d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)

        if i == 2:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde2.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                para.DefineFixPara('seed',i)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 2
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/diffusion_convection_reaction_lfem_mixedbc_2d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)
        
        if i == 3:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde3.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 3
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/helmholtz_robinbc_2d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)
        
        if i == 4:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde4.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 4
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/linear_elasticity_lfem_2d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)
        
        if i == 5:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde5.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 5
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/maxwell_nedelec_3d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)
        

    with open(script_path,'w',encoding='utf-8') as f:
        f.writelines(header)
        f.writelines(contents)
        f.writelines(footer)


def GenerateScript2(script_path, header, footer):
    json_dir = './JsonFiles/'
    os.makedirs(json_dir,exist_ok=True)
    mat_type = 'SciCSR'
    mat_dir = './MatData/'
    os.makedirs(mat_dir,exist_ok=True)
    need_rhs = True
    
    contents = []
    for i in range(2000):
        
        if mat_type == 'SciCSR':
            mat_path = os.path.join(mat_dir,f'scipy_csr{i}.npz')
        elif mat_type =='SciCOO': 
            mat_path = os.path.join(mat_dir,f'scipy_coo{i}.npz')
        elif mat_type =='COO': 
            mat_path = os.path.join(mat_dir,f'coo{i}.txt')


        if i < 500:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde1.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 1
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/poisson_lfem_mixedbc_2d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)

        if  500 <= i < 1000:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde2.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                para.DefineFixPara('seed',i)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 2
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/diffusion_convection_reaction_lfem_mixedbc_2d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)
        
        if 1000 <= i < 1500:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde3.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 3
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/helmholtz_robinbc_2d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)
        
        if 1500 <= i < 2000:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde4.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 4
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/linear_elasticity_lfem_2d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)
        
        if 2000 <= i < 2500:
            if not os.path.exists(mat_path):
                print(f'generate command for matrix {i}')
                output = f'echo begin to generate matrix {i} \n'
                contents.append(output)

                para = pde5.Para()
                para.DefineFixPara('mat_type',mat_type)
                para.DefineFixPara('mat_path',mat_path)
                para.DefineFixPara('need_rhs',need_rhs)
                
                json_path = os.path.join(json_dir,f'result{i}.json')
                json_result = {}
                json_result['PDE_idx'] = 5
                json_result['GenMat'] = para.para
                line = ' '
                for k,v in para.para.items():
                    line = line + f' --{k} {v} '
            
                one_command = 'python ./PDEs/maxwell_nedelec_3d.py' + line + '\n'
                contents.append(one_command)

                with open(json_path,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)
        

    with open(script_path,'w',encoding='utf-8') as f:
        f.writelines(header)
        f.writelines(contents)
        f.writelines(footer)


def GenerateScript3(script_path, header, footer):
    '''
    select matrix from SuiteSparse Matrix Collection
    '''
    import re
    json_dir = './JsonFiles/'
    os.makedirs(json_dir,exist_ok=True)
    mat_type = 'SciCSR'
    mat_dir = './MatData/'
    os.makedirs(mat_dir,exist_ok=True)
    need_rhs = False
    contents = []

    with open('./PDEs/suitesparse/meta.json','r') as f:
        info = json.load(f)

    # select matrices related to computational fluid dynamics
    num = len(info)
    ss_id = []
    for i in range(num):
        mat_info = info[i]
        kind = mat_info['kind']
        # if re.search('fluid',kind) and mat_info['posdef'] and (mat_info['rows'] == mat_info['cols']):
        if re.search('fluid',kind) and (mat_info['rows'] == mat_info['cols']):
            ss_id.append(mat_info['id'])

    # the matrix with id 2384 is 3.4G
    ss_id.remove(2384)

    # generate commands
    print('the number of selected matrices is:',len(ss_id))
    for idx in ss_id:
        mat_path = os.path.join(mat_dir,f'scipy_csr{idx}.npz')
        if not os.path.exists(mat_path):
            print(f'generate command for matrix {idx}')
            output = f'echo begin to generate matrix {idx} \n'
            contents.append(output)

            para = pde0.Para()
            para.DefineFixPara('mat_type',mat_type)
            para.DefineFixPara('mat_path',mat_path)
            para.DefineFixPara('ss_id',idx)

            json_path = os.path.join(json_dir,f'result{idx}.json')
            json_result = {}
            json_result['PDE_idx'] = 0
            json_result['GenMat'] = para.para
            line = ' '
            for k,v in para.para.items():
                line = line + f' --{k} {v} '
        
            one_command = 'python ./PDEs/SuiteSparse.py' + line + '\n'
            contents.append(one_command)

            # if the matrix doesn't exist, regardless whether the json file exists or not,
            # just write json to the json file
            with open(json_path,'w',encoding='utf-8') as f:
                json.dump(json_result,f,indent=4)
            
    with open(script_path,'w',encoding='utf-8') as f:
        f.writelines(header)
        f.writelines(contents)
        f.writelines(footer)

def ReConstructMatByJson(json_list,script_path,header,footer):
    func_list = ['python ./PDEs/SuiteSparse.py',
                 'python ./PDEs/poisson_lfem_mixedbc_2d.py',
                 'python ./PDEs/diffusion_convection_reaction_lfem_mixedbc_2d.py',
                 'python ./PDEs/helmholtz_robinbc_2d.py',
                 'python ./PDEs/linear_elasticity_lfem_2d.py' ,
                 'python ./PDEs/maxwell_nedelec_3d.py' 
                ]
    
    contents = []
    for item in json_list:
        with open(item,'r',encoding='utf-8') as f:
            json_result = json.load(f)

        para = json_result['GenMat']
        mat_path = para['mat_path']
        if not os.path.exists(mat_path):
            print(f"generate command for matrix: {para['mat_path']}")
            output = f"echo begin to generate matrix: {para['mat_path']} \n"
            contents.append(output)

            PDE_idx = json_result['PDE_idx']
            line = ' '
            for k,v in para.items():
                line = line + f' --{k} {v} '

            one_command = func_list[PDE_idx] + line + '\n'
            contents.append(one_command)
            
    with open(script_path,'w',encoding='utf-8') as f:
        f.writelines(header)
        f.writelines(contents)
        f.writelines(footer)

def TestGen():
    script_path = 'genmat.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    # GenerateScript(script_path,header,footer)
    # GenerateScript2(script_path,header,footer)
    GenerateScript3(script_path,header,footer)

def TestReConstruct():
    # use the json file generated by TestGen() function
    # delete the matrix files in MatData
    mat_file_list = os.listdir('./MatData/')
    for file in mat_file_list:
        file_path = os.path.join('./MatData',file)
        os.remove(file_path)
    print('delete the matrix files')

    json_file_list = os.listdir('./JsonFiles/')
    json_list = [os.path.join('./JsonFiles',item) for item in json_file_list]

    script_path = 'reconstruct.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    ReConstructMatByJson(json_list,script_path,header,footer)

if __name__ == '__main__':
    TestGen()
    # TestReConstruct()
