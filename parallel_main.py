import os
import numpy as np
import re
import json
import time
import subprocess
import psutil
import random
from operator import itemgetter, attrgetter

# from scipy.sparse.linalg import spsolve, inv, norm
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, csc_matrix 

# import multiprocessing
from multiprocessing import Pool, current_process, Lock

import PDEs.PoissonFEM2d as pde0
import PDEs.ConvectionDiffusionReactionFEMwithDirichletBC2d as pde1

class Parameter:
    def __init__(self):
        self.para = {}

    def DefineRandInt(self,name,begin,end,num=1):
        if num == 1:
            self.para[name] = np.random.randint(begin,end)
        else:
            self.para[name] = np.random.randint(begin,end,num)

    def DefineRandFloat(self,name,begin,end,num=1):
        if num == 1:
            self.para[name] = np.random.uniform(begin,end)
        else:
            self.para[name] = np.random.uniform(begin,end,num)
    
    def DefineFixPara(self,name,val):
        self.para[name] = val

    def CopyValue(self,name,name1):
        self.para[name1] = self.para[name]
        
    def RandChoose(self,name,list_val):
        idx = np.random.randint(0,len(list_val))
        self.para[name] = list_val[idx]

    def Register(self,info):
        info.update(self.para)
        return info

class CreateData:
    def __init__(self,num):
        self.num = num
        self.root_path = './MatData/'
        self.total_file = os.path.join(self.root_path,'total.json')
        self.mat_path = os.path.join(self.root_path,'mat')
        os.makedirs(self.root_path,exist_ok=True)
        os.makedirs(self.mat_path,exist_ok=True)

        cpu_list = psutil.Process().cpu_affinity()
        print(f'original cpu list is {cpu_list}')
        self.num_cpu = 2
        self.cpu_list = cpu_list[0:self.num_cpu]
        print(f'cpu list is {self.cpu_list}')
        
        ksp_types =  ['cg','bicg','bcgsl','gmres']
        pc_types = ['jacobi','sor','bjacobi','gamg','ilu','asm','hypre','none'] # hypre means amg method in hypre
        self.ksp_pc = []
        for k in ksp_types:
            for p in pc_types:
                name = f'{k}-{p}'
                self.ksp_pc.append(name)

        if os.path.exists(self.total_file):
            with open(self.total_file,'r') as f:
                self.total = json.load(f)
        else:
            self.total = {}
            self.total['total_num'] = 0
            self.total['total_list'] = []
            self.total['useless_num'] = 0
            self.total['useless'] = []
            self.total['none_num'] = 0
            self.total['none'] = []
            for item in self.ksp_pc:
                tmp_name = item+'_num'
                self.total[tmp_name] = 0
                self.total[item] = []

    def Process(self):
        print('begin to process')
        for i in range(self.num):
            if i not in self.total['total_list']:
                self.AllMethodsForOneMat(i)

    def AllMethodsForOneMat(self,i):
        np.random.seed(i)
        random.seed(i)

        result_file = os.path.join(self.root_path,f'result{i}.json')
        if os.path.exists(result_file):
            with open(result_file,'r') as f:
                info = json.load(f)
        else:
            info = {}
            info['finished'] = False
            info['seed'] = i


        print('========================================================')
        print(f'begin to run, seed = {i}')

        print('begin to generate matrix')
        scimat_path = os.path.join(self.mat_path,f'scipy_csr{i}.npz')
        if not os.path.exists(scimat_path):
            if i < 1000:
                info['PDE_type'] = 0
                para = Parameter()
                para.DefineRandInt('nx', 50, 300)
                para.CopyValue('nx', 'ny')
                para.DefineRandInt('blockx',20,40)
                para.CopyValue('blockx', 'blocky')
                para.RandChoose('meshtype',['tri','quad'])
                para.DefineRandInt('p',1,4)

                A = pde0.GenerateMat(**para.para)
                info = para.Register(info)
            elif i>=1000 and i<2000:
                info['PDE_type'] = 1
                para = Parameter()
                para.DefineRandInt('nx', 50, 300)
                para.CopyValue('nx', 'ny')
                para.DefineRandInt('blockx',20,40)
                para.CopyValue('blockx', 'blocky')
                para.RandChoose('meshtype',['tri','quad'])
                para.DefineRandInt('p',1,4)

                A = pde1.GenerateMat(**para.para)
                info = para.Register(info)

            csr_A = csr_matrix(A)
            sparse.save_npz(scimat_path, csr_A)

        batch_size = 3
        #begin = time.time()
        with Pool(processes=self.num_cpu) as pool:
            results = pool.imap_unordered(CallPetsc,[ [i,scimat_path,keyname,info,self.cpu_list,batch_size] for keyname in self.ksp_pc ])
            for res in results:
                if type(res[0]) == str:
                    keyname = res[0] 
                    info[keyname] = {}
                    info[keyname]['iter'] = res[1]
                    info[keyname]['stop_reason'] = res[2]
                    info[keyname]['r_norm'] = res[3]
                    info[keyname]['b_norm'] = res[4]
                    info[keyname]['relative_norm'] = res[5]
                    info[keyname]['time'] = res[6]
                    info[keyname]['avg_time'] = sum(res[6])/len(res[6])

                    with open(result_file,'w') as f:
                        json.dump(info,f,indent=4)

        #end = time.time()
        print('========================================================')
        #print(f'total time = {end-begin}')
        
        reserve = self.DataAnalysis(info,result_file,i)
        if reserve == False:
            os.remove(scimat_path)

    def DataAnalysis(self,info,result_file,i):
        time_list = []
        threshold = 0.1
        reserve = True
        for keyname in self.ksp_pc:
            if info[keyname]['stop_reason'] > 0:
                time_list.append( (keyname, info[keyname]['avg_time']) )

        if len(time_list) == 0:
            class_name = 'none'
            reserve = False
        elif len(time_list) == 1:
            class_name = time_list[0][0]
        else:
            sorted_time_list = sorted(time_list, key = itemgetter(1))
            info['sorted_time'] = sorted_time_list
            fastest_time = sorted_time_list[0][1]
            second_fastest = sorted_time_list[1][1]
            if (second_fastest - fastest_time)/fastest_time > threshold:
                class_name = sorted_time_list[0][0]
            else:
                class_name = 'useless'
                reserve = False


        info['finished'] = True
        info['class'] = class_name
        self.total['total_num'] += 1
        self.total['total_list'].append(i)
        self.total[class_name].append(i)
        self.total[class_name+'_num'] += 1
        
        with open(result_file,'w') as f:
            json.dump(info,f,indent=4)

        with open(self.total_file,'w') as f:
            json.dump(self.total,f,indent=4)

        return reserve


def CallPetsc(arg_list):
    cpu_list = arg_list[4]
    cur = current_process()
    rank = (cur._identity[0] - 1)%( len(cpu_list) )
    cpu_idx = cpu_list[rank]
    p = psutil.Process()
    p.cpu_affinity([cpu_idx])

    i = arg_list[0]
    scimat_path = arg_list[1]
    keyname = arg_list[2]
    info = arg_list[3]

    batch_size = arg_list[5]

    if (keyname in info) and ('time' in info[keyname]): 
        return (-1,0,0,0,0,0,0)
    else:
        print('===========================')
        print(f'begin to run with {keyname}, rank = {rank}, cpu = {cpu_idx}')

        ksp_name, pc_name = keyname.split('-')
        #cmd = f'python3 petsc_para_worker.py -idx {cpu_idx}  -path {scimat_path} -ksp {ksp_name} -pc {pc_name}'
        cmd = f'python3 petsc_worker.py  -path {scimat_path} -ksp {ksp_name} -pc {pc_name}'

        iter_num = 10000
        reason = -10
        resi = 0	
        b_norm = 0
        rlt_resi = 0
        elapsed_time = [1e8] * batch_size
        is_succ = True

        #tmp1 = time.time()
        try:
            run_output = subprocess.check_output(cmd,shell=True)
        except subprocess.CalledProcessError as error:
            print('\033[31m running fail! i = {}  \033[0m'.format(i))
            print(error.output.decode('utf-8'))
            is_succ = False
        else:
            contents = run_output.decode('utf-8')
            lines = contents.split('\n')

            for line in lines:
                if re.search('iter_num',line):
                    iter_num = int( line.split('=')[-1] )
                if re.search('stop_reason',line):
                    reason = int( line.split('=')[-1] )
                if re.search('residual_norm',line):
                    resi = float( line.split('=')[-1] )
                if re.search('b_norm',line):
                    b_norm = float( line.split('=')[-1] )
                if re.search('relative_resi',line):
                    rlt_resi = float( line.split('=')[-1] )
                if re.search('elapsed_time',line):
                    elapsed_time[0] = float( line.split('=')[-1] )

        #tmp2 = time.time()
        #print(f'{keyname} time = {tmp2 - tmp1} ')

        if is_succ:
            for i in range(1,batch_size):
                run_output = subprocess.check_output(cmd,shell=True)
                contents = run_output.decode('utf-8')
                lines = contents.split('\n')
                for line in lines:
                    if re.search('elapsed_time',line):
                        elapsed_time[i] = float( line.split('=')[-1] )

        return (keyname,iter_num,reason,resi,b_norm,rlt_resi,elapsed_time)

def main():
    mat = CreateData(3)
    mat.Process()


if __name__ == '__main__':
    main()
