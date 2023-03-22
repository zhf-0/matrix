import os
import numpy as np
import re
import json
import time
import subprocess
import psutil
from operator import itemgetter, attrgetter
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, csc_matrix 
from multiprocessing import Pool, current_process, Lock

class SolveAndAnalysis:
    def __init__(self,json_dir,batch_size,summary_name):
        self.json_dir = json_dir
        os.makedirs(self.json_dir,exist_ok=True)
        self.summary_file = os.path.join(self.json_dir,summary_name) 

        self.batch_size = batch_size
        self.cpu_list = None

        self.InitSummary()

    def InitSummary(self):
        ksp_types =  ['cg','bicg','bcgsl','gmres']
        pc_types = ['jacobi','sor','bjacobi','gamg','ilu','asm','hypre','none'] # hypre means amg method in hypre
        self.ksp_pc = []
        for k in ksp_types:
            for p in pc_types:
                name = f'{k}-{p}'
                self.ksp_pc.append(name)

        if os.path.exists(self.summary_file):
            with open(self.summary_file,'r') as f:
                self.summary = json.load(f)
        else:
            self.summary = {}
            self.summary['total_num'] = 0
            self.summary['total_list'] = []
            self.summary['useless_num'] = 0
            self.summary['useless'] = []
            self.summary['none_num'] = 0
            self.summary['none'] = []
            for item in self.ksp_pc:
                tmp_name = item+'_num'
                self.summary[tmp_name] = 0
                self.summary[item] = []

    def Process(self,idx,mat_path,parameter,need_rhs=0):
        if idx not in self.summary['total_list']:
            result_path = os.path.join(self.json_dir,f'result{idx}.json')
            if os.path.exists(result_path):
                with open(result_path,'r') as f:
                    info = json.load(f)
            else:
                info = {}
                info['finished'] = False
                
            info.update(parameter.others)
            info['parameter'] = parameter.para

            print('========================================================')
            print(f'begin to solve matrix {idx}')
            
            inputs = []
            for keyname in self.ksp_pc:
                if (keyname in info) and ('time' in info[keyname]): 
                    pass
                else:
                    tmp_input = [mat_path,keyname,need_rhs]
                    inputs.append(tmp_input)

            if len(inputs) > 0:
                info = self.SolveWithAllMethods(inputs,info,result_path)

            self.DataAnalysis(info,result_path,idx)
        else:
            print(f'matrix {idx} has been solved')

    def SolveWithAllMethods(self,inputs,info,result_path):

        # begin = time.time()
        for item in inputs:
            res = self.CallWorker(item)
            keyname = res[0] 
            info[keyname] = {}
            info[keyname]['iter'] = res[1]
            info[keyname]['stop_reason'] = res[2]
            info[keyname]['r_norm'] = res[3]
            info[keyname]['b_norm'] = res[4]
            info[keyname]['relative_norm'] = res[5]
            info[keyname]['time'] = res[6]
            info[keyname]['avg_time'] = sum(res[6])/len(res[6])

            with open(result_path,'w') as f:
                json.dump(info,f,indent=4)

        # end = time.time()
        # print(f'total time = {end-begin}')

        return info
        

    def DataAnalysis(self,info,result_path,idx):
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


        info['finished'] = True
        info['class'] = class_name
        self.summary['total_num'] += 1
        self.summary['total_list'].append(idx)
        self.summary[class_name].append(idx)
        self.summary[class_name+'_num'] += 1
        
        with open(result_path,'w') as f:
            json.dump(info,f,indent=4)

        with open(self.summary_file,'w') as f:
            json.dump(self.summary,f,indent=4)

        return reserve

    def CallWorker(self,arg_list):
        mat_path = arg_list[0]
        keyname = arg_list[1]
        need_rhs = arg_list[2]

        if self.cpu_list is not None:
            cur = current_process()
            rank = (cur._identity[0] - 1)%( len(self.cpu_list) )
            cpu_idx = self.cpu_list[rank]
            p = psutil.Process()
            p.cpu_affinity([cpu_idx])
            print(f'begin to run with {keyname}, rank = {rank}, cpu = {cpu_idx}')
        else:
            print(f'begin to run with {keyname}')


        ksp_name, pc_name = keyname.split('-')
        cmd = f'python3 PetscWorker.py  -path {mat_path} -ksp {ksp_name} -pc {pc_name} -rhs {need_rhs}'

        iter_num = 10000
        reason = -100
        resi = 0	
        b_norm = 0
        rlt_resi = 0
        elapsed_time = [1e8] * self.batch_size
        is_succ = True

        #tmp1 = time.time()
        try:
            run_output = subprocess.check_output(cmd,shell=True)
        except subprocess.CalledProcessError as error:
            print('\033[31m running fail! matrix: {}  \033[0m'.format(mat_path))
            print(error.output.decode('utf-8'))
            is_succ = False
        else:
            contents = run_output.decode('utf-8')
            lines = contents.split('\n')

            for line in lines:
                if re.search('iter_num',line):
                    iter_num = int( line.split('=')[-1] )
                elif re.search('stop_reason',line):
                    reason = int( line.split('=')[-1] )
                elif re.search('residual_norm',line):
                    resi = float( line.split('=')[-1] )
                elif re.search('b_norm',line):
                    b_norm = float( line.split('=')[-1] )
                elif re.search('relative_resi',line):
                    rlt_resi = float( line.split('=')[-1] )
                elif re.search('elapsed_time',line):
                    elapsed_time[0] = float( line.split('=')[-1] )

            #tmp2 = time.time()
            #print(f'{keyname} time = {tmp2 - tmp1} ')

        if is_succ:
            for i in range(1,self.batch_size):
                run_output = subprocess.check_output(cmd,shell=True)
                contents = run_output.decode('utf-8')
                lines = contents.split('\n')
                for line in lines:
                    if re.search('elapsed_time',line):
                        elapsed_time[i] = float( line.split('=')[-1] )

        return (keyname,iter_num,reason,resi,b_norm,rlt_resi,elapsed_time)

    def SortSummaryByNum(self):
        num_list = []
        for key in self.summary:
            if key.endswith('num'):
                num_list.append( (key, self.summary[key]) )

        sorted_num_list = sorted(num_list, key = itemgetter(1))
        self.summary['sorted_list'] = sorted_num_list

        with open(self.summary_file,'w') as f:
            json.dump(self.summary,f,indent=4)

class ParaSolveAndAnalysis(SolveAndAnalysis):
    def __init__(self,json_dir,batch_size,summary_name,num_cpu):
        super().__init__(json_dir,batch_size,summary_name)

        cpu_list = psutil.Process().cpu_affinity()
        print(f'original cpu list is {cpu_list}')
        self.num_cpu = num_cpu
        self.cpu_list = cpu_list[0:self.num_cpu]
        print(f'cpu list is {self.cpu_list}')
        
    def SolveWithAllMethods(self,inputs,info,result_path):

        # begin = time.time()
        with Pool(processes=self.num_cpu) as pool:
            # results = pool.imap_unordered(CallPetsc,inputs)
            results = pool.imap_unordered(self.CallWorker,inputs)
            for res in results:
                keyname = res[0] 
                info[keyname] = {}
                info[keyname]['iter'] = res[1]
                info[keyname]['stop_reason'] = res[2]
                info[keyname]['r_norm'] = res[3]
                info[keyname]['b_norm'] = res[4]
                info[keyname]['relative_norm'] = res[5]
                info[keyname]['time'] = res[6]
                info[keyname]['avg_time'] = sum(res[6])/len(res[6])

                with open(result_path,'w') as f:
                    json.dump(info,f,indent=4)

        # end = time.time()
        # print(f'total time = {end-begin}')

        return info
        
class ParaSolveAndAnalysis2(ParaSolveAndAnalysis):
    '''
    Three labels classification using Petsc's default ksp and pc
    '''
    def __init__(self,json_dir,batch_size,summary_name,num_cpu):
        super().__init__(json_dir,batch_size,summary_name,num_cpu)

    def InitSummary(self):
        ksp_types =  ['cg','bicg','bcgsl','gmres']
        pc_types = ['jacobi','sor','bjacobi','gamg','ilu','asm','none'] 
        self.ksp_pc = []
        for k in ksp_types:
            for p in pc_types:
                name = f'{k}-{p}'
                self.ksp_pc.append(name)

        if os.path.exists(self.summary_file):
            with open(self.summary_file,'r') as f:
                self.summary = json.load(f)
        else:
            self.summary = {}
            self.summary['total_num'] = 0
            self.summary['total_list'] = []
            self.summary['none_num'] = 0
            self.summary['none_list'] = []

            self.summary['label0_num'] = 0
            self.summary['label0_list'] = []
            self.summary['label1_num'] = 0
            self.summary['label1_list'] = []
            self.summary['label2_num'] = 0
            self.summary['label2_list'] = []

    def DataAnalysis(self,info,result_path,idx):
        time_list = []
        label_list = []
        ratio = 1.1
        
        for keyname in self.ksp_pc:
            if info[keyname]['stop_reason'] > 0:
                time_list.append( (keyname, info[keyname]['avg_time']) )
            else:
                label_list.append((idx,keyname,0))
                self.summary['label0_num'] += 1
                self.summary['label0_list'].append((idx,keyname,0))

        if len(time_list) == 0:
            self.summary['none_num'] += 1
            self.summary['none_list'].append(idx)
        else:
            sorted_time_list = sorted(time_list, key = itemgetter(1))
            fastest_time = sorted_time_list[0][1]
            for item in sorted_time_list:
                if item[1] <= fastest_time * ratio:
                    label_list.append((idx,item[0],2))
                    self.summary['label2_num'] += 1
                    self.summary['label2_list'].append((idx,item[0],2))
                else:
                    label_list.append((idx,item[0],1))
                    self.summary['label1_num'] += 1
                    self.summary['label1_list'].append((idx,item[0],1))


        info['finished'] = True
        info['labels'] = label_list
        self.summary['total_num'] += 1
        self.summary['total_list'].append(idx)
        
        with open(result_path,'w') as f:
            json.dump(info,f,indent=4)

        with open(self.summary_file,'w') as f:
            json.dump(self.summary,f,indent=4)

class ParaSolveAndAnalysis3(ParaSolveAndAnalysis2):
    '''
    Three labels classification using methods in the paper 
    '''
    def __init__(self,json_dir,batch_size,summary_name,num_cpu):
        super().__init__(json_dir,batch_size,summary_name,num_cpu)

    def InitSummary(self):
        ksp_types =  ['cg','gmres','bcgsl','lsqr','fgmres']
        # pc_types = ['jacobi','sor','bjacobi','gamg','ilu','asm','none'] 
        # pc_types = ['BjIlu0QMD','BjIlu1QMD','BjLu','asm1','asm2','boomer','euclid','parasails','BjGmres']
        pc_types = ['BjIlu0QMD','BjIlu1QMD','BjLu','asm1','asm2','boomer','euclid','parasails']
        self.ksp_pc = []
        for k in ksp_types:
            for p in pc_types:
                name = f'{k}-{p}'
                self.ksp_pc.append(name)

        if os.path.exists(self.summary_file):
            with open(self.summary_file,'r') as f:
                self.summary = json.load(f)
        else:
            self.summary = {}
            self.summary['total_num'] = 0
            self.summary['total_list'] = []
            self.summary['none_num'] = 0
            self.summary['none_list'] = []

            self.summary['label0_num'] = 0
            self.summary['label0_list'] = []
            self.summary['label1_num'] = 0
            self.summary['label1_list'] = []
            self.summary['label2_num'] = 0
            self.summary['label2_list'] = []

class ParaSolveAndAnalysis4(ParaSolveAndAnalysis3):
    '''
    Three labels classification using different data analysis method 
    '''
    def __init__(self,json_dir,batch_size,summary_name,num_cpu):
        super().__init__(json_dir,batch_size,summary_name,num_cpu)

    def DataAnalysis(self,info,result_path,idx):
        import math
        w1 = 1.0
        w2 = 1.0
        eps = 10**(-8)
        score_list = []
        label_list = []
        ratio = 0.9
        
        for keyname in self.ksp_pc:
            if info[keyname]['stop_reason'] > 0:
                score = math.log(1+w1/info[keyname]['avg_time']) * math.log(1+w2/(eps+info[keyname]['r_norm']))
                score_list.append( (keyname, score) )
            else:
                label_list.append((idx,keyname,0))
                self.summary['label0_num'] += 1
                self.summary['label0_list'].append((idx,keyname,0))

        if len(score_list) == 0:
            self.summary['none_num'] += 1
            self.summary['none_list'].append(idx)
        else:
            sorted_score_list = sorted(score_list, key = itemgetter(1))
            max_score = sorted_score_list[-1][1]
            for item in sorted_score_list:
                if item[1] >= max_score * ratio:
                    label_list.append((idx,item[0],2))
                    self.summary['label2_num'] += 1
                    self.summary['label2_list'].append((idx,item[0],2))
                else:
                    label_list.append((idx,item[0],1))
                    self.summary['label1_num'] += 1
                    self.summary['label1_list'].append((idx,item[0],1))


        info['finished'] = True
        info['labels'] = label_list
        self.summary['total_num'] += 1
        self.summary['total_list'].append(idx)
        
        with open(result_path,'w') as f:
            json.dump(info,f,indent=4)

        with open(self.summary_file,'w') as f:
            json.dump(self.summary,f,indent=4)


class ParaSolveAndAnalysis5(ParaSolveAndAnalysis4):
    '''
    Three labels classification using different data analysis method 
    split label 0 
    '''
    def __init__(self,json_dir,batch_size,summary_name,num_cpu):
        super().__init__(json_dir,batch_size,summary_name,num_cpu)

    def DataAnalysis(self,info,result_path,idx):
        import math
        w1 = 1.0
        w2 = 1.0
        eps = 10**(-8)
        score_list = []
        label_list = []
        ratio = 0.9
        
        for keyname in self.ksp_pc:
            if info[keyname]['stop_reason'] > 0 or info[keyname]['stop_reason'] == -3 :
                score = math.log(1+w1/info[keyname]['avg_time']) * math.log(1+w2/(eps+info[keyname]['r_norm']))
                score_list.append( (keyname, score) )
            else:
                label_list.append((idx,keyname,0))
                self.summary['label0_num'] += 1
                self.summary['label0_list'].append((idx,keyname,0))

        if len(score_list) == 0:
            self.summary['none_num'] += 1
            self.summary['none_list'].append(idx)
        else:
            sorted_score_list = sorted(score_list, key = itemgetter(1))
            max_score = sorted_score_list[-1][1]
            for item in sorted_score_list:
                if item[1] >= max_score * ratio:
                    label_list.append((idx,item[0],2))
                    self.summary['label2_num'] += 1
                    self.summary['label2_list'].append((idx,item[0],2))
                else:
                    label_list.append((idx,item[0],1))
                    self.summary['label1_num'] += 1
                    self.summary['label1_list'].append((idx,item[0],1))


        info['finished'] = True
        info['labels'] = label_list
        self.summary['total_num'] += 1
        self.summary['total_list'].append(idx)
        
        with open(result_path,'w') as f:
            json.dump(info,f,indent=4)

        with open(self.summary_file,'w') as f:
            json.dump(self.summary,f,indent=4)

