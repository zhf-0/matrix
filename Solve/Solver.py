import os
from operator import itemgetter
import json
import yaml
from scipy.sparse import csr_matrix, load_npz
from petsc4py import PETSc
import numpy as np

class SeqTaskParRun:
    def __init__(self,json_dir,yaml_dir,mat_dir,batch_size,idx_list):
        self.yaml_dir = yaml_dir
        os.makedirs(self.yaml_dir,exist_ok=True)

        self.json_dir = json_dir
        os.makedirs(self.json_dir,exist_ok=True)

        self.mat_dir = mat_dir
        os.makedirs(self.mat_dir,exist_ok=True)

        self.batch_size = batch_size
        self.idx_list = idx_list

        self.paras = []

        self.summary_file = os.path.join(self.json_dir,'summary.json') 
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
            with open(self.summary_file,'r',encoding='utf-8') as f:
                self.summary = json.load(f)
        else:
            self.summary = {}
            self.summary['total_num'] = 0
            self.summary['total_list'] = []
            self.summary['failed_num'] = 0
            self.summary['failed'] = []
            for item in self.ksp_pc:
                tmp_name = item+'_num'
                self.summary[tmp_name] = 0
                self.summary[item] = []

    def DealYamlFile(self,json_result,mat_file,yaml_file):
        if os.path.exists(yaml_file):
            print('read yaml file')

            with open(yaml_file,'r',encoding='utf-8') as f:
                tmp = yaml.load_all(f,Loader=yaml.FullLoader)
                yaml_result = list(tmp)

            for item in yaml_result:
                keyname = item['ksp_pc']

                if keyname not in json_result['Solve']:
                    json_result['Solve'][keyname] = {}
                    json_result['Solve'][keyname]['iter'] = item['iter']
                    json_result['Solve'][keyname]['stop_reason'] = item['stop_reason']
                    json_result['Solve'][keyname]['r_norm'] = item['r_norm']
                    json_result['Solve'][keyname]['b_norm'] = item['b_norm']
                    json_result['Solve'][keyname]['relative_norm'] = item['relative_norm']

                    json_result['Solve'][keyname]['time'] = []
                    json_result['Solve'][keyname]['time'].append( item['time'] )
                elif item['processed'] == 0:
                    json_result['Solve'][keyname]['time'].append( item['time'] )

                item['processed'] = 1

            finished = True
            for keyname in self.ksp_pc:
                ksp_name, pc_name = keyname.split('-')
                if keyname not in json_result['Solve']:
                    repeat = self.batch_size
                else:
                    counts = len(json_result['Solve'][keyname]['time'])
                    repeat = self.batch_size - counts

                for _ in range(repeat):
                    self.paras.append( [ksp_name,pc_name,mat_file,yaml_file] )
                    finished = False

                if repeat == 0:
                    json_result['Solve'][keyname]['avg_time'] = sum(json_result['Solve'][keyname]['time']) / self.batch_size

            json_result['finished'] = finished

            with open(yaml_file,'w',encoding='utf-8') as f:
                yaml.dump_all(yaml_result,f)

        else:
            for keyname in self.ksp_pc:
                ksp_name, pc_name = keyname.split('-')
                for _ in range(self.batch_size):
                    self.paras.append( [ksp_name,pc_name,mat_file,yaml_file] )

    def ChangeMatFormat(self,out_file,in_mat_file,in_vec_file=None):
        csr_A = load_npz(in_mat_file)
        petsc_A = PETSc.Mat().createAIJ(size=csr_A.shape, csr=(csr_A.indptr, csr_A.indices, csr_A.data))

        if in_vec_file is None:
            b = np.ones(csr_A.shape[0])
        else:
            b = np.load(in_vec_file)
        petsc_b = PETSc.Vec().createSeq(len(b)) 
        petsc_b.setValues(range(len(b)), b) 

        viewer = PETSc.Viewer().createBinary(out_file, 'w')
        viewer(petsc_A)
        viewer(petsc_b)

    def DataAnalysis(self,json_result,idx):
        time_list = []
        for keyname in self.ksp_pc:
            if json_result['Solve'][keyname]['stop_reason'] > 0:
                time_list.append( (keyname, json_result['Solve'][keyname]['avg_time']) )

        if len(time_list) == 0:
            class_name = 'failed'
        else:
            sorted_time_list = sorted(time_list, key = itemgetter(1))
            json_result['Solve']['sorted_time'] = sorted_time_list
            class_name = sorted_time_list[0][0]


        json_result['class'] = class_name
        self.summary['total_num'] += 1
        self.summary['total_list'].append(idx)
        self.summary[class_name].append(idx)
        self.summary[class_name+'_num'] += 1
        
        with open(self.summary_file,'w',encoding='utf-8') as f:
            json.dump(self.summary,f,indent=4)

    def Process(self,mat_name_template):
        for idx in self.idx_list:
            if idx not in self.summary['total_list']:
                print('=====================================')
                print(f'begin to deal with matrix {idx}')

                json_file = os.path.join(self.json_dir,f'result{idx}.json')
                if os.path.exists(json_file):
                    with open(json_file,'r',encoding='utf-8') as f:
                        json_result = json.load(f)
                        
                    if 'finished' not in json_result:
                        json_result['finished'] = False

                    if 'Solve' not in json_result:
                        json_result['Solve'] = {}

                else:
                    json_result = {}
                    json_result['finished'] = False
                    json_result['Solve'] = {}

                mat_file = os.path.join(self.mat_dir,f'petsc{idx}.mat')
                if not os.path.exists(mat_file):
                    print('begin to change the format of matrix')
                    in_mat_file = mat_name_template.format(idx)

                    in_vec_file = os.path.dirname(in_mat_file) + f'/rhs{idx}.npy'
                    self.ChangeMatFormat(mat_file,in_mat_file,in_vec_file)
                    # self.ChangeMatFormat(mat_file,in_mat_file)

                    json_result['GenMat'] = {}
                    json_result['GenMat']['num_para'] = 0

                yaml_file = os.path.join(self.yaml_dir,f'result{idx}.yaml') 
                self.DealYamlFile(json_result,mat_file,yaml_file)

                if json_result['finished']:
                    print(f'finished solving matrix {idx}')
                    self.DataAnalysis(json_result,idx)

                with open(json_file,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)

    def GenerateScript(self,file_name,header,footer,command):
        contents = []
        if len(self.paras) == 0:
            print('Finished all computation')
        else:
            for item in self.paras:
                one_command = command.format(*item)

                keyname = f'{item[0]}-{item[1]}'
                yaml_file = item[3]
                contents.append(one_command)
                contents.append('if [ $? != 0 ]; then \n')
                contents.append(f'echo --- >> {yaml_file} \n')
                contents.append(f'echo ksp_pc: {keyname} >> {yaml_file} \n')
                contents.append(f'echo iter: 100000 >> {yaml_file} \n')
                contents.append(f'echo stop_reason: -100 >> {yaml_file} \n')
                contents.append(f'echo r_norm: 0.0 >> {yaml_file} \n')
                contents.append(f'echo b_norm: 0.0 >> {yaml_file} \n')
                contents.append(f'echo relative_norm: 0.0 >> {yaml_file} \n')
                contents.append(f'echo time: 100000.0 >> {yaml_file} \n')
                contents.append(f'echo processed: 0 >> {yaml_file} \n')
                contents.append('fi \n')

        with open(file_name,'w',encoding='utf-8') as f:
            f.writelines(header)
            f.writelines(contents)
            f.writelines(footer)

class ParTaskParRunCluster(SeqTaskParRun):
    def __init__(self,json_dir,yaml_dir,mat_dir,batch_size,idx_list,num_task):
        super().__init__(json_dir,yaml_dir,mat_dir,batch_size,idx_list)
        self.num_task = num_task

    def DealYamlFile(self,json_result,mat_file,yaml_file):
        any_yaml_file_exist = False
        for k in range(self.num_task): 
            tmp_name = yaml_file + f'{k}'

            if os.path.exists(tmp_name):
                any_yaml_file_exist = True
                print('read yaml file')

                with open(tmp_name,'r',encoding='utf-8') as f:
                    tmp = yaml.load_all(f,Loader=yaml.FullLoader)
                    yaml_result = list(tmp)

                for item in yaml_result:
                    keyname = item['ksp_pc']

                    if keyname not in json_result['Solve']:
                        json_result['Solve'][keyname] = {}
                        json_result['Solve'][keyname]['iter'] = item['iter']
                        json_result['Solve'][keyname]['stop_reason'] = item['stop_reason']
                        json_result['Solve'][keyname]['r_norm'] = item['r_norm']
                        json_result['Solve'][keyname]['b_norm'] = item['b_norm']
                        json_result['Solve'][keyname]['relative_norm'] = item['relative_norm']

                        json_result['Solve'][keyname]['time'] = []
                        json_result['Solve'][keyname]['time'].append( item['time'] )
                    elif item['processed'] == 0:
                        json_result['Solve'][keyname]['time'].append( item['time'] )

                    item['processed'] = 1

                with open(tmp_name,'w',encoding='utf-8') as f:
                    yaml.dump_all(yaml_result,f)

        if any_yaml_file_exist:
            finished = True
            for keyname in self.ksp_pc:
                ksp_name, pc_name = keyname.split('-')
                if keyname not in json_result['Solve']:
                    repeat = self.batch_size
                else:
                    counts = len(json_result['Solve'][keyname]['time'])
                    repeat = self.batch_size - counts

                for _ in range(repeat):
                    self.paras.append( [ksp_name,pc_name,mat_file,yaml_file] )
                    finished = False

                if repeat == 0:
                    json_result['Solve'][keyname]['avg_time'] = sum(json_result['Solve'][keyname]['time']) / self.batch_size

            json_result['finished'] = finished
        else:
            for keyname in self.ksp_pc:
                ksp_name, pc_name = keyname.split('-')
                for _ in range(self.batch_size):
                    self.paras.append( [ksp_name,pc_name,mat_file,yaml_file] )


    def GenerateScript(self,file_name,header,footer,command):
        contents = [ [] for i in range(self.num_task)]

        if len(self.paras) == 0:
            print('Finished all computation')
        else:
            num_run = len(self.paras)
            count_list = LoadBalance(num_run,self.num_task)

            for i in range(self.num_task):
                begin = count_list[i]
                end = count_list[i+1]
                for item in self.paras[begin:end]:
                    yaml_file = item[3] + f'{i}'
                    one_command = command.format(item[0],item[1],item[2],yaml_file)

                    keyname = f'{item[0]}-{item[1]}'
                    contents[i].append(one_command)

                    contents[i].append('if [ $? != 0 ]; then \n')
                    contents[i].append(f'echo --- >> {yaml_file} \n')
                    contents[i].append(f'echo ksp_pc: {keyname} >> {yaml_file} \n')
                    contents[i].append(f'echo iter: 100000 >> {yaml_file} \n')
                    contents[i].append(f'echo stop_reason: -100 >> {yaml_file} \n')
                    contents[i].append(f'echo r_norm: 0.0 >> {yaml_file} \n')
                    contents[i].append(f'echo b_norm: 0.0 >> {yaml_file} \n')
                    contents[i].append(f'echo relative_norm: 0.0 >> {yaml_file} \n')
                    contents[i].append(f'echo time: 100000.0 >> {yaml_file} \n')
                    contents[i].append(f'echo processed: 0 >> {yaml_file} \n')
                    contents[i].append('fi \n')

        for k in range(self.num_task):
            real_name = file_name + f'{k}'
            with open(real_name,'w',encoding='utf-8') as f:
                f.writelines(header)
                f.writelines(contents[k])
                f.writelines(footer)

def LoadBalance(num, n):
    q = num // n
    r = num % n

    begin = 0
    out = []
    for _ in range(n):
        if r > 0:
            end = begin + q + 1
            r -= 1
        else:
            end = begin + q

        out.append(begin)
        begin = end

    out.append(num)
    return out


class ParTaskParRunDesktop(ParTaskParRunCluster):
    def __init__(self,json_dir,yaml_dir,mat_dir,batch_size,idx_list,num_task,cpu_list):
        super().__init__(json_dir,yaml_dir,mat_dir,batch_size,idx_list,num_task)
        self.cpu_list = cpu_list
        num_cpu_per_task = len(self.cpu_list) // self.num_task
        self.cpu_per_task = []
        for i in range(self.num_task):
            begin = i * num_cpu_per_task
            end = (i + 1) * num_cpu_per_task
            s = ' '
            for item in self.cpu_list[begin:end-1]:
                s += f'{item},'

            s += f'{self.cpu_list[end-1]} '
            self.cpu_per_task.append( s )
            # self.cpu_per_task.append( self.cpu_list[begin:end] )

    def GenerateScript(self,file_name,header,footer,command):
        contents = [ [] for i in range(self.num_task)]

        if len(self.paras) == 0:
            print('Finished all computation')
        else:
            num_run = len(self.paras)
            count_list = LoadBalance(num_run,self.num_task)

            for i in range(self.num_task):
                begin = count_list[i]
                end = count_list[i+1]
                for item in self.paras[begin:end]:
                    yaml_file = item[3] + f'{i}'
                    one_command = command.format(self.cpu_per_task[i], item[0],item[1],item[2],yaml_file)

                    keyname = f'{item[0]}-{item[1]}'
                    contents[i].append(one_command)

                    contents[i].append('if [ $? != 0 ]; then \n')
                    contents[i].append(f'echo --- >> {yaml_file} \n')
                    contents[i].append(f'echo ksp_pc: {keyname} >> {yaml_file} \n')
                    contents[i].append(f'echo iter: 100000 >> {yaml_file} \n')
                    contents[i].append(f'echo stop_reason: -100 >> {yaml_file} \n')
                    contents[i].append(f'echo r_norm: 0.0 >> {yaml_file} \n')
                    contents[i].append(f'echo b_norm: 0.0 >> {yaml_file} \n')
                    contents[i].append(f'echo relative_norm: 0.0 >> {yaml_file} \n')
                    contents[i].append(f'echo time: 100000.0 >> {yaml_file} \n')
                    contents[i].append(f'echo processed: 0 >> {yaml_file} \n')
                    contents[i].append('fi \n')

        for k in range(self.num_task):
            real_name = file_name + f'{k}'
            with open(real_name,'w',encoding='utf-8') as f:
                f.writelines(header)
                f.writelines(contents[k])
                f.writelines(footer)
                

# class ParTaskParRun(SeqTaskParRun):
#     def __init__(self,json_dir,yaml_dir,mat_dir,batch_size,idx_list,cpu_list,num_task):
#         super().__init__(json_dir,yaml_dir,mat_dir,batch_size,idx_list)
#         self.cpu_list = cpu_list
#         self.num_task = num_task
#         num_cpu_per_task = len(self.cpu_list) // self.num_task
#         self.cpu_per_task = []
#         for i in range(self.num_task):
#             begin = i * num_cpu_per_task
#             end = (i + 1) * num_cpu_per_task
#             s = ' '
#             for item in self.cpu_list[begin:end-1]:
#                 s += f'{item},'

#             s += f'{self.cpu_list[end-1]} '
#             self.cpu_per_task.append( s )
#             # self.cpu_per_task.append( self.cpu_list[begin:end] )

#     def DealYamlFile(self,json_result,mat_file,yaml_file):
#         any_yaml_file_exist = False
#         for k in range(self.num_task): 
#             tmp_name = yaml_file + f'{k}'

#             if os.path.exists(tmp_name):
#                 any_yaml_file_exist = True
#                 print('read yaml file')

#                 with open(tmp_name,'r',encoding='utf-8') as f:
#                     tmp = yaml.load_all(f,Loader=yaml.FullLoader)
#                     yaml_result = list(tmp)

#                 for item in yaml_result:
#                     keyname = item['ksp_pc']

#                     if keyname not in json_result['Solve']:
#                         json_result['Solve'][keyname] = {}
#                         json_result['Solve'][keyname]['iter'] = item['iter']
#                         json_result['Solve'][keyname]['stop_reason'] = item['stop_reason']
#                         json_result['Solve'][keyname]['r_norm'] = item['r_norm']
#                         json_result['Solve'][keyname]['b_norm'] = item['b_norm']
#                         json_result['Solve'][keyname]['relative_norm'] = item['relative_norm']

#                         json_result['Solve'][keyname]['time'] = []
#                         json_result['Solve'][keyname]['time'].append( item['time'] )
#                     elif item['processed'] == 0:
#                         json_result['Solve'][keyname]['time'].append( item['time'] )

#                     item['processed'] = 1

#                 with open(tmp_name,'w',encoding='utf-8') as f:
#                     yaml.dump_all(yaml_result,f)

#         if any_yaml_file_exist:
#             finished = True
#             for keyname in self.ksp_pc:
#                 ksp_name, pc_name = keyname.split('-')
#                 if keyname not in json_result['Solve']:
#                     repeat = self.batch_size
#                 else:
#                     counts = len(json_result['Solve'][keyname]['time'])
#                     repeat = self.batch_size - counts

#                 for _ in range(repeat):
#                     self.paras.append( [ksp_name,pc_name,mat_file,yaml_file] )
#                     finished = False

#                 if repeat == 0:
#                     json_result['Solve'][keyname]['avg_time'] = sum(json_result['Solve'][keyname]['time']) / self.batch_size

#             json_result['finished'] = finished
#         else:
#             for keyname in self.ksp_pc:
#                 ksp_name, pc_name = keyname.split('-')
#                 for _ in range(self.batch_size):
#                     self.paras.append( [ksp_name,pc_name,mat_file,yaml_file] )


#     def GenerateScript(self,file_name,header,footer,command):
#         contents = [ [] for i in range(self.num_task)]

#         if len(self.paras) == 0:
#             print('Finished all computation')
#         else:
#             num_run = len(self.paras)
#             count_list = LoadBalance(num_run,self.num_task)

#             for i in range(self.num_task):
#                 begin = count_list[i]
#                 end = count_list[i+1]
#                 for item in self.paras[begin:end]:
#                     yaml_file = item[3] + f'{i}'
#                     one_command = command.format(self.cpu_per_task[i], item[0],item[1],item[2],yaml_file)

#                     keyname = f'{item[0]}-{item[1]}'
#                     contents[i].append(one_command)

#                     contents[i].append('if [ $? != 0 ]; then \n')
#                     contents[i].append(f'echo --- >> {yaml_file} \n')
#                     contents[i].append(f'echo ksp_pc: {keyname} >> {yaml_file} \n')
#                     contents[i].append(f'echo iter: 100000 >> {yaml_file} \n')
#                     contents[i].append(f'echo stop_reason: -100 >> {yaml_file} \n')
#                     contents[i].append(f'echo r_norm: 0.0 >> {yaml_file} \n')
#                     contents[i].append(f'echo b_norm: 0.0 >> {yaml_file} \n')
#                     contents[i].append(f'echo relative_norm: 0.0 >> {yaml_file} \n')
#                     contents[i].append(f'echo time: 100000.0 >> {yaml_file} \n')
#                     contents[i].append(f'echo processed: 0 >> {yaml_file} \n')
#                     contents[i].append('fi \n')

#         for k in range(self.num_task):
#             real_name = file_name + f'{k}'
#             with open(real_name,'w',encoding='utf-8') as f:
#                 f.writelines(header)
#                 f.writelines(contents[k])
#                 f.writelines(footer)


def TestMat1():
    indptr = np.array([0, 2, 3, 6]) # row vec
    indices = np.array([0, 2, 2, 0, 1, 2]) # col vec
    data = np.array([1, 2, 3, 4, 5, 6]) # val vec
    csr_A = csr_matrix((data, indices, indptr), shape=(3, 3))
    out_file = 'petsc.dat'


    petsc_A = PETSc.Mat().createAIJ(size=csr_A.shape, csr=(csr_A.indptr, csr_A.indices, csr_A.data))

    b = np.ones(csr_A.shape[0])
    petsc_b = PETSc.Vec().createSeq(len(b)) 
    petsc_b.setValues(range(len(b)), b) 

    viewer = PETSc.Viewer().createBinary(out_file, 'w')
    viewer(petsc_A)
    viewer(petsc_b)

def TestMat2():
    csr_A = load_npz('./csrA.npz')
    out_file = 'petsc.dat'

    petsc_A = PETSc.Mat().createAIJ(size=csr_A.shape, csr=(csr_A.indptr, csr_A.indices, csr_A.data))

    b = np.ones(csr_A.shape[0])
    petsc_b = PETSc.Vec().createSeq(len(b)) 
    petsc_b.setValues(range(len(b)), b) 

    viewer = PETSc.Viewer().createBinary(out_file, 'w')
    viewer(petsc_A)
    viewer(petsc_b)


def TestMat3():
    #csr_A = load_npz('../test1/Mat/csr1175.npz')
    #out_file = './PetscMat/petsc1175.mat'

    csr_A = load_npz('../test1/Mat/csr1068.npz')
    out_file = './PetscMat/petsc1068.mat'

    print('generate matrix:',out_file)

    petsc_A = PETSc.Mat().createAIJ(size=csr_A.shape, csr=(csr_A.indptr, csr_A.indices, csr_A.data))

    b = np.ones(csr_A.shape[0])
    petsc_b = PETSc.Vec().createSeq(len(b))
    petsc_b.setValues(range(len(b)), b)

    viewer = PETSc.Viewer().createBinary(out_file, 'w')
    viewer(petsc_A)
    viewer(petsc_b)

def TestSeqTaskParRun():
    import re

    yaml_dir = 'YamlFiles'
    json_dir = 'JsonFiles'
    mat_dir = 'PetscMat'
    batch_size = 3

    command = 'mpirun -n 2 ./rs -ksp_type {} -pc_type {} -mat_file {} -yaml_file {} \n'

    dir_name = '../test1/Mat/'
    mat_template = dir_name + 'csr{}.npz'
    files = os.listdir(dir_name)
    idx_list = []
    for item in files:
        idx = int( re.sub(r'\D','',item) )
        idx_list.append(idx)

    a = SeqTaskParRun(json_dir,yaml_dir,mat_dir,batch_size,idx_list)
    a.Process(mat_template)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

def TestParTaskParRun():
    import re

    yaml_dir = 'YamlFiles'
    json_dir = 'JsonFiles'
    mat_dir = 'PetscMat'
    batch_size = 3
    cpu_list = [0,1,2,3]
    num_task = 2

    command = 'mpirun --cpu-set {} -n 2 ./rs -ksp_type {} -pc_type {} -mat_file {} -yaml_file {} \n'

    dir_name = '../test1/Mat/'
    mat_template = dir_name + 'csr{}.npz'
    files = os.listdir(dir_name)
    idx_list = []
    for item in files:
        idx = int( re.sub(r'\D','',item) )
        idx_list.append(idx)

    a = ParTaskParRunDesktop(json_dir,yaml_dir,mat_dir,batch_size,idx_list,num_task,cpu_list)
    a.Process(mat_template)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

if __name__ == '__main__':
    # TestSeqTaskParRun()
    TestParTaskParRun()
