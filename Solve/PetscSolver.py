import os
from operator import itemgetter
import json
import yaml
from scipy.sparse import csr_matrix, load_npz
from petsc4py import PETSc
import numpy as np

import BaseSolver

class SingleTask4Petsc(BaseSolver.SingleTaskGenWithYamlJson):
    def __init__(self,metric_list,label_list,inner_para_list,batch_size=3,permutation=None,json_dir='../JsonFiles/',yaml_dir='./YamlFiles/'):
        super().__init__(metric_list,label_list,inner_para_list,batch_size,permutation,json_dir,yaml_dir)

    def ChangeMatFormat(self,idx_list,out_list,mat_list,vec_list=None):
        '''
        there is only one output in binary that includes the matrix and the vector
        '''
        len1 = len(idx_list)
        len2 = len(out_list)
        len3 = len(mat_list)
        assert len1 == len2
        assert len1 == len3

        vec_exist = True
        if vec_list is not None:
            len4 = len(vec_list)
            assert len1 == len4
        else:
            vec_exist = False

        for i in range(len1):
            idx = idx_list[i]
            mat_file = mat_list[i]
            vec_file = vec_list[i]
            out_file = out_list[i]
            if not os.path.exists(out_file):
                print(f'begin to change the format of matrix {idx}')
                csr_A = load_npz(mat_file)
                petsc_A = PETSc.Mat().createAIJ(size=csr_A.shape, csr=(csr_A.indptr, csr_A.indices, csr_A.data))

                if vec_exist:
                    b = np.ones(csr_A.shape[0])
                else:
                    b = np.load(vec_file)
                petsc_b = PETSc.Vec().createSeq(len(b)) 
                petsc_b.setValues(range(len(b)), b) 

                viewer = PETSc.Viewer().createBinary(out_file, 'w')
                viewer(petsc_A)
                viewer(petsc_b)

    def DataAnalysis(self,idx_list):
        '''
        statistic
        '''
        self.summary['failed_num'] = 0
        self.summary['failed'] = []
        for label in self.label_list:
            tmp_name = label+'_num'
            self.summary[tmp_name] = 0
            self.summary[label] = []

        for idx in idx_list:
            json_file = os.path.join(self.json_dir,f'result{idx}.json')
            with open(json_file,'r',encoding='utf-8') as f:
                json_result = json.load(f)

            time_list = []
            for label in self.label_list:
                if json_result['Solve'][label]['stop_reason'] > 0:
                    avg_time = ( json_result['Solve'][label]['time'] )/self.batch_size
                    time_list.append( (label, avg_time) )

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
            
            with open(json_file,'w',encoding='utf-8') as f:
                json.dump(json_result,f,indent=4)

        with open(self.summary_file,'w',encoding='utf-8') as f:
            json.dump(self.summary,f,indent=4)

class MultiTask4PetscInDesktop(BaseSolver.MultiTaskGenWithYamlJson):
    def __init__(self,cpu_list,num_task,metric_list,label_list,inner_para_list,batch_size=3,permutation=None,json_dir='../JsonFiles/',yaml_dir='./YamlFiles/'):
        super().__init__(num_task,metric_list,label_list,inner_para_list,batch_size,permutation,json_dir,yaml_dir)

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

    def ChangeMatFormat(self,idx_list,out_list,mat_list,vec_list=None):
        '''
        there is only one output in binary that includes the matrix and the vector
        '''
        len1 = len(idx_list)
        len2 = len(out_list)
        len3 = len(mat_list)
        assert len1 == len2
        assert len1 == len3

        vec_exist = True
        if vec_list is not None:
            len4 = len(vec_list)
            assert len1 == len4
        else:
            vec_exist = False

        for i in range(len1):
            idx = idx_list[i]
            mat_file = mat_list[i]
            vec_file = vec_list[i]
            out_file = out_list[i]
            if not os.path.exists(out_file):
                print(f'begin to change the format of matrix {idx}')
                csr_A = load_npz(mat_file)
                petsc_A = PETSc.Mat().createAIJ(size=csr_A.shape, csr=(csr_A.indptr, csr_A.indices, csr_A.data))

                if vec_exist:
                    b = np.ones(csr_A.shape[0])
                else:
                    b = np.load(vec_file)
                petsc_b = PETSc.Vec().createSeq(len(b)) 
                petsc_b.setValues(range(len(b)), b) 

                viewer = PETSc.Viewer().createBinary(out_file, 'w')
                viewer(petsc_A)
                viewer(petsc_b)

    def DataAnalysis(self,idx_list):
        for idx in idx_list:
            json_file = os.path.join(self.json_dir,f'result{idx}.json')
            with open(json_file,'r',encoding='utf-8') as f:
                json_result = json.load(f)

    def GenerateScript(self,file_name,header,footer,command):
        contents = [ [] for i in range(self.num_task)]
        total_num = len(self.all_para_list)

        if total_num == 0:
            print('Finished all computation')
        else:
            count_list = BaseSolver.LoadBalance(total_num,self.num_task)

            for i in range(self.num_task):
                begin = count_list[i]
                end = count_list[i+1]
                for j in range(begin,end):
                    para = self.all_para_list[j]
                    label = self.auxiliary_label_list[j]

                    # add suffix 'i' to the name of yaml file according to the task index 
                    yaml_file = para[-1] + f'{i}'
                    para[-1] = yaml_file

                    # move the cpu index to the first place by permutation,
                    # change the cpu index according to the task index
                    para[0] = self.cpu_per_task[i]

                    one_command = command.format(*para)

                    contents[i].append(one_command)
                    contents[i].append('if [ $? != 0 ]; then \n')
                    contents[i].append(f'echo --- >> {yaml_file} \n')
                    contents[i].append(f'echo solve_label: {label} >> {yaml_file} \n')
                    for metric in self.metric_list:
                        contents[i].append(f'echo {metric}: -100 >> {yaml_file} \n')
                    contents[i].append('fi \n')

        for k in range(self.num_task):
            real_name = file_name + f'{k}'
            with open(real_name,'w',encoding='utf-8') as f:
                f.writelines(header)
                f.writelines(contents[k])
                f.writelines(footer)


def TestMultiTask():
    yaml_dir = './YamlFiles'
    json_dir = './JsonFiles'
    batch_size = 1
    cpu_list = [20,21,22,23]
    num_task = 4
    metric_list = ['iter','stop_reason','r_norm','b_norm','relative_norm','time']
    ksp = 'gmres'
    inner_para_list = [[] for i in range(99)]
    label_list = []
    for i in range(99):
        label = ksp+str(i+1)
        label_list.append(label)
        inner_para_list[i].append( (i+1)/100 )
        inner_para_list[i].append(label)


    idx_list = list(range(10))
    outer_para_list = [[] for i in range(len(idx_list))]
    yaml_template = 'result{}.yaml'
    mat_list = []
    vec_list = []
    mat_dir = '../MatData/'
    mat_template = 'scipy_csr{}.npz'
    vec_template = 'b{}.npy'
    formated_mat_dir = './PetscMat'
    os.makedirs(formated_mat_dir,exist_ok=True)
    out_template = 'petsc{}.dat'
    out_list = []
    for i,idx in enumerate(idx_list):
        mat_path = os.path.join(mat_dir,mat_template.format(idx))
        mat_list.append(mat_path)
        
        vec_path = os.path.join(mat_dir,vec_template.format(idx)) 
        vec_list.append(vec_path)

        # placeholder
        outer_para_list[i].append(' ')

        formated_mat_path = os.path.join(formated_mat_dir,out_template.format(idx))
        outer_para_list[i].append(formated_mat_path)
        out_list.append(formated_mat_path)

        yaml_path = os.path.join(yaml_dir,yaml_template.format(idx))
        outer_para_list[i].append(yaml_path)


    # move the first parameer of outer_para_list to the first place
    permutation = [1,2,0,3,4]

    a = MultiTask4PetscInDesktop(cpu_list,num_task,metric_list,label_list,inner_para_list,batch_size=batch_size,permutation=permutation,json_dir=json_dir)

    # pre-process
    # a.ChangeMatFormat(idx_list,out_list,mat_list,vec_list)

    a.Process(idx_list,outer_para_list)

    # generate script
    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    command = 'mpirun --cpu-set {} -n 1 ./rs -ksp_type gmres -pc_type hypre -pc_hypre_boomeramg_stong_threshold {} -solve_label {}  -mat_file {} -yaml_file {} \n'
    a.GenerateScript(script_file,header,footer,command)

if __name__ == '__main__':
    TestMultiTask()
