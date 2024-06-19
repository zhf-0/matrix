import os
import json
import yaml
import scipy
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from operator import itemgetter

import BaseSolver

class SingleTask4Fasp(BaseSolver.SingleTaskGenWithYamlJson):
    def __init__(self,metric_list,label_list,inner_para_list,batch_size=3,permutation=None,json_dir='../JsonFiles/',yaml_dir='./YamlFiles/'):
        super().__init__(metric_list,label_list,inner_para_list,batch_size,permutation,json_dir,yaml_dir)

    def ChangeMatFormat(self,out_mat_list,out_vec_list,in_mat_list,in_vec_list=None):
        '''
        the format of the input matrix is scipy.sparse.csr_matrix
        the format of the input vector is numpy
        the format of the output matrix is mtx
        the format of the output vector is txt
        '''
        len1 = len(out_mat_list)
        len2 = len(out_vec_list)
        len3 = len(in_mat_list)
        assert len1 == len2
        assert len1 == len3

        vec_exist = True
        if in_vec_list is not None:
            len4 = len(in_vec_list)
            assert len1 == len4
        else:
            vec_exist = False

        for i in range(len1):
            in_mat_file = in_mat_list[i]
            out_mat_file = out_mat_list[i]
            out_vec_file = out_vec_list[i]
            if not os.path.exists(out_mat_file):
                print(f'begin to change the format of matrix: {in_mat_file}')
                csr_A = load_npz(in_mat_file)
                scipy.io.mmwrite(out_mat_file,csr_A.tocoo())

                if vec_exist:
                    vec_file = in_vec_list[i]
                    b = np.load(vec_file)
                else:
                    b = np.ones(csr_A.shape[0])

                vec = b.tolist()
                with open(out_vec_file,'w',encoding='utf-8') as f:
                    f.write(f'{b.shape[0]} \n')
                    for line in vec:
                        f.write(str(line)+ '\n')

    def DataAnalysis(self,idx_list):
        print('==============================================')
        print('begin to analysis data')

        self.summary['failed_num'] = 0
        self.summary['failed'] = []
        for label in self.label_list:
            tmp_name = label+'_num'
            self.summary[tmp_name] = 0
            self.summary[label] = []

        for idx in idx_list:
            if idx not in self.summary['total_list']:
                print('the matrix has not been solved:',idx)
            else:
                json_file = os.path.join(self.json_dir,f'result{idx}.json')
                with open(json_file,'r',encoding='utf-8') as f:
                    json_result = json.load(f)

                time_list = []
                for label in self.label_list:
                    if json_result['Solve'][label]['iter'] > 0:
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

class MultiTask4Fasp(BaseSolver.MultiTaskGenWithYamlJson):
    def __init__(self,num_task,metric_list,label_list,inner_para_list,batch_size=3,permutation=None,json_dir='../JsonFiles/',yaml_dir='./YamlFiles/'):
        super().__init__(num_task,metric_list,label_list,inner_para_list,batch_size,permutation,json_dir,yaml_dir)


    def ChangeMatFormat(self,out_mat_list,out_vec_list,in_mat_list,in_vec_list=None):
        '''
        the format of the input matrix is scipy.sparse.csr_matrix
        the format of the input vector is numpy
        the format of the output matrix is mtx
        the format of the output vector is txt
        '''
        len1 = len(out_mat_list)
        len2 = len(out_vec_list)
        len3 = len(in_mat_list)
        assert len1 == len2
        assert len1 == len3

        vec_exist = True
        if in_vec_list is not None:
            len4 = len(in_vec_list)
            assert len1 == len4
        else:
            vec_exist = False

        for i in range(len1):
            in_mat_file = in_mat_list[i]
            out_mat_file = out_mat_list[i]
            out_vec_file = out_vec_list[i]
            if not os.path.exists(out_mat_file):
                print(f'begin to change the format of matrix: {in_mat_file}')
                csr_A = load_npz(in_mat_file)
                scipy.io.mmwrite(out_mat_file,csr_A.tocoo())

                if vec_exist:
                    vec_file = in_vec_list[i]
                    b = np.load(vec_file)
                else:
                    b = np.ones(csr_A.shape[0])

                vec = b.tolist()
                with open(out_vec_file,'w',encoding='utf-8') as f:
                    f.write(f'{b.shape[0]} \n')
                    for line in vec:
                        f.write(str(line)+ '\n')


    def DataAnalysis(self,idx_list):
        import re
        from operator import itemgetter

        print('==============================================')
        print('begin to analysis data')
        self.summary['analysis'] = {}
        self.summary['analysis']['0.25_time'] = []
        self.summary['analysis']['0.25_iter'] = []
        self.summary['analysis']['0.50_time'] = []
        self.summary['analysis']['0.50_iter'] = []
        self.summary['analysis']['min_time'] = []
        self.summary['analysis']['min_iter'] = []

        for idx in idx_list:
            if idx not in self.summary['total_list']:
                print('the matrix has not been solved:',idx)
            else:
                json_file = os.path.join(self.json_dir,f'result{idx}.json')
                with open(json_file,'r',encoding='utf-8') as f:
                    json_result = json.load(f)

                json_result['Solve']['analysis'] = {}
                time_list = []
                iter_list = []
                for label in self.label_list:
                    if json_result['Solve'][label]['iter'][0] > 0:
                        avg_time = sum(json_result['Solve'][label]['time'])/self.batch_size
                        time_list.append( (label,avg_time) )

                        avg_iter = sum(json_result['Solve'][label]['iter'])/self.batch_size
                        iter_list.append( (label,avg_iter) )

                    if re.search('25',label):
                        json_result['Solve']['analysis']['0.25_time'] = sum(json_result['Solve'][label]['time'])/self.batch_size
                        self.summary['analysis']['0.25_time'].append(sum(json_result['Solve'][label]['time'])/self.batch_size)

                        json_result['Solve']['analysis']['0.25_iter'] = sum(json_result['Solve'][label]['iter'])/self.batch_size
                        self.summary['analysis']['0.25_iter'].append(sum(json_result['Solve'][label]['iter'])/self.batch_size)
                        if json_result['Solve'][label]['iter'][0] > 0:
                            json_result['Solve']['analysis']['0.25_succ'] = True
                        else:
                            json_result['Solve']['analysis']['0.25_succ'] = False

                    if re.search('50',label):
                        json_result['Solve']['analysis']['0.50_time'] = sum(json_result['Solve'][label]['time'])/self.batch_size
                        self.summary['analysis']['0.50_time'].append(sum(json_result['Solve'][label]['time'])/self.batch_size)
                        json_result['Solve']['analysis']['0.50_iter'] = sum(json_result['Solve'][label]['iter'])/self.batch_size
                        self.summary['analysis']['0.50_iter'].append(sum(json_result['Solve'][label]['iter'])/self.batch_size)
                        if json_result['Solve'][label]['iter'][0] > 0:
                            json_result['Solve']['analysis']['0.50_succ'] = True
                        else:
                            json_result['Solve']['analysis']['0.50_succ'] = False

                if len(time_list) == 0:
                    json_result['Solve']['analysis']['min_time_label'] = None
                    self.summary['analysis']['min_time'].append(-1)
                else:
                    sorted_time_list = sorted(time_list, key = itemgetter(1))
                    json_result['Solve']['analysis']['min_time_label'] = sorted_time_list[0][0]
                    json_result['Solve']['analysis']['min_time'] = sorted_time_list[0][1]
                    self.summary['analysis']['min_time'].append(sorted_time_list[0][1])
                    
                if len(iter_list) == 0:
                    json_result['Solve']['analysis']['min_iter_label'] = None
                    self.summary['analysis']['min_iter'].append(-1)
                else:
                    sorted_iter_list = sorted(iter_list, key = itemgetter(1))
                    json_result['Solve']['analysis']['min_iter_label'] = sorted_iter_list[0][0]
                    json_result['Solve']['analysis']['min_iter'] = sorted_iter_list[0][1]
                    self.summary['analysis']['min_iter'].append(sorted_iter_list[0][1])
                    
                with open(json_file,'w',encoding='utf-8') as f:
                    json.dump(json_result,f,indent=4)

        with open(self.summary_file,'w',encoding='utf-8') as f:
            json.dump(self.summary,f,indent=4)


def TestMultiTask():
    yaml_dir = './YamlFiles'
    yaml_template = 'result{}.yaml'
    json_dir = './JsonFiles'
    batch_size = 1
    num_task = 4
    metric_list = ['iter','r_norm','b_norm','relative_norm','time']
    ksp = 'gmres'
    inner_para_list = [[] for i in range(49)]
    label_list = []
    for i in range(49):
        label = ksp+str(i+1)
        label_list.append(label)
        inner_para_list[i].append( (i+1)/100 )
        inner_para_list[i].append(label)


    idx_list = list(range(10))
    outer_para_list = [[] for i in range(len(idx_list))]

    in_mat_template = '../MatData/scipy_csr{}.npz'
    in_vec_template = '../MatData/b{}.npy'
    in_mat_list = []
    in_vec_list = []

    os.makedirs('./MtxMat',exist_ok=True)
    out_mat_template = './MtxMat/mat{}.mtx'
    out_vec_template = './MtxMat/vec{}.txt'
    out_mat_list = []
    out_vec_list = []

    for i,idx in enumerate(idx_list):
        in_mat_file = in_mat_template.format(idx)
        in_mat_list.append(in_mat_file)
        
        in_vec_file = in_vec_template.format(idx) 
        in_vec_list.append(in_vec_file)


        out_mat_file = out_mat_template.format(idx)
        out_mat_list.append(out_mat_file)
        outer_para_list[i].append(out_mat_file)

        out_vec_file = out_vec_template.format(idx)
        out_vec_list.append(out_vec_file)
        outer_para_list[i].append(out_vec_file)

        yaml_file = os.path.join(yaml_dir,yaml_template.format(idx))
        outer_para_list[i].append(yaml_file)



    a = MultiTask4Fasp(num_task,metric_list,label_list,inner_para_list,batch_size=batch_size,json_dir=json_dir)

    # pre-process
    # a.ChangeMatFormat(idx_list,out_list,mat_list,vec_list)

    a.Process(idx_list,outer_para_list)

    # generate script
    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    command = './fasp -ini amg_ua.dat -couple {} -solve_label {} -mat_file {} -vec_file {} -yaml_file {} \n'
    a.GenerateScript(script_file,header,footer,command)

    a.DataAnalysis(idx_list)

if __name__ == '__main__':
    TestMultiTask()
