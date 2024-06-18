import os
import json
import yaml

class TaskGen:
    def __init__(self,label_list,inner_para_list,batch_size=3,permutation=None):
        '''
        label_list and inner_para_list: for example, there are two inner paramerters pa and pb,
                  the value of pa can be 0 or 1, the value of pb can be 2 or 3.

                  inner_para_list is the list that contains all combinations of pa and pb, 
                  which means inner_para_list = [[0, 2],
                                                 [0, 3],
                                                 [1, 2],
                                                 [1, 3]]
                  each combination should have a unique label and the type of the label is str,
                  such as label_list = ['a', 'b', 'c', 'd']. 

        batch_size: the number of times the same program executes, default batch_size = 3

        permutation: usually the parameter order of one command is inner parameters first, then 
                     come with the outer parameters. But sometimes some parameters in outer 
                     parameters should be ahead of the inner parameters. This parameter can be
                     used to change the order of all parameters in one command.

                     the type of permutation is list, and the list stores new index of each 
                     parameter. For example there are 4 parameters, the first two are inner 
                     paremerters, and the last two are outer parameters. If permutation = [1,2,0,3],
                     then it means move the third parameter to the first place.

        '''
        assert len(label_list) == len(inner_para_list)
        self.num_label = len(label_list)
        self.label_list = label_list
        self.inner_para_list = inner_para_list

        self.batch_size = batch_size

        self.permutation = permutation

        # it's used to store all parameters for all problems
        self.all_para_list = []

        # the length of those two list should be equal to the length of self.all_para_list
        # those two lists store the index and label of the item in self.all_para_list  
        self.auxiliary_idx_list = []
        self.auxiliary_label_list = []

        current_path = os.path.dirname(__file__)
        self.summary_file = os.path.join(current_path,'summary4solver.json') 
        self.InitSummary()

    def InitSummary(self):
        if os.path.exists(self.summary_file):
            with open(self.summary_file,'r',encoding='utf-8') as f:
                self.summary = json.load(f)
        else:
            self.summary = {}
            self.summary['total_num'] = 0
            self.summary['total_list'] = []
            
    def ProcessOneProblem(self,idx,one_outer_para):
        result_file = one_outer_para[-1]
        finished = True

        if self.permutation is not None:
            num_all_para = len(self.inner_para_list[0]) + len(one_outer_para)
            assert len(self.permutation) == num_all_para 
            tmp_list = [None] * num_all_para

        for i in range(self.num_label):
            label = self.label_list[i]
            for j in range(self.batch_size):
                result_dir = os.path.dirname(result_file)
                result_name = os.path.basename(result_file)
                prefix_name = f'{label}-{j}-'+result_name
                prefix_result_file = os.path.join(result_dir,prefix_name)
                if not os.path.exists(prefix_result_file):
                    one_outer_para[-1] = prefix_result_file
                    finished = False

                    all_para = self.inner_para_list[i].copy()
                    all_para.extend( one_outer_para )

                    # begin to change order 
                    if self.permutation is not None:
                        tmp_list[:] = all_para[:]
                        for k in range(num_all_para):
                            new_idx = self.permutation[k]
                            all_para[new_idx] = tmp_list[k]

                    self.all_para_list.append( all_para )
                    self.auxiliary_idx_list.append( idx )
                    self.auxiliary_label_list.append( label )

        if finished:
            self.summary['total_num'] += 1
            self.summary['total_list'].append(idx)

    def Process(self,idx_list,outer_para_list):
        ''''
        idx_list: the index of problem needed to be dealt with, and the value can be discontinued
                  such as [0,1,2,10,20]

        outer_para_list: the list of parameters that are different for different problems.
                  note that the last item in outer_para_list must be the result file path, 
                  because the function will add prefix to the the result file path accoring
                  to the batch size and inner parameters
        '''
        num = len(idx_list)
        for k in range(num):
            idx = idx_list[k]
            if idx not in self.summary['total_list']:
                print('=====================================')
                print(f'begin to deal with problem {idx}')
                one_outer_para = outer_para_list[k]
                self.ProcessOneProblem(idx,one_outer_para)

        with open(self.summary_file,'w',encoding='utf-8') as f:
            json.dump(self.summary,f,indent=4)

    def GenerateScript(self,file_name,header,footer,command,num_task=1):
        '''
        num_task: the number of tasks that running parallel, each task includes a part of commands
        '''
        contents = []
        total_num = len(self.all_para_list)
        
        if total_num == 0:
            print('Finished all computation')

            with open(file_name,'w',encoding='utf-8') as f:
                f.writelines(header)
                f.writelines(contents)
                f.writelines(footer)
        elif num_task == 1:
            for i in range(total_num):
                para = self.all_para_list[i]
                idx = self.auxiliary_idx_list[i]
                label = self.auxiliary_label_list[i]
                one_command = command.format(*para)

                contents.append(one_command)
                contents.append('if [ $? != 0 ]; then \n')
                contents.append(f'echo SolveError {idx} {label}: solving failed \n')
                contents.append('fi \n')

            with open(file_name,'w',encoding='utf-8') as f:
                f.writelines(header)
                f.writelines(contents)
                f.writelines(footer)
        elif num_task > 1:
            contents = [ [] for i in range(num_task)]
            count_list = LoadBalance(total_num,num_task)
            for i in range(num_task):
                begin = count_list[i]
                end = count_list[i+1]
                for j in range(begin,end):
                    para = self.all_para_list[j]
                    idx = self.auxiliary_idx_list[j]
                    label = self.auxiliary_label_list[j]
                    one_command = command.format(*para)

                    contents[i].append(one_command)
                    contents[i].append('if [ $? != 0 ]; then \n')
                    contents[i].append(f'echo SolveError {idx} {label}: solving failed \n')
                    contents[i].append('fi \n')

            for k in range(num_task):
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


class SingleTaskGenWithYamlJson(TaskGen):
    def __init__(self,metric_list,label_list,inner_para_list,batch_size=3,permutation=None,json_dir='../JsonFiles/',yaml_dir='./YamlFiles/'):
        '''
        metric_list: the list of metrics that used in solver, such as time, number of iteration.
                     the type of metric is str, and the name of the metric should be the same as 
                     the one used in the yaml file
        
        json_dir: the dir that contains json files of each matrix, each json file includes all 
                  solving results of one matrix. Default json_dir = '../JsonFiles/', the dir can 
                  either be an absolute path or relative path

        yaml_dir: the dir that contains yaml files, each yaml file store the solving results
                  temporally. Default yaml_dir = './YamlFiles/', the dir can either be an absolute
                  path or relative path
        
        WARNING: the inner_para_list must include the label of the parameter combination. The label
                 is used in the output of the solving program with the name 'solve_label' to 
                 determine the belonging of the solving results.

        # formated_mat_dir: the dir that store the format-transformed matrices and vectors
        #           default formated_mat_dir = './FormatedMat/'
        #           the dir can either be an absolute path or relative path
        
        # result_dir: the dir that contains solving result files, default result_dir = './ResultFiles/',
        #             the dir can either be an absolute path or relative path
        
        '''
        super().__init__(label_list,inner_para_list,batch_size,permutation)

        current_path = os.path.dirname(__file__)
        if os.path.isabs(json_dir):
            self.json_dir = json_dir
        else:
            self.json_dir = os.path.join(current_path,json_dir)
        os.makedirs(self.json_dir,exist_ok=True)

        if os.path.isabs(yaml_dir):
            self.yaml_dir = yaml_dir
        else:
            self.yaml_dir = os.path.join(current_path,yaml_dir)
        os.makedirs(self.yaml_dir,exist_ok=True)

        self.metric_list = metric_list

    def CheckJsonFile(self,idx,json_result):
        for label in self.label_list:
            if label not in json_result['Solve']:
                # raise ValueError(f'JsonError {idx} {label}: the label does not exist')
                print(f'JsonError {idx} {label}: the label does not exist')
            else:
                for metric in self.metric_list:
                    count = len(json_result['Solve'][label][metric])
                    if count != self.batch_size:
                        # raise ValueError(f'JsonError {idx} {label}: the count is not equal the batch size')
                        print(f'JsonError {idx} {label}: the count is not equal to the batch size')

    def ProcessOneProblem(self,idx,one_outer_para,json_result):
        yaml_file = one_outer_para[-1]

        if self.permutation is not None:
            num_all_para = len(self.inner_para_list[0]) + len(one_outer_para)
            assert len(self.permutation) == num_all_para 
            tmp_list = [None] * num_all_para

        if os.path.exists(yaml_file):
            print('read yaml file')

            with open(yaml_file,'r',encoding='utf-8') as f:
                tmp = yaml.load_all(f,Loader=yaml.FullLoader)
                yaml_result = list(tmp)

            for item in yaml_result:
                label = item['solve_label']

                if label not in json_result['Solve']:
                    json_result['Solve'][label] = {}
                    for metric in self.metric_list:
                        json_result['Solve'][label][metric] = []   
                        json_result['Solve'][label][metric].append( item[metric] )   
                    item['processed'] = 1
                elif item['processed'] == 0:
                    for metric in self.metric_list:
                        json_result['Solve'][label][metric].append( item[metric] )   
                    item['processed'] = 1
            
            finished = True
            for i,label in enumerate(self.label_list):
                if label not in json_result['Solve']:
                    repeat = self.batch_size
                else:
                    # choose one metric to get the counts, the first one is used here
                    metric = self.metric_list[0]
                    counts = len(json_result['Solve'][label][metric])
                    if counts > self.batch_size:
                        raise ValueError(f'CountError {idx} {label}: the count is larger than batch size')

                    repeat = self.batch_size - counts

                all_para = self.inner_para_list[i].copy()
                all_para.extend( one_outer_para )
                # begin to change order 
                if self.permutation is not None:
                    tmp_list[:] = all_para[:]
                    for k in range(num_all_para):
                        new_idx = self.permutation[k]
                        all_para[new_idx] = tmp_list[k]

                for _ in range(repeat):
                    # deep copy all_para in case of modification 
                    self.all_para_list.append( all_para.copy() )
                    self.auxiliary_idx_list.append( idx )
                    self.auxiliary_label_list.append( label )
                    finished = False

            json_result['Solve']['finished'] = finished

            # save variable to the file in the last, in case there are errors 
            with open(yaml_file,'w',encoding='utf-8') as f:
                yaml.dump_all(yaml_result,f)
        else:
            finished = False
            for i,label in enumerate(self.label_list):
                all_para = self.inner_para_list[i].copy()
                all_para.extend( one_outer_para )
                # begin to change order 
                if self.permutation is not None:
                    tmp_list[:] = all_para[:]
                    for k in range(num_all_para):
                        new_idx = self.permutation[k]
                        all_para[new_idx] = tmp_list[k]

                for _ in range(self.batch_size):
                    self.all_para_list.append( all_para.copy() )
                    self.auxiliary_idx_list.append( idx )
                    self.auxiliary_label_list.append( label )

        if finished:
            self.summary['total_num'] += 1
            self.summary['total_list'].append(idx)

            # check the content of json file before saving, and the error would not cause exit
            # the error should be verified by hand 
            self.CheckJsonFile(idx,json_result)

    def Process(self,idx_list,outer_para_list):
        ''''
        idx_list: the list of matrix index needed to be dealt with,
                  such as [0,1,2,10,20]

        outer_para_list: the list of parameters that are different for different matrices,
                  note that the outer_para_list dosen't include yaml file path! 
        '''
        num = len(idx_list)
        for k in range(num):
            idx = idx_list[k]
            if idx not in self.summary['total_list']:
                print('=====================================')
                print(f'begin to deal with matrix {idx}')

                json_file = os.path.join(self.json_dir,f'result{idx}.json')
                if os.path.exists(json_file):
                    with open(json_file,'r',encoding='utf-8') as f:
                        json_result = json.load(f)
                        
                    if 'Solve' not in json_result:
                        json_result['Solve'] = {}

                    if 'finished' not in json_result:
                        json_result['Solve']['finished'] = False

                else:
                    json_result = {}
                    json_result['Solve'] = {}
                    json_result['Solve']['finished'] = False

                if json_result['Solve']['finished']:
                    self.summary['total_num'] += 1
                    self.summary['total_list'].append(idx)
                else:
                    one_outer_para = outer_para_list[k]
                    self.ProcessOneProblem(idx,one_outer_para,json_result)

                    with open(json_file,'w',encoding='utf-8') as f:
                        json.dump(json_result,f,indent=4)


        with open(self.summary_file,'w',encoding='utf-8') as f:
            json.dump(self.summary,f,indent=4)

    def GenerateScript(self,file_name,header,footer,command):
        contents = []
        total_num = len(self.all_para_list)

        for i in range(total_num):
            para = self.all_para_list[i]
            label = self.auxiliary_label_list[i]
            yaml_file = para[-1]
            one_command = command.format(*para)

            contents.append(one_command)
            contents.append('if [ $? != 0 ]; then \n')
            contents.append(f'echo --- >> {yaml_file} \n')
            contents.append(f'echo solve_label: {label} >> {yaml_file} \n')
            for metric in self.metric_list:
                contents.append(f'echo {metric}: 0 >> {yaml_file} \n')
            contents.append(f'echo processed: 0 >> {yaml_file} \n')
            contents.append('fi \n')

        with open(file_name,'w',encoding='utf-8') as f:
            f.writelines(header)
            f.writelines(contents)
            f.writelines(footer)

class MultiTaskGenWithYamlJson(SingleTaskGenWithYamlJson):
    def __init__(self,num_task,metric_list,label_list,inner_para_list,batch_size=3,permutation=None,json_dir='../JsonFiles/',yaml_dir='./YamlFiles/'):
        super().__init__(metric_list,label_list,inner_para_list,batch_size,permutation,json_dir,yaml_dir)
        self.num_task = num_task

    def ProcessOneProblem(self,idx,one_outer_para,json_result):
        yaml_file = one_outer_para[-1]

        if self.permutation is not None:
            num_all_para = len(self.inner_para_list[0]) + len(one_outer_para)
            assert len(self.permutation) == num_all_para 
            tmp_list = [None] * num_all_para

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
                    label = item['solve_label']

                    if label not in json_result['Solve']:
                        json_result['Solve'][label] = {}
                        for metric in self.metric_list:
                            json_result['Solve'][label][metric] = []   
                            json_result['Solve'][label][metric].append( item[metric] )   
                        item['processed'] = 1
                    elif item['processed'] == 0:
                        for metric in self.metric_list:
                            json_result['Solve'][label][metric].append( item[metric] )   
                        item['processed'] = 1

                with open(tmp_name,'w',encoding='utf-8') as f:
                    yaml.dump_all(yaml_result,f)

        if any_yaml_file_exist:
            if self.permutation is not None:
                num_all_para = len(self.inner_para_list[0]) + len(one_outer_para)
                assert len(self.permutation) == num_all_para 
                tmp_list = [None] * num_all_para

            finished = True
            for i,label in enumerate(self.label_list):
                if label not in json_result['Solve']:
                    repeat = self.batch_size
                else:
                    # choose one metric to get the counts, the first one is used here
                    metric = self.metric_list[0]
                    counts = len(json_result['Solve'][label][metric])
                    if counts > self.batch_size:
                        raise ValueError(f'CountError {idx} {label}: the count is larger than batch size')

                    repeat = self.batch_size - counts

                all_para = self.inner_para_list[i].copy()
                all_para.extend( one_outer_para )
                # begin to change order 
                if self.permutation is not None:
                    tmp_list[:] = all_para[:]
                    for k in range(num_all_para):
                        new_idx = self.permutation[k]
                        all_para[new_idx] = tmp_list[k]

                for _ in range(repeat):
                    # deep copy all_para in case of modification 
                    self.all_para_list.append( all_para.copy() )
                    self.auxiliary_idx_list.append( idx )
                    self.auxiliary_label_list.append( label )
                    finished = False

            json_result['Solve']['finished'] = finished
        else:
            finished = False
            for i,label in enumerate(self.label_list):
                all_para = self.inner_para_list[i].copy()
                all_para.extend( one_outer_para )
                # begin to change order 
                if self.permutation is not None:
                    tmp_list[:] = all_para[:]
                    for k in range(num_all_para):
                        new_idx = self.permutation[k]
                        all_para[new_idx] = tmp_list[k]

                for _ in range(self.batch_size):
                    self.all_para_list.append( all_para.copy() )
                    self.auxiliary_idx_list.append( idx )
                    self.auxiliary_label_list.append( label )


        if finished:
            self.summary['total_num'] += 1
            self.summary['total_list'].append(idx)

            # check the content of json file before saving, and the error would not cause exit
            # the error should be verified by hand 
            self.CheckJsonFile(idx,json_result)

    def GenerateScript(self,file_name,header,footer,command):
        contents = [ [] for i in range(self.num_task)]
        total_num = len(self.all_para_list)

        if total_num == 0:
            print('Finished all computation')
        else:
            count_list = LoadBalance(total_num,self.num_task)

            for i in range(self.num_task):
                begin = count_list[i]
                end = count_list[i+1]
                for j in range(begin,end):
                    para = self.all_para_list[j]
                    label = self.auxiliary_label_list[j]

                    # add suffix 'i' to the name of yaml file  
                    yaml_file = para[-1] + f'{i}'
                    para[-1] = yaml_file

                    one_command = command.format(*para)

                    contents[i].append(one_command)
                    contents[i].append('if [ $? != 0 ]; then \n')
                    contents[i].append(f'echo --- >> {yaml_file} \n')
                    contents[i].append(f'echo solve_label: {label} >> {yaml_file} \n')
                    for metric in self.metric_list:
                        contents[i].append(f'echo {metric}: 0 >> {yaml_file} \n')
                    contents.append(f'echo processed: 0 >> {yaml_file} \n')
                    contents[i].append('fi \n')

        for k in range(self.num_task):
            real_name = file_name + f'{k}'
            with open(real_name,'w',encoding='utf-8') as f:
                f.writelines(header)
                f.writelines(contents[k])
                f.writelines(footer)


def TestTaskGen():
    label_list = ['a','b','c','d']
    inner_para_list = [[0,2],[0,3],[1,2],[1,3]]
    idx_list = [0,4,8]
    outer_para_list = [['mat0.txt','vec0.txt','output0.txt'],['mat4.txt','vec4.txt','output4.txt'],['mat8.txt','vec8.txt','output8.txt']]

    command = './run -pa {} -pb {} -mat_file {} -vec_file {} -out_file {} \n'


    a = TaskGen(label_list,inner_para_list)
    a.Process(idx_list,outer_para_list)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

def TestTaskGenPermutation():
    permutation = [2,3,0,1,4]
    label_list = ['a','b','c','d']
    inner_para_list = [[0,2],[0,3],[1,2],[1,3]]
    idx_list = [0,4,8]
    outer_para_list = [['mat0.txt','vec0.txt','output0.txt'],['mat4.txt','vec4.txt','output4.txt'],['mat8.txt','vec8.txt','output8.txt']]

    command = './run -pa {} -pb {} -mat_file {} -vec_file {} -out_file {} \n'


    a = TaskGen(label_list,inner_para_list,permutation=permutation)
    a.Process(idx_list,outer_para_list)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

def TestTaskGenMultiTask():
    label_list = ['a','b','c','d']
    inner_para_list = [[0,2],[0,3],[1,2],[1,3]]
    idx_list = [0,4,8]
    outer_para_list = [['mat0.txt','vec0.txt','output0.txt'],['mat4.txt','vec4.txt','output4.txt'],['mat8.txt','vec8.txt','output8.txt']]

    command = './run -pa {} -pb {} -mat_file {} -vec_file {} -out_file {} \n'


    a = TaskGen(label_list,inner_para_list)
    a.Process(idx_list,outer_para_list)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command,2)

def TestSingleTaskGen():
    label_list = ['a','b','c','d']
    inner_para_list = [[0,2],[0,3],[1,2],[1,3]]

    # add label into the inner_para_list
    for i,item in enumerate(inner_para_list):
        item.append(label_list[i])
        
    idx_list = [0,4,8]
    outer_para_list = [['mat0.txt','vec0.txt','result0.yaml'],['mat4.txt','vec4.txt','result4.yaml'],['mat8.txt','vec8.txt','result8.yaml']]
    metric_list = ['time','iter','resi']

    command = './run -pa {} -pb {} -solve_label {} -mat_file {} -vec_file {} -out_file {} \n'

    a = SingleTaskGenWithYamlJson(metric_list,label_list,inner_para_list)
    a.Process(idx_list,outer_para_list)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

def TestSingleTaskGenPermutation():
    permutation = [1,3,0,2,4,5]
    label_list = ['a','b','c','d']
    inner_para_list = [[0,2],[0,3],[1,2],[1,3]]

    # add label into the inner_para_list
    for i,item in enumerate(inner_para_list):
        item.append(label_list[i])
        
    idx_list = [0,4,8]
    outer_para_list = [['mat0.txt','vec0.txt','result0.yaml'],['mat4.txt','vec4.txt','result4.yaml'],['mat8.txt','vec8.txt','result8.yaml']]
    metric_list = ['time','iter','resi']

    command = './run -pa {} -pb {} -solve_label {} -mat_file {} -vec_file {} -out_file {} \n'

    a = SingleTaskGenWithYamlJson(metric_list,label_list,inner_para_list,permutation=permutation)
    a.Process(idx_list,outer_para_list)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)


def TestMultiTaskGen():
    num_task = 2
    label_list = ['a','b','c','d']
    inner_para_list = [[0,2],[0,3],[1,2],[1,3]]

    # add label into the inner_para_list
    for i,item in enumerate(inner_para_list):
        item.append(label_list[i])
        
    idx_list = [0,4,8]
    outer_para_list = [['mat0.txt','vec0.txt','result0.yaml'],['mat4.txt','vec4.txt','result4.yaml'],['mat8.txt','vec8.txt','result8.yaml']]
    metric_list = ['time','iter','resi']

    command = './run -pa {} -pb {} -solve_label {} -mat_file {} -vec_file {} -out_file {} \n'

    a = MultiTaskGenWithYamlJson(num_task,metric_list,label_list,inner_para_list)
    a.Process(idx_list,outer_para_list)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

def TestMultiTaskGenPermutation():
    permutation = [1,3,0,2,4,5]
    num_task = 2
    label_list = ['a','b','c','d']
    inner_para_list = [[0,2],[0,3],[1,2],[1,3]]

    # add label into the inner_para_list
    for i,item in enumerate(inner_para_list):
        item.append(label_list[i])
        
    idx_list = [0,4,8]
    outer_para_list = [['mat0.txt','vec0.txt','result0.yaml'],['mat4.txt','vec4.txt','result4.yaml'],['mat8.txt','vec8.txt','result8.yaml']]
    metric_list = ['time','iter','resi']

    command = './run -pa {} -pb {} -solve_label {} -mat_file {} -vec_file {} -out_file {} \n'

    a = MultiTaskGenWithYamlJson(num_task,metric_list,label_list,inner_para_list,permutation=permutation)
    a.Process(idx_list,outer_para_list)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

def TestTaskGenNoInnerPar():
    label_list = ['non']
    inner_para_list = [['']]
    idx_list = [0,4,8]
    outer_para_list = [['mat0.txt','vec0.txt','output0.txt'],['mat4.txt','vec4.txt','output4.txt'],['mat8.txt','vec8.txt','output8.txt']]

    command = './run {} -pa a -pb b -mat_file {} -vec_file {} -out_file {} \n'


    a = TaskGen(label_list,inner_para_list)
    a.Process(idx_list,outer_para_list)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

if __name__ == '__main__':
    # TestTaskGen()
    # TestTaskGenPermutation()
    # TestTaskGenMultiTask()

    # TestSingleTaskGen()
    # TestSingleTaskGenPermutation()
    
    # TestMultiTaskGen()
    # TestMultiTaskGenPermutation()

    TestTaskGenNoInnerPar()
