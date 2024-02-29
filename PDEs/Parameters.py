import numpy as np

class Parameter:
    def __init__(self):
        self.para = {}
        # self.mat_info = {}

        self.AddParas()

    def AddParas(self):
        pass

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

    # def Register(self,info):
    #     info['para'] = self.para
    #     return info
