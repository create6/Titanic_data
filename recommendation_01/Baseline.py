import pandas as pd
import numpy as np

class BaselineCFBySGD(object):
    def __init__(self,number_epochs,alpha,reg,columns=['uid','iid','rating']):
        
        # �ݶ��½���ߵ�������
        self.number_epochs = number_epochs
        # ѧϰ��
        self.alpha = alpha
        # �������
        self.reg = reg
        # ���ݼ���user-item-rating�ֶε�����
        self.columns = columns
        
    
    def fit(self,dataset):
        self.dataset = dataset
        #�û���������
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        #�û���������
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # ����ȫ��ƽ����
        self.global_mean = self.dataset[self.columns[2]].mean()
        # ����sgd����ѵ��ģ�Ͳ���
        self.bu, self.bi = self.sgd()
        # print(self.bu)
        
    def sgd(self):

        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))
        
        #����bu,bi
        #number_epochs ��������,alphaѧϰ��,reg����ϵͳ
        for i in range(self.number_epochs):
            # print('inter%d'%i)
            for uid,iid,real_rating in dataset.itertuples(index=False):
                #��ֵ(������ʧ)  error = ��ʵֵ - Ԥ��ֵ
                error =real_rating -(self.global_mean +bu[uid] +bi[iid])
                # �ݶ��½����Ƶ�
                # bu  = bu+��?(��u,i��R(rui?��?bu?bi)?��?bu) 
                # ����ݶ��½�
                # bu = bu + a*(error - ��?bu)
                bu[uid] += self.alpha *(error -self.reg*bu[uid])
                bi[iid] += self.alpha *(error -self.reg*bi[iid])
        return bu,bi
    
    #Ԥ��
    def predict(self,uid,iid):
        predict_rating =self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating
    

#����
dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
dataset = pd.read_csv("../../data/ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))

bcf = BaselineCFBySGD(20, 0.1, 0.1, ["userId", "movieId", "rating"])
bcf.fit(dataset)
print(bcf.predict(3, 2))

