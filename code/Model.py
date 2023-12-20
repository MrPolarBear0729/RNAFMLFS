# %%
#导入包与设定随机函数

import os
import pandas as pd
import fm
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,random_split,TensorDataset
import numpy as np
import random
import torch.nn.functional as F

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(42)

# %%
#导入RNA-FM
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = get_device()
print(f"使用的设备是: {device}")
backbone, alphabet = fm.pretrained.rna_fm_t12()
print("create RNA-FM_backbone sucessfully")

# %%
#读取并提取数据信息
publictest=pd.read_csv('../inputdata/inputdata.csv')
publictestdata=[]
for i in range(0,len(publictest['sequence'])):
    publictestdata.append(('RNA'+str(i),publictest['sequence'][i]))
batch_converter = alphabet.get_batch_converter()
batch_labels_publictestdata, batch_strs_publictestdata, batch_tokens_publictestdata = batch_converter(publictestdata)

# %%
#定义模型结构
class Model(nn.Module):
    def __init__(self,Backbone=backbone):
        super().__init__()
        self.RNAFM=Backbone
        self.lstm1 = nn.LSTM(input_size=640, hidden_size=256,batch_first=True,bidirectional=True,num_layers=2)
        self.gru1 = nn.GRU(input_size=256*2,hidden_size=256,batch_first=True,bidirectional=True,num_layers=2)
        self.linear1 = nn.Linear(256*2, 3)
    def forward(self,inputx):
        result = self.RNAFM(inputx,repr_layers=[12])["representations"][12]
        lstm_out1, (h_n1, h_c1) = self.lstm1(result)
        gru1_out1,_ = self.gru1(lstm_out1)
        linear1_out=self.linear1(gru1_out1)
        return linear1_out


# %%
publictestdataset = TensorDataset(batch_tokens_publictestdata)
publictest_data_loader = DataLoader(publictestdataset,batch_size=20)

# %%
pred_model = torch.load('./model.pt').to(device)
pred_model.eval()

# %%
i=0
pred_len = 107
for x in publictest_data_loader:
    Pred_x = x[0].to(device)
    Pred_y = pred_model(Pred_x)[:,:pred_len,:]
    for j in range(0,Pred_y.shape[0]):
        np.savetxt('../tempfile/'+str(i*20+(j+1))+'.csv',Pred_y[j].cpu().detach().numpy(),delimiter=',')
    i+=1

# %%
#合并CSV文件
folder_path = '../tempfile/'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
csv_files = sorted(csv_files, key=lambda x: int(x.split('.')[0]))

# %%
merged_data = pd.DataFrame()
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path,header=None)
    merged_data = pd.concat([merged_data, df], ignore_index=False,axis=0)
merged_data.reset_index(inplace=True,drop=True)

# %%
def delete_all_files_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                
            elif os.path.isdir(file_path):
                delete_all_files_in_folder(file_path)
        except Exception as e:
            print(f"删除文件失败: {file_path}, 错误信息: {e}")

# 指定要删除文件的文件夹路径
folder_to_delete = "../tempfile"

# 调用函数删除文件夹中的所有文件
delete_all_files_in_folder(folder_to_delete)


# %%


# %%
templist = []
for k in publictest['id'].to_list():
    for q in range(0,pred_len):
        templist.append(k+'_'+str(q))
merged_data['id_seqpos']=templist
merged_data.columns=["reactivity", "deg_Mg_pH10", "deg_Mg_50C",'id_seqpos']
merged_data.to_csv('../outputdata/result.csv',index=False)


