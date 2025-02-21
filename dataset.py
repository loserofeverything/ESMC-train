import pandas as pd
import torch
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import h5py
import ast
from multiprocessing import Pool
import tqdm

def merge_paths_and_labels(list_a, list_b):
    # 检查两个列表的长度是否相等
    if len(list_a) != len(list_b):
        raise ValueError("A 和 B 列表的长度不相等。")

    # 合并路径和标签
    merged_list = [f"{path}:::{label}" for path, label in zip(list_a, list_b)]
    
    # 返回合并后的列表
    return merged_list


def get_matching_files(list_a, folder_b):
    # 提取列表A中的文件编号，按顺序存储在一个列表中
    file_numbers_ordered = [os.path.basename(path).split('.')[0] for path in list_a]
    
    # 创建一个字典用于快速查找文件夹B中的文件
    files_in_b = {file_name.split('_')[0]: file_name for file_name in os.listdir(folder_b) if file_name.endswith('_raEncoded.tsv')}

    # 初始化结果列表
    matched_files = []

    # 根据列表A中的顺序，查找文件编号并添加相应路径
    for file_number in file_numbers_ordered:
        if file_number in files_in_b:
            matched_files.append(os.path.join(folder_b, files_in_b[file_number]))

    # 返回结果列表
    return matched_files





class dataset(Dataset):
    def __init__(self, file_label_list, label_encoder, mode, topk = 20000, \
    h5_root_dir = '/xiongjun/test/NEW-MIL/data/dataset/12_25_balanced', is_load = False, is_seq=True, \
        split = 2, classname="lab", is_load_encoded=True, is_load_filenames = False, exclude_k = None):
        self.label_encoder = label_encoder
        self.file_label_list = file_label_list
        self.exclude_k = exclude_k
        self.k = topk
        self.mode = mode
        self.split = split
        self.is_load_encoded = is_load_encoded
        self.is_load_filenames = is_load_filenames
        self.is_seq = is_seq
        classname = classname
        self.h5_folder = "{}split_topk{}_dataset_{}".format(split, int(topk), classname)
        self.h5_path = os.path.join(h5_root_dir, self.h5_folder)
        if not os.path.exists(self.h5_path):
            os.makedirs(self.h5_path)
        self.h5_file = os.path.join(self.h5_path, "{}_data.h5".format(mode))
        self.is_load = is_load
        self.chosen_sequence = None
        self.magic_number = 114514
        if os.path.exists(self.h5_file):
            if self.is_load:
                self.load_h5_file()
        else:
            self.preload_data()


    
    def load_h5_file(self):
      with h5py.File(self.h5_file, 'r') as f:
        self.clone_fractions = f['clone_fractions'][:]
        self.labels = f['labels'][:]
        if self.is_load_filenames:
            self.file_names = f['file_names'][:]
        if self.is_load_encoded:
            self.cdr3_encoded = f['cdr3_encoded'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.data_indices = f['data_indices'][:]
        self.seq_lens = f['seq_lens'][:]
        
        if self.exclude_k is not None:
            # 处理单个数值或列表输入
            if isinstance(self.exclude_k, (int, float)):
                indices = self.labels != self.exclude_k
            else:  # k是列表
                indices = ~np.zeros_like(self.labels, dtype=bool)
                for label in self.exclude_k:
                    indices = np.logical_and(indices, self.labels != label)
            
            self.exclude_indices = indices
            
            # 应用过滤
            self.clone_fractions = self.clone_fractions[indices]
            self.labels = self.labels[indices]
            if self.is_load_filenames:
                self.file_names = self.file_names[indices]
            if self.is_load_encoded:
                self.cdr3_encoded = self.cdr3_encoded[indices]
            self.cdr3_seq = self.cdr3_seq[indices]
        
        self.corpus_lines = len(self.labels)
        

  
    

    def get_corpus_line(self, idx):
        
        if self.is_seq:
            SID = idx
            FID = idx//self.k
            label = self.labels[idx]
            cdr3 = (self.cdr3_seq[idx]).decode('utf-8')
            if self.is_load_encoded:
                cdr3_encoded = self.cdr3_encoded[idx]
            cloneFraction = self.clone_fractions[idx]
            seq_len = len(cdr3)
            if self.is_load_encoded:
                return SID, FID, label, cdr3, cdr3_encoded, seq_len, cloneFraction
            else:
                return SID, FID, label, cdr3, seq_len, cloneFraction
        else:
            FID = idx
            start_idx, l = idx * self.k, self.k
            SID = None
            labels = self.labels[start_idx: start_idx + l]
            cdr3 = self.cdr3_seq[start_idx: start_idx + l]
            cdr3s = [x.decode('utf-8') for x in cdr3]
            if self.is_load_encoded:
                cdr3_encodeds = self.cdr3_encoded[start_idx: start_idx + l]
            cloneFractions = self.clone_fractions[start_idx: start_idx + l]
            seq_lens = [len(x) for x in cdr3]
            if self.is_load_encoded:
                return SID, FID, labels, cdr3s, cdr3_encodeds, seq_lens, cloneFractions
            else:
                return SID, FID, labels, cdr3s, seq_lens, cloneFractions

    def process_file(self, file_label):
        file_path = file_label.split(":::")[0]
        file_name = os.path.basename(file_path)
        f_label = file_label.split(":::")[1]
        df = pd.read_csv(file_path, sep='\t')
        actual_rows = len(df)
        df = df.nlargest(self.k, 'cloneFraction')
        if self.is_load_encoded:
            df['cdr3-encoded'] = df['cdr3-encoded'].apply(ast.literal_eval)
        seq_labels = [f_label] * self.k
        # validate_label = df['label'].tolist()
        # flag = False
        # if seq_labels == validate_label:
        #     flag = True
        # assert flag, "label not consistent"      
        clone_fraction = df['cloneFraction'].tolist()
        cdr3_seq = df['aaSeqCDR3'].tolist()
        seq_len = [len(x) for x in cdr3_seq]
        if self.is_load_encoded:
            cdr3_encoded = df['cdr3-encoded'].tolist()
        labels = self.label_encoder.transform(seq_labels)
        
        if actual_rows < self.k:
            missing_rows = self.k - actual_rows
            clone_fraction.extend([0.0] * missing_rows)
            cdr3_seq.extend(['XXXXXXXXXXXXXXXX'] * missing_rows)
            seq_len.extend([16] * missing_rows)
            # labels.extend([-1] * missing_rows)
            if self.is_load_encoded:
                cdr3_encoded.extend([[[0.0] * self.split] * 8] * missing_rows)
        
        if self.is_load_encoded:
            return cdr3_seq, cdr3_encoded, clone_fraction, labels, file_name, seq_len
        else:
            return cdr3_seq, clone_fraction, labels, file_name, seq_len

    def preload_data(self):

        all_clone_fractions = []
        all_labels = []
        all_cdr3_encode = []
        all_cdr3_seq = []
        all_seq_len = []
        # all_ema = []
        data_indices = []
        file_names = []

        start_idx = 0
        with Pool(processes=os.cpu_count()) as pool:
        # with Pool(processes=1) as pool:
            results = pool.map(self.process_file, self.file_label_list)
        #for data_list, clone_fraction, ema_targets, labels in results:
        if self.is_load_encoded:
            for cdr3_seq, cdr3_encoded, clone_fraction, labels, file_name, seq_len in results:
                all_clone_fractions.extend(clone_fraction)
                # all_ema.extend(ema_targets)
                data_indices.append((start_idx, len(cdr3_seq)))
                start_idx += len(cdr3_seq)
                file_names.append(file_name)
                all_labels.extend(labels)
                all_seq_len.extend(seq_len)
                all_cdr3_seq.extend(cdr3_seq)
                all_cdr3_encode.extend(cdr3_encoded)
        else:
            for cdr3_seq, clone_fraction, labels, file_name, seq_len in results:
                all_clone_fractions.extend(clone_fraction)
                # all_ema.extend(ema_targets)
                data_indices.append((start_idx, len(cdr3_seq)))
                start_idx += len(cdr3_seq)
                file_names.append(file_name)
                all_labels.extend(labels)
                all_seq_len.extend(seq_len)
                all_cdr3_seq.extend(cdr3_seq)
        with h5py.File(self.h5_file, 'w') as f:
            # max_k = max(arr.shape[0] if arr.ndim == 2 else 1 for arr in all_data)
            max_k = 10
            self.max_k = max_k
            f.create_dataset('clone_fractions', data=np.array(all_clone_fractions, dtype=np.float32))
            if self.is_load_encoded:
                f.create_dataset('cdr3_encoded', data=np.array(all_cdr3_encode, dtype=np.float32))
            f.create_dataset('labels', data=np.array(all_labels, dtype=np.int32))
            f.create_dataset('data_indices', data=np.array(data_indices, dtype=np.int32))
            f.create_dataset('seq_lens', data=np.array(all_seq_len, dtype=np.int32))
            f.create_dataset('cdr3_seq', data=np.array(all_cdr3_seq, dtype=h5py.string_dtype(encoding='utf-8')))
            f.create_dataset('file_names', data=np.array(file_names, dtype=h5py.string_dtype(encoding='utf-8')))
            # f.create_dataset('ema_targets', data=np.array(all_ema, dtype=np.float32))
        if self.is_load:
            self.load_h5_file()

    
    
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    
    
    def __len__(self):
        if self.is_seq:
            return len(self.labels)
        else:
            return len(self.labels)//self.k

    
    def __getitem__(self, idx):
        
        if self.is_load_encoded:
            SID, FID, label, cdr3, cdr3_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)

            
            return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long),\
                    torch.tensor(cdr3_encoded, dtype=torch.float32), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
        
        else:
            SID, FID, label, cdr3, seq_len, cloneFraction = self.get_corpus_line(idx)

            
            return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
                   
            

  


class CMVdataset(Dataset):
    def __init__(self, file_path, is_seq=True):
        assert os.path.exists(file_path), "file not exists"
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line[:-1].split("\t") for line in tqdm.tqdm(f, desc="Loading Dataset")]
        columns = ['ID', 'TRA_cdr3_3Mer', 'TRB_cdr3_3Mer', 'label', 'TRA_cdr3', 'TRA_v_gene', 'TRA_j_gene', 'TRB_cdr3', 'TRB_v_gene', 'TRB_j_gene', 'cloneFraction']
        self.df = pd.DataFrame(self.lines, columns=columns)
        self.df = self.df.drop(columns=['TRA_cdr3_3Mer', 'TRB_cdr3_3Mer', 'TRA_cdr3', 'TRA_v_gene', 'TRA_j_gene', 'TRB_v_gene', 'TRB_j_gene'])
        self.is_seq = is_seq
        self.magic_number = 114514
        self.unique_ids = self.df.iloc[:, 0].unique()


    def get_values_by_id(self, target_id: int, cid: int) -> list:
        values = self.df[self.df.iloc[:, 0] == target_id].iloc[:, cid].tolist()
        return values

    def get_corpus_line(self, idx):
        
        if self.is_seq:
            SID = idx
            FID = int(self.df.iat[idx,0])
            label = int(self.df.iat[idx,1])
            cdr3 = self.df.iat[idx, 2]
            cloneFraction = float(self.df.iat[idx, 3])
            seq_len = len(cdr3)
            return SID, FID, label, cdr3, seq_len, cloneFraction
        else:
            
            SID = None
            FID = self.unique_ids[idx]
            labels = self.get_values_by_id(FID, 1)
            labels = [int(x) for x in labels]
            cdr3s = self.get_values_by_id(FID, 2)
            seq_lens = [len(x) for x in cdr3s]
            cloneFractions = self.get_values_by_id(FID, 3)
            cloneFractions = [float(x) for x in cloneFractions]
            return SID, FID, labels, cdr3s, seq_lens, cloneFractions

    
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    

    def __len__(self):
        if self.is_seq:
            return len(self.df)
        else:
            return len(self.unique_ids)

    
    def __getitem__(self, idx):
        
        # cdr3s string 
        SID, FID, label, cdr3s, seq_lens, cloneFractions = self.get_corpus_line(idx)

        return int(SID) if SID is not None else self.magic_number,\
            int(FID) if FID is not None else self.magic_number,\
            cdr3s, torch.tensor(label, dtype=torch.long), seq_lens,\
                torch.tensor(cloneFractions, dtype=torch.float32)
                
                                
                    
# df = df.nlargest(self.k, 'cloneFraction')
# df['cdr3-encoded'] = df['cdr3-encoded'].apply(ast.literal_eval)
# seq_labels = [f_label] * self.k
# clone_fraction = df['cloneFraction'].tolist()
# cdr3_seq = df['aaSeqCDR3'].tolist()
# seq_len = [len(x) for x in cdr3_seq]
# cdr3_encoded = df['cdr3-encoded'].tolist()
# labels = self.label_encoder.transform(seq_labels)



class Metadataset(Dataset):
    def __init__(self, file_label_list, label_encoder, mode, \
    h5_root_dir = '/xiongjun/test/NEW-MIL/data/Metadataset', is_load = False,  split = 2, classname = "new-meta"):

        self.label_encoder = label_encoder
        self.file_label_list = file_label_list
        self.split = split
        self.h5_folder = "{}split_dataset_{}".format(split, classname)
        os.makedirs(h5_root_dir, exist_ok=True)
        self.h5_path = os.path.join(h5_root_dir, self.h5_folder)
        if not os.path.exists(self.h5_path):
            os.makedirs(self.h5_path)
        self.h5_file = os.path.join(self.h5_path, "{}_data.h5".format(mode))
        self.is_load = is_load
        self.chosen_sequence = None
        self.magic_number = 114514
        if os.path.exists(self.h5_file):
            if self.is_load:
                self.load_h5_file()
        else:
            self.preload_data()


    
    def load_h5_file(self):
      with h5py.File(self.h5_file, 'r') as f:

        self.labels = f['labels'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.corpus_lines = len(self.labels)
        self.data_indices = f['data_indices'][:]
        self.seq_lens = f['seq_lens'][:]
        
    

    def get_corpus_line(self, idx):
        
        SID = idx
        label = self.labels[idx]
        cdr3 = (self.cdr3_seq[idx]).decode('utf-8')

        seq_len = len(cdr3)
        return SID, label, cdr3, seq_len
        

    def process_file(self, file_label):
        file_path = file_label.split(":::")[0]
        file_name = os.path.basename(file_path)
        f_label = file_label.split(":::")[1]
        df = pd.read_csv(file_path, sep='\t', header=None, names=['aaSeqCDR3'])
        # df = pd.read_csv(file_path, sep='\t')
        actual_rows = len(df)
        seq_labels = [f_label] * actual_rows    
        cdr3_seq = df['aaSeqCDR3'].tolist()
        seq_len = [len(x) for x in cdr3_seq]
        labels = self.label_encoder.transform(seq_labels)
        
        return cdr3_seq, labels, file_name, seq_len

    def preload_data(self):


        all_labels = []

        all_cdr3_seq = []
        all_seq_len = []
        # all_ema = []
        data_indices = []
        file_names = []

        start_idx = 0
        with Pool(processes=os.cpu_count()) as pool:
        # with Pool(processes=1) as pool:
            results = pool.map(self.process_file, self.file_label_list)
        #for data_list, clone_fraction, ema_targets, labels in results:
        for cdr3_seq, labels, file_name, seq_len in results:

            # all_ema.extend(ema_targets)
            data_indices.append((start_idx, len(cdr3_seq)))
            start_idx += len(cdr3_seq)
            file_names.append(file_name)
            all_labels.extend(labels)
            all_seq_len.extend(seq_len)
            all_cdr3_seq.extend(cdr3_seq)

        with h5py.File(self.h5_file, 'w') as f:

            f.create_dataset('labels', data=np.array(all_labels, dtype=np.int32))
            f.create_dataset('data_indices', data=np.array(data_indices, dtype=np.int32))
            f.create_dataset('seq_lens', data=np.array(all_seq_len, dtype=np.int32))
            f.create_dataset('cdr3_seq', data=np.array(all_cdr3_seq, dtype=h5py.string_dtype(encoding='utf-8')))
            f.create_dataset('file_names', data=np.array(file_names, dtype=h5py.string_dtype(encoding='utf-8')))
            # f.create_dataset('ema_targets', data=np.array(all_ema, dtype=np.float32))
        if self.is_load:
            self.load_h5_file()

    

    
    
    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, idx):
        
        SID, label, cdr3, seq_len = self.get_corpus_line(idx)

        
        return int(SID) if SID is not None else self.magic_number,\
            cdr3, torch.tensor(label, dtype=torch.long),seq_len
            
            
            
            
class Meta_bert_embs_dataset(Dataset):
    def __init__(self,h5_file_path, is_full = False, is_seq = True, topk = 100, is_use_Sparse = False):

        self.h5_file = h5_file_path
        self.magic_number = 114514
        self.is_full = is_full
        self.is_seq = is_seq
        self.topk = topk
        self.is_use_Sparse = is_use_Sparse
        self.load_h5_file()
        
    
    def load_h5_file(self):
      with h5py.File(self.h5_file, 'r') as f:
        self.labels = f['labels'][:]
        self.bert_encoded = f['cdr3_bert_encoded'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.sequence_id = f['sequence_id'][:]
        self.seq_cnt = len(self.labels)
        if self.is_full:
            self.bert_encoded = f['cdr3_full_bert_encoded'][:]
        self.patien_cnt = self.seq_cnt // self.topk
        if self.is_use_Sparse:
            self.sparse_encoded = f['sparse_encoded'][:]
        

    def get_corpus_line(self, idx):
        if self.is_seq:
            SID = idx
            FID = None
            label = self.labels[idx]
            cdr3 = (self.cdr3_seq[idx]).decode('utf-8')
            bert_encoded = self.bert_encoded[idx]
            seq_len = len(cdr3)
            if self.is_use_Sparse is False:
                return SID, FID, label, cdr3, bert_encoded, seq_len
            else:
                sparse_encoded = self.sparse_encoded[idx]
                return SID, FID, label, cdr3, bert_encoded, sparse_encoded, seq_len
        else:
            FID = idx
            start_idx = idx * self.topk
            l = self.topk
            SID = None
            labels = self.labels[start_idx: start_idx + l]
            cdr3 = self.cdr3_seq[start_idx: start_idx + l]
            cdr3s = [x.decode('utf-8') for x in cdr3]
            bert_encodeds = self.bert_encoded[start_idx: start_idx + l]
            seq_lens = [len(x) for x in cdr3]
            if self.is_use_Sparse is False:
                return SID, FID, labels, cdr3s, bert_encodeds, seq_lens
            else:
                sparse_encodeds = self.sparse_encoded[start_idx: start_idx + l]
                return SID, FID, labels, cdr3s, bert_encodeds, sparse_encodeds, seq_lens
    
    
    
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    
    def __len__(self):
        if self.is_seq:
            return self.seq_cnt
        else:
            return self.patien_cnt

    def __getitem__(self, idx):
        
        if self.is_use_Sparse is False:  
            SID, FID, label, cdr3, bert_encoded, seq_len = self.get_corpus_line(idx)
            return int(SID) if SID is not None else self.magic_number, \
            int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long),\
                torch.tensor(bert_encoded, dtype=torch.float32), seq_len
        else:
            SID, FID, label, cdr3, bert_encoded, sparse_encoded, seq_len = self.get_corpus_line(idx)
            return int(SID) if SID is not None else self.magic_number, \
            int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long),\
                torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(sparse_encoded, dtype=torch.float32), seq_len
            
                   
class NaiveDataset4DivideMix(Dataset):
    def __init__(self, features, labels, prob=None):
        self.features = features
        self.labels = labels.long()
        self.porb = prob
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.porb is None:
            return self.features[idx]
        else:
            return self.features[idx], self.labels[idx], self.porb[idx]








class bert_embs_dataset(Dataset):
    def __init__(self,h5_file_path, is_seq=True, is_full = False, use_ra_encoded = False, exclude_k = None, key = "bert"):
        self.is_seq = is_seq
        self.key = key
        self.h5_file = h5_file_path
        self.magic_number = 114514
        self.is_full = is_full
        self.use_ra_encoded = use_ra_encoded
        self.use_noisy_labels = False
        self.exclude_indices = None
        self.load_h5_file(exclude_k)


    
    def load_h5_file(self, k=None):
      with h5py.File(self.h5_file, 'r') as f:
        self.clone_fractions = f['clone_fractions'][:]
        self.labels = f['labels'][:]
        if self.key == "bert":
            self.bert_encoded = f['cdr3_bert_encoded'][:]
        elif self.key == "esm":
            self.bert_encoded = f['cdr3_esm_encoded'][:]
        self.patient_id = f['patient_id'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.sequence_id = f['sequence_id'][:]
        if self.is_full:
            self.bert_encoded = f['cdr3_full_bert_encoded'][:]
        if self.use_ra_encoded:
            self.ra_encoded = f['ra_encoded'][:]
        
        if k is not None:
            # 处理单个数值或列表输入
            if isinstance(k, (int, float)):
                indices = self.labels != k
            else:  # k是列表
                indices = ~np.zeros_like(self.labels, dtype=bool)
                for label in k:
                    indices = np.logical_and(indices, self.labels != label)
            
            self.exclude_indices = indices
            
            # 应用过滤
            self.clone_fractions = self.clone_fractions[indices]
            self.labels = self.labels[indices]
            self.bert_encoded = self.bert_encoded[indices]
            self.patient_id = self.patient_id[indices]
            self.cdr3_seq = self.cdr3_seq[indices]
            if self.is_full:
                self.bert_encoded = self.bert_encoded[indices]
            if self.use_ra_encoded:
                self.ra_encoded = self.ra_encoded[indices]
            
            
        
        self.patient_cnt = len(set(self.patient_id))
        
        self.seq_cnt = len(self.labels)
        self.seq_per_patient = self.seq_cnt // self.patient_cnt
  
    
    def get_values_with_step(self, A, d, B):
        # 为每个A中的数生成两个连续索引
        indices = np.concatenate([(A*d)[:, None] + np.arange(d), ], axis=1).flatten()
        # 从B中取值
        return B[indices]
    
    def get_independent_dataset(self, idxs):
        if self.is_seq:
            raise ValueError("Not supported for sequence format")
        else:
            mask = idxs < self.patient_cnt
            idxs = idxs[mask]
            self.clone_fractions = self.get_values_with_step(idxs, self.seq_per_patient, self.clone_fractions)
            self.labels = self.get_values_with_step(idxs, self.seq_per_patient, self.labels)
            self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            self.patient_id = self.get_values_with_step(idxs, self.seq_per_patient, self.patient_id)
            self.cdr3_seq = self.get_values_with_step(idxs, self.seq_per_patient, self.cdr3_seq)
            if self.is_full:
                self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            if self.use_ra_encoded:
                self.ra_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.ra_encoded)

        
            self.patient_cnt = len(set(self.patient_id))
            self.seq_cnt = len(self.labels)
            self.seq_per_patient = self.seq_cnt // self.patient_cnt
            
    def get_corpus_line(self, idx):
        
        if self.is_seq:
            SID = idx
            FID = self.patient_id[idx]
            label = self.labels[idx]
            
            cdr3 = (self.cdr3_seq[idx]).decode('utf-8')
            bert_encoded = self.bert_encoded[idx]
            cloneFraction = self.clone_fractions[idx]
            seq_len = len(cdr3)
            if self.use_ra_encoded:
                ra_encoded = self.ra_encoded[idx]
                if self.use_noisy_labels:
                    noisy_label = self.noisy_labels[idx]
                    return SID, FID, noisy_label, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction
                else:
                    return SID, FID, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction
            else:
                if self.use_noisy_labels:
                    noisy_label = self.noisy_labels[idx]
                    return SID, FID, noisy_label, label, cdr3, bert_encoded, seq_len, cloneFraction
                else:
                    return SID, FID, label, cdr3, bert_encoded, seq_len, cloneFraction
        else:
            FID = idx
            start_idx = idx * self.seq_per_patient
            l = self.seq_per_patient
            SID = None
            labels = self.labels[start_idx: start_idx + l]
            cdr3 = self.cdr3_seq[start_idx: start_idx + l]
            cdr3s = [x.decode('utf-8') for x in cdr3]
            bert_encodeds = self.bert_encoded[start_idx: start_idx + l]
            cloneFractions = self.clone_fractions[start_idx: start_idx + l]
            seq_lens = [len(x) for x in cdr3]
            if self.use_ra_encoded:
                ra_encodeds = self.ra_encoded[start_idx: start_idx + l]
                return SID, FID, labels, cdr3s, bert_encodeds, ra_encodeds, seq_lens, cloneFractions
            else:
                return SID, FID, labels, cdr3s, bert_encodeds, seq_lens, cloneFractions

 
    
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    
    def get_noisy_labels(self, n_l):
        self.noisy_labels = n_l
        self.use_noisy_labels = True
    
    def use_purified_labels(self, purified_labels, non_cancer_label = 3):
        cancer_indices = (self.labels != non_cancer_label)
        non_cancer_indices = (self.labels == non_cancer_label)
        # if self.exclude_indices is not None:
        #     purified_labels = purified_labels[self.exclude_indices[:len(purified_labels)]]
            
        self.labels[cancer_indices] = purified_labels
        self.labels[non_cancer_indices] = 0

    
    def __len__(self):
        if self.is_seq:
            return self.seq_cnt
        else:
            return self.patient_cnt

    
    def __getitem__(self, idx):
        
        if self.use_ra_encoded is False:
            
            if self.use_noisy_labels is False:
                SID, FID, label, cdr3, bert_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long),\
                    torch.tensor(bert_encoded, dtype=torch.float32), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
            else:
                SID, FID, noisy_label, label, cdr3, bert_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(noisy_label, dtype=torch.long),torch.tensor(label, dtype=torch.long),\
                    torch.tensor(bert_encoded, dtype=torch.float32), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
                   
        else:
            if self.use_noisy_labels is False:
                SID, FID, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                
            else:
                SID, FID, noisy_label, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(noisy_label, dtype=torch.long), torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
        
        
class bert_embs_memmap_dataset(Dataset):
    def __init__(self,h5_file_path, identify_name, is_seq=True, is_full = False, use_ra_encoded = False, exclude_k = None):
        self.is_seq = is_seq
        self.h5_file = h5_file_path
        self.magic_number = 114514
        self.is_full = is_full
        self.use_ra_encoded = use_ra_encoded
        self.use_noisy_labels = False
        self.exclude_indices = None
        self.identify_name = identify_name
        self.load_h5_file(exclude_k)


    
    def load_h5_file(self, k=None):
      with h5py.File(self.h5_file, 'r') as f:
        self.clone_fractions = f['clone_fractions'][:]
        self.labels = f['labels'][:]
        
        self.patient_id = f['patient_id'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.sequence_id = f['sequence_id'][:]
        
        if self.use_ra_encoded:
            self.ra_encoded = f['ra_encoded'][:]
        
        chunk_size = 8192
        
        mmap_path = os.path.join("/xiongjun/test/NEW-MIL-dir-backup/backup/NEW-MIL-dir-backup/data/tmp_mmap", "{}_chunksize{}.mmap".format(self.identify_name, chunk_size))
        if self.is_full is False:
            shape = f['cdr3_bert_encoded'].shape
            dtype = f['cdr3_bert_encoded'].dtype
        else:
            shape = f['cdr3_full_bert_encoded'].shape
            dtype = f['cdr3_full_bert_encoded'].dtype

                
        self.bert_encoded = np.memmap(
            mmap_path,
            dtype=dtype,
            mode='w+',
            shape=shape
        )
        
        for i in range(0, shape[0], chunk_size):
            end = min(i + chunk_size, shape[0])
            if self.is_full is False:
                self.bert_encoded[i:end] = f['cdr3_bert_encoded'][i:end]
            else:
                self.bert_encoded[i:end] = f['cdr3_full_bert_encoded'][i:end]

            
        if k is not None:
            self._apply_exclusion(k)
        
        
        
        self.patient_cnt = len(set(self.patient_id))
        
        self.seq_cnt = len(self.labels)
        self.seq_per_patient = self.seq_cnt // self.patient_cnt
    
    
    
    def _apply_exclusion(self, k):
        if isinstance(k, (int, float)):
            indices = self.labels != k
        else:
            indices = ~np.zeros_like(self.labels, dtype=bool)
            for label in k:
                indices = np.logical_and(indices, self.labels != label)
        
        self.labels = self.labels[indices]
        self.patient_id = self.patient_id[indices]
        self.cdr3_seq = self.cdr3_seq[indices]
        self.clone_fractions = self.clone_fractions[indices]
        self.bert_encoded = self.bert_encoded[indices]
        if self.use_ra_encoded:
            self.ra_encoded = self.ra_encoded[indices]
    
    def get_values_with_step(self, A, d, B):
        # 为每个A中的数生成两个连续索引
        indices = np.concatenate([(A*d)[:, None] + np.arange(d), ], axis=1).flatten()
        # 从B中取值
        return B[indices]
    
    def get_independent_dataset(self, idxs):
        if self.is_seq:
            raise ValueError("Not supported for sequence format")
        else:
            mask = idxs < self.patient_cnt
            idxs = idxs[mask]
            self.clone_fractions = self.get_values_with_step(idxs, self.seq_per_patient, self.clone_fractions)
            self.labels = self.get_values_with_step(idxs, self.seq_per_patient, self.labels)
            self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            self.patient_id = self.get_values_with_step(idxs, self.seq_per_patient, self.patient_id)
            self.cdr3_seq = self.get_values_with_step(idxs, self.seq_per_patient, self.cdr3_seq)
            if self.is_full:
                self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            if self.use_ra_encoded:
                self.ra_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.ra_encoded)

        
            self.patient_cnt = len(set(self.patient_id))
            self.seq_cnt = len(self.labels)
            self.seq_per_patient = self.seq_cnt // self.patient_cnt
            
    def get_corpus_line(self, idx):
        
        if self.is_seq:
            SID = idx
            FID = self.patient_id[idx]
            label = self.labels[idx]
            
            cdr3 = (self.cdr3_seq[idx]).decode('utf-8')
            bert_encoded = self.bert_encoded[idx]
            cloneFraction = self.clone_fractions[idx]
            seq_len = len(cdr3)
            if self.use_ra_encoded:
                ra_encoded = self.ra_encoded[idx]
                if self.use_noisy_labels:
                    noisy_label = self.noisy_labels[idx]
                    return SID, FID, noisy_label, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction
                else:
                    return SID, FID, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction
            else:
                if self.use_noisy_labels:
                    noisy_label = self.noisy_labels[idx]
                    return SID, FID, noisy_label, label, cdr3, bert_encoded, seq_len, cloneFraction
                else:
                    return SID, FID, label, cdr3, bert_encoded, seq_len, cloneFraction
        else:
            FID = idx
            start_idx = idx * self.seq_per_patient
            l = self.seq_per_patient
            SID = None
            labels = self.labels[start_idx: start_idx + l]
            cdr3 = self.cdr3_seq[start_idx: start_idx + l]
            cdr3s = [x.decode('utf-8') for x in cdr3]
            bert_encodeds = self.bert_encoded[start_idx: start_idx + l]
            cloneFractions = self.clone_fractions[start_idx: start_idx + l]
            seq_lens = [len(x) for x in cdr3]
            if self.use_ra_encoded:
                ra_encodeds = self.ra_encoded[start_idx: start_idx + l]
                return SID, FID, labels, cdr3s, bert_encodeds, ra_encodeds, seq_lens, cloneFractions
            else:
                return SID, FID, labels, cdr3s, bert_encodeds, seq_lens, cloneFractions
            
            
            
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    
    def get_noisy_labels(self, n_l):
        self.noisy_labels = n_l
        self.use_noisy_labels = True
    
    def use_purified_labels(self, purified_labels, non_cancer_label = 3):
        cancer_indices = (self.labels != non_cancer_label)
        non_cancer_indices = (self.labels == non_cancer_label)
        # if self.exclude_indices is not None:
        #     purified_labels = purified_labels[self.exclude_indices[:len(purified_labels)]]
            
        self.labels[cancer_indices] = purified_labels
        self.labels[non_cancer_indices] = 0

    
    def __len__(self):
        if self.is_seq:
            return self.seq_cnt
        else:
            return self.patient_cnt

    
    def __getitem__(self, idx):
        
        if self.use_ra_encoded is False:
            
            if self.use_noisy_labels is False:
                SID, FID, label, cdr3, bert_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long),\
                    torch.tensor(bert_encoded, dtype=torch.float32), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
            else:
                SID, FID, noisy_label, label, cdr3, bert_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(noisy_label, dtype=torch.long),torch.tensor(label, dtype=torch.long),\
                    torch.tensor(bert_encoded, dtype=torch.float32), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
                   
        else:
            if self.use_noisy_labels is False:
                SID, FID, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                
            else:
                SID, FID, noisy_label, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(noisy_label, dtype=torch.long), torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                        
                        
                        
class bert_embs_OnDemands_dataset(Dataset):
    def __init__(self,h5_file_path, is_seq=True, is_full = False, use_ra_encoded = False, exclude_k = None):
        self.is_seq = is_seq
        self.h5_file = h5_file_path
        self.magic_number = 114514
        self.is_full = is_full
        self.use_ra_encoded = use_ra_encoded
        self.use_noisy_labels = False
        self.exclude_indices = None
        self.file = None
        self.load_h5_file(exclude_k)


    
    def load_h5_file(self, k=None):
      with h5py.File(self.h5_file, 'r') as f:
        self.clone_fractions = f['clone_fractions'][:]
        self.labels = f['labels'][:]
        # self.bert_encoded = f['cdr3_bert_encoded'][:]
        self.patient_id = f['patient_id'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.sequence_id = f['sequence_id'][:]
        # if self.is_full:
        #     self.bert_encoded = f['cdr3_full_bert_encoded'][:]
        if self.use_ra_encoded:
            self.ra_encoded = f['ra_encoded'][:]
        
        if k is not None:
            # 处理单个数值或列表输入
            if isinstance(k, (int, float)):
                indices = self.labels != k
            else:  # k是列表
                indices = ~np.zeros_like(self.labels, dtype=bool)
                for label in k:
                    indices = np.logical_and(indices, self.labels != label)
            
            self.exclude_indices = np.where(indices)[0]

            
            # 应用过滤
            self.clone_fractions = self.clone_fractions[indices]
            self.labels = self.labels[indices]
            # self.bert_encoded = self.bert_encoded[indices]
            self.patient_id = self.patient_id[indices]
            self.cdr3_seq = self.cdr3_seq[indices]
            # if self.is_full:
            #     self.bert_encoded = self.bert_encoded[indices]
            if self.use_ra_encoded:
                self.ra_encoded = self.ra_encoded[indices]
            
            
        
        self.patient_cnt = len(set(self.patient_id))
        
        self.seq_cnt = len(self.labels)
        self.seq_per_patient = self.seq_cnt // self.patient_cnt
  
    
    # def get_values_with_step(self, A, d, B):
    #     # 为每个A中的数生成两个连续索引
    #     indices = np.concatenate([(A*d)[:, None] + np.arange(d), ], axis=1).flatten()
    #     # 从B中取值
    #     return B[indices]
    
    # def get_independent_dataset(self, idxs):
    #     if self.is_seq:
    #         raise ValueError("Not supported for sequence format")
    #     else:
    #         mask = idxs < self.patient_cnt
    #         idxs = idxs[mask]
    #         self.clone_fractions = self.get_values_with_step(idxs, self.seq_per_patient, self.clone_fractions)
    #         self.labels = self.get_values_with_step(idxs, self.seq_per_patient, self.labels)
    #         self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
    #         self.patient_id = self.get_values_with_step(idxs, self.seq_per_patient, self.patient_id)
    #         self.cdr3_seq = self.get_values_with_step(idxs, self.seq_per_patient, self.cdr3_seq)
    #         if self.is_full:
    #             self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
    #         if self.use_ra_encoded:
    #             self.ra_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.ra_encoded)

        
    #         self.patient_cnt = len(set(self.patient_id))
    #         self.seq_cnt = len(self.labels)
    #         self.seq_per_patient = self.seq_cnt // self.patient_cnt
            
    
    def init_file(self):
        self.file = h5py.File(self.h5_file, 'r')
    
    def get_corpus_line(self, idx):
        
        if self.is_seq:
            SID = idx
            FID = self.patient_id[idx]
            encoded_idx = self.exclude_indices[idx]
            
            if self.file is None:
                self.init_file()
            if self.is_full is False:
                bert_encoded = self.file['cdr3_bert_encoded'][encoded_idx]
            else:
                bert_encoded = self.file['cdr3_full_bert_encoded'][encoded_idx]
            label = self.labels[idx]
            cdr3 = (self.cdr3_seq[idx]).decode('utf-8')
            
            cloneFraction = self.clone_fractions[idx]
            seq_len = len(cdr3)
            if self.use_ra_encoded:
                ra_encoded = self.ra_encoded[idx]
                if self.use_noisy_labels:
                    noisy_label = self.noisy_labels[idx]
                    return SID, FID, noisy_label, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction
                else:
                    return SID, FID, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction
            else:
                if self.use_noisy_labels:
                    noisy_label = self.noisy_labels[idx]
                    return SID, FID, noisy_label, label, cdr3, bert_encoded, seq_len, cloneFraction
                else:
                    return SID, FID, label, cdr3, bert_encoded, seq_len, cloneFraction
        else:
            FID = idx
            start_idx = idx * self.seq_per_patient
            encoded_start_idx = self.exclude_indices[start_idx]
            l = self.seq_per_patient
            SID = None
            labels = self.labels[start_idx: start_idx + l]
            cdr3 = self.cdr3_seq[start_idx: start_idx + l]
            cdr3s = [x.decode('utf-8') for x in cdr3]
            if self.file is None:
                self.init_file()
            if self.is_full is False:
                bert_encodeds = self.file['cdr3_bert_encoded'][encoded_start_idx: encoded_start_idx + l]
            else:
                bert_encodeds = self.file['cdr3_full_bert_encoded'][encoded_start_idx: encoded_start_idx + l]
            cloneFractions = self.clone_fractions[start_idx: start_idx + l]
            seq_lens = [len(x) for x in cdr3]
            if self.use_ra_encoded:
                ra_encodeds = self.ra_encoded[start_idx: start_idx + l]
                return SID, FID, labels, cdr3s, bert_encodeds, ra_encodeds, seq_lens, cloneFractions
            else:
                return SID, FID, labels, cdr3s, bert_encodeds, seq_lens, cloneFractions

 
    
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    
    def get_noisy_labels(self, n_l):
        self.noisy_labels = n_l
        self.use_noisy_labels = True
    
    def use_purified_labels(self, purified_labels, non_cancer_label = 3):
        cancer_indices = (self.labels != non_cancer_label)
        non_cancer_indices = (self.labels == non_cancer_label)
        # if self.exclude_indices is not None:
        #     purified_labels = purified_labels[self.exclude_indices[:len(purified_labels)]]
            
        self.labels[cancer_indices] = purified_labels
        self.labels[non_cancer_indices] = 0

    
    def __len__(self):
        if self.is_seq:
            return self.seq_cnt
        else:
            return self.patient_cnt

    
    def __getitem__(self, idx):
        
        if self.use_ra_encoded is False:
            
            if self.use_noisy_labels is False:
                SID, FID, label, cdr3, bert_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long),\
                    torch.tensor(bert_encoded, dtype=torch.float32), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
            else:
                SID, FID, noisy_label, label, cdr3, bert_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(noisy_label, dtype=torch.long),torch.tensor(label, dtype=torch.long),\
                    torch.tensor(bert_encoded, dtype=torch.float32), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
                   
        else:
            if self.use_noisy_labels is False:
                SID, FID, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                
            else:
                SID, FID, noisy_label, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(noisy_label, dtype=torch.long), torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                        
                        
                        
                                 
                        
class NewNaiveDataset4DivideMix(Dataset):
    def __init__(self, h5_file_path, chosen_idxs, prob, exclude_k = None):
        self.h5_file = h5_file_path
        self.magic_number = 114514
        self.exclude_indices = None
        self.chosen_idxs = chosen_idxs
        self.prob = prob
        self.load_h5_file(exclude_k)
    
    def load_h5_file(self, k=None):
      with h5py.File(self.h5_file, 'r') as f:
        self.labels = f['labels'][:]
        self.bert_encoded = f['cdr3_bert_encoded'][:]
        
        if k is not None:
            # 处理单个数值或列表输入
            if isinstance(k, (int, float)):
                indices = self.labels != k
            else:  # k是列表
                indices = ~np.zeros_like(self.labels, dtype=bool)
                for label in k:
                    indices = np.logical_and(indices, self.labels != label)
            
            exclude_indices = np.where(indices)[0]
            chosen_indices = []
            for idx in self.chosen_idxs:
                chosen_indices.append(exclude_indices[idx])
            chosen_indices = np.array(chosen_indices)
            self.bert_encoded = self.bert_encoded[chosen_indices]
            self.labels = self.labels[chosen_indices] 
        
        self.seq_cnt = len(self.labels)
    
    def __len__(self):
        return self.seq_cnt
    
    def __getitem__(self, idx):
        if self.porb is None:
            return self.bert_encoded[idx]
        else:
            return self.bert_encoded[idx], self.labels[idx], self.porb[idx]
        
        
        
        

class DivideMix_bert_embs_dataset(Dataset):
    def __init__(self,h5_file_path, is_seq=True, is_full = False, use_ra_encoded = False, exclude_k = None):
        self.is_seq = is_seq
        self.x_idx = None
        self.u_dix = None
        self.is_divided = False
        self.h5_file = h5_file_path
        self.magic_number = 114514
        self.is_full = is_full
        self.use_ra_encoded = use_ra_encoded
        self.prob = None
        self.use_noisy_labels = False
        self.exclude_indices = None
        self.load_h5_file(exclude_k)


    
    def load_h5_file(self, k=None):
      with h5py.File(self.h5_file, 'r') as f:
        self.clone_fractions = f['clone_fractions'][:]
        self.labels = f['labels'][:]
        self.bert_encoded = f['cdr3_bert_encoded'][:]
        self.patient_id = f['patient_id'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.sequence_id = f['sequence_id'][:]
        if self.is_full:
            self.bert_encoded = f['cdr3_full_bert_encoded'][:]
        if self.use_ra_encoded:
            self.ra_encoded = f['ra_encoded'][:]
        
        if k is not None:
            # 处理单个数值或列表输入
            if isinstance(k, (int, float)):
                indices = self.labels != k
            else:  # k是列表
                indices = ~np.zeros_like(self.labels, dtype=bool)
                for label in k:
                    indices = np.logical_and(indices, self.labels != label)
            
            self.exclude_indices = np.where(indices)[0]
            
            # 应用过滤
            self.clone_fractions = self.clone_fractions[indices]
            self.labels = self.labels[indices]
            self.bert_encoded = self.bert_encoded[indices]
            self.patient_id = self.patient_id[indices]
            self.cdr3_seq = self.cdr3_seq[indices]
            if self.is_full:
                self.bert_encoded = self.bert_encoded[indices]
            if self.use_ra_encoded:
                self.ra_encoded = self.ra_encoded[indices]
            
            
        
        self.patient_cnt = len(set(self.patient_id))
        
        self.seq_cnt = len(self.labels)
        self.seq_per_patient = self.seq_cnt // self.patient_cnt
  
    
    def get_values_with_step(self, A, d, B):
        # 为每个A中的数生成两个连续索引
        indices = np.concatenate([(A*d)[:, None] + np.arange(d), ], axis=1).flatten()
        # 从B中取值
        return B[indices]
    
    def get_independent_dataset(self, idxs):
        if self.is_seq:
            raise ValueError("Not supported for sequence format")
        else:
            mask = idxs < self.patient_cnt
            idxs = idxs[mask]
            self.clone_fractions = self.get_values_with_step(idxs, self.seq_per_patient, self.clone_fractions)
            self.labels = self.get_values_with_step(idxs, self.seq_per_patient, self.labels)
            self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            self.patient_id = self.get_values_with_step(idxs, self.seq_per_patient, self.patient_id)
            self.cdr3_seq = self.get_values_with_step(idxs, self.seq_per_patient, self.cdr3_seq)
            if self.is_full:
                self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            if self.use_ra_encoded:
                self.ra_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.ra_encoded)

        
            self.patient_cnt = len(set(self.patient_id))
            self.seq_cnt = len(self.labels)
            self.seq_per_patient = self.seq_cnt // self.patient_cnt
            
    
    
    def get_divided_dataset(self, divided_idxs):

        chosen_idxs = np.array(divided_idxs)
        self.bert_encoded = self.bert_encoded[chosen_idxs]
        self.labels = self.labels[chosen_idxs]
        self.clone_fractions = self.clone_fractions[chosen_idxs]
        self.patient_id = self.patient_id[chosen_idxs]
        self.cdr3_seq = self.cdr3_seq[chosen_idxs]
        if self.use_ra_encoded:
            self.ra_encoded = self.ra_encoded[chosen_idxs]
            
            
    def set_divided_mode(self, divided_mode=False, x_idx=None, u_idx=None, prob=None):
        self.x_idx = x_idx
        self.u_idx = u_idx
        self.is_divided = divided_mode
        self.prob = prob
    
    
    def get_corpus_line(self, idx):
        
        if self.is_seq:
            if self.is_divided is False:
                SID = idx
                FID = self.patient_id[idx]
                label = self.labels[idx]
                cdr3 = (self.cdr3_seq[idx]).decode('utf-8')
                bert_encoded = self.bert_encoded[idx]
                cloneFraction = self.clone_fractions[idx]
                seq_len = len(cdr3)
                if self.use_ra_encoded:
                    ra_encoded = self.ra_encoded[idx]
                    if self.use_noisy_labels:
                        noisy_label = self.noisy_labels[idx]
                        return SID, FID, noisy_label, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction
                    else:
                        return SID, FID, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction
                
                else:
                    if self.use_noisy_labels:
                        noisy_label = self.noisy_labels[idx]
                        return SID, FID, noisy_label, label, cdr3, bert_encoded, seq_len, cloneFraction
                    else:
                        return SID, FID, label, cdr3, bert_encoded, seq_len, cloneFraction
                
                
            else:
                idx_x = self.x_idx[idx]
                idx_u = self.u_idx[idx%len(self.u_idx)]
                SID_u = idx_u
                FID_u = self.patient_id[idx_u]
                SID_x = idx_x
                FID_x = self.patient_id[idx_x]
                prob_x = self.prob[idx]
                label_x = self.labels[idx_x]
                cdr3_x = (self.cdr3_seq[idx_x]).decode('utf-8')
                cdr3_u = (self.cdr3_seq[idx_u]).decode('utf-8')
                label_u = self.labels[idx_u]

                bert_encoded_x = self.bert_encoded[idx_x]
                bert_encoded_u = self.bert_encoded[idx_u]
            
                return SID_x, SID_u, FID_x, FID_u, label_x, label_u, bert_encoded_x, bert_encoded_u, cdr3_x, cdr3_u, prob_x
        else:
            FID = idx
            start_idx = idx * self.seq_per_patient
            l = self.seq_per_patient
            SID = None
            labels = self.labels[start_idx: start_idx + l]
            cdr3 = self.cdr3_seq[start_idx: start_idx + l]
            cdr3s = [x.decode('utf-8') for x in cdr3]
            bert_encodeds = self.bert_encoded[start_idx: start_idx + l]
            cloneFractions = self.clone_fractions[start_idx: start_idx + l]
            seq_lens = [len(x) for x in cdr3]
            if self.use_ra_encoded:
                ra_encodeds = self.ra_encoded[start_idx: start_idx + l]
                return SID, FID, labels, cdr3s, bert_encodeds, ra_encodeds, seq_lens, cloneFractions
            else:
                return SID, FID, labels, cdr3s, bert_encodeds, seq_lens, cloneFractions

 
    
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    
    def get_noisy_labels(self, n_l):
        self.noisy_labels = n_l
        self.use_noisy_labels = True
    
    def use_purified_labels(self, purified_labels, non_cancer_label = 3):
        cancer_indices = (self.labels != non_cancer_label)
        non_cancer_indices = (self.labels == non_cancer_label)
        # if self.exclude_indices is not None:
        #     purified_labels = purified_labels[self.exclude_indices[:len(purified_labels)]]
            
        self.labels[cancer_indices] = purified_labels
        self.labels[non_cancer_indices] = 0

    
    def __len__(self):
        if self.is_seq:
            if self.is_divided is False:
                return self.seq_cnt
            else:
                return len(self.x_idx)
        else:
            return self.patient_cnt

    
    def __getitem__(self, idx):
        
        if self.is_divided is False:
            if self.use_ra_encoded is False:
                
                if self.use_noisy_labels is False:
                    SID, FID, label, cdr3, bert_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                    return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                else:
                    SID, FID, noisy_label, label, cdr3, bert_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                    return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(noisy_label, dtype=torch.long),torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                    
            else:
                if self.use_noisy_labels is False:
                    SID, FID, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                    return int(SID) if SID is not None else self.magic_number,\
                        int(FID) if FID is not None else self.magic_number,\
                        cdr3, torch.tensor(label, dtype=torch.long),\
                            torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                            torch.tensor(cloneFraction, dtype=torch.float32)
                    
                else:
                    SID, FID, noisy_label, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                    return int(SID) if SID is not None else self.magic_number,\
                        int(FID) if FID is not None else self.magic_number,\
                        cdr3, torch.tensor(noisy_label, dtype=torch.long), torch.tensor(label, dtype=torch.long),\
                            torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                            torch.tensor(cloneFraction, dtype=torch.float32)
        
        else:
            SID_x, SID_u, FID_x, FID_u, label_x, label_u, bert_encoded_x, bert_encoded_u, cdr3_x, cdr3_u, prob_x = self.get_corpus_line(idx)
            return int(SID_x) if SID_x is not None else self.magic_number,\
                int(SID_u) if SID_u is not None else self.magic_number,\
                int(FID_x) if FID_x is not None else self.magic_number,\
                int(FID_u) if FID_u is not None else self.magic_number,\
                torch.tensor(label_x, dtype=torch.long), torch.tensor(label_u, dtype=torch.long),\
                    torch.tensor(bert_encoded_x, dtype=torch.float32), torch.tensor(bert_encoded_u, dtype=torch.float32), cdr3_x, cdr3_u, prob_x
                    
                    
                    
                    
      
                    
class DivideMix_B62_embs_dataset(Dataset):
    def __init__(self,h5_file_path, is_seq=True, is_full = False, use_ra_encoded = False, exclude_k = None):
        self.is_seq = is_seq
        self.x_idx = None
        self.u_dix = None
        self.is_divided = False
        self.h5_file = h5_file_path
        self.magic_number = 114514
        self.is_full = is_full
        self.use_ra_encoded = use_ra_encoded
        self.prob = None
        self.use_noisy_labels = False
        self.exclude_indices = None
        self.load_h5_file(exclude_k)


    
    def load_h5_file(self, k=None):
      with h5py.File(self.h5_file, 'r') as f:
        self.clone_fractions = f['clone_fractions'][:]
        self.labels = f['labels'][:]
        self.bert_encoded = f['cdr3_bert_encoded'][:]
        self.bert_mask = f['cdr3_bert_mask'][:]
        self.patient_id = f['patient_id'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.sequence_id = f['sequence_id'][:]
        if self.is_full:
            self.bert_encoded = f['cdr3_full_bert_encoded'][:]
        if self.use_ra_encoded:
            self.ra_encoded = f['ra_encoded'][:]
        
        if k is not None:
            # 处理单个数值或列表输入
            if isinstance(k, (int, float)):
                indices = self.labels != k
            else:  # k是列表
                indices = ~np.zeros_like(self.labels, dtype=bool)
                for label in k:
                    indices = np.logical_and(indices, self.labels != label)
            
            self.exclude_indices = np.where(indices)[0]
            
            # 应用过滤
            self.clone_fractions = self.clone_fractions[indices]
            self.labels = self.labels[indices]
            self.bert_encoded = self.bert_encoded[indices]
            self.bert_mask = self.bert_mask[indices]
            self.patient_id = self.patient_id[indices]
            self.cdr3_seq = self.cdr3_seq[indices]
            if self.is_full:
                self.bert_encoded = self.bert_encoded[indices]
            if self.use_ra_encoded:
                self.ra_encoded = self.ra_encoded[indices]
            
            
        
        self.patient_cnt = len(set(self.patient_id))
        
        self.seq_cnt = len(self.labels)
        self.seq_per_patient = self.seq_cnt // self.patient_cnt
  
    
    def get_values_with_step(self, A, d, B):
        # 为每个A中的数生成两个连续索引
        indices = np.concatenate([(A*d)[:, None] + np.arange(d), ], axis=1).flatten()
        # 从B中取值
        return B[indices]
    
    def get_independent_dataset(self, idxs):
        if self.is_seq:
            raise ValueError("Not supported for sequence format")
        else:
            mask = idxs < self.patient_cnt
            idxs = idxs[mask]
            self.clone_fractions = self.get_values_with_step(idxs, self.seq_per_patient, self.clone_fractions)
            self.labels = self.get_values_with_step(idxs, self.seq_per_patient, self.labels)
            self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            self.bert_mask = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_mask)
            self.patient_id = self.get_values_with_step(idxs, self.seq_per_patient, self.patient_id)
            self.cdr3_seq = self.get_values_with_step(idxs, self.seq_per_patient, self.cdr3_seq)
            if self.is_full:
                self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            if self.use_ra_encoded:
                self.ra_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.ra_encoded)

        
            self.patient_cnt = len(set(self.patient_id))
            self.seq_cnt = len(self.labels)
            self.seq_per_patient = self.seq_cnt // self.patient_cnt
            
    
    
    def get_divided_dataset(self, divided_idxs):

        chosen_idxs = np.array(divided_idxs)
        self.bert_encoded = self.bert_encoded[chosen_idxs]
        self.bert_mask = self.bert_mask[chosen_idxs]
        self.labels = self.labels[chosen_idxs]
        self.clone_fractions = self.clone_fractions[chosen_idxs]
        self.patient_id = self.patient_id[chosen_idxs]
        self.cdr3_seq = self.cdr3_seq[chosen_idxs]
        if self.use_ra_encoded:
            self.ra_encoded = self.ra_encoded[chosen_idxs]
            
            
    def set_divided_mode(self, divided_mode=False, x_idx=None, u_idx=None, prob=None):
        self.x_idx = x_idx
        self.u_idx = u_idx
        self.is_divided = divided_mode
        self.prob = prob
    
    
    def get_corpus_line(self, idx):
        
        if self.is_seq:
            if self.is_divided is False:
                SID = idx
                FID = self.patient_id[idx]
                label = self.labels[idx]
                cdr3 = (self.cdr3_seq[idx]).decode('utf-8')
                bert_encoded = self.bert_encoded[idx]
                bert_mask = self.bert_mask[idx]
                cloneFraction = self.clone_fractions[idx]
                seq_len = len(cdr3)
                if self.use_ra_encoded:
                    ra_encoded = self.ra_encoded[idx]
                    if self.use_noisy_labels:
                        noisy_label = self.noisy_labels[idx]
                        return SID, FID, noisy_label, label, cdr3, bert_encoded, bert_mask, ra_encoded, seq_len, cloneFraction
                    else:
                        return SID, FID, label, cdr3, bert_encoded, bert_mask, ra_encoded, seq_len, cloneFraction
                
                else:
                    if self.use_noisy_labels:
                        noisy_label = self.noisy_labels[idx]
                        return SID, FID, noisy_label, label, cdr3, bert_encoded, bert_mask, seq_len, cloneFraction
                    else:
                        return SID, FID, label, cdr3, bert_encoded, bert_mask, seq_len, cloneFraction
                
                
            else:
                idx_x = self.x_idx[idx]
                idx_u = self.u_idx[idx%len(self.u_idx)]
                SID_u = idx_u
                FID_u = self.patient_id[idx_u]
                SID_x = idx_x
                FID_x = self.patient_id[idx_x]
                prob_x = self.prob[idx]
                label_x = self.labels[idx_x]
                cdr3_x = (self.cdr3_seq[idx_x]).decode('utf-8')
                cdr3_u = (self.cdr3_seq[idx_u]).decode('utf-8')
                label_u = self.labels[idx_u]

                bert_encoded_x = self.bert_encoded[idx_x]
                bert_encoded_u = self.bert_encoded[idx_u]
                
                bert_mask_x = self.bert_mask[idx_x]
                bert_mask_u = self.bert_mask[idx_u]
            
                return SID_x, SID_u, FID_x, FID_u, label_x, label_u, bert_encoded_x, bert_encoded_u, bert_mask_x, bert_mask_u, cdr3_x, cdr3_u, prob_x
        else:
            FID = idx
            start_idx = idx * self.seq_per_patient
            l = self.seq_per_patient
            SID = None
            labels = self.labels[start_idx: start_idx + l]
            cdr3 = self.cdr3_seq[start_idx: start_idx + l]
            cdr3s = [x.decode('utf-8') for x in cdr3]
            bert_encodeds = self.bert_encoded[start_idx: start_idx + l]
            cloneFractions = self.clone_fractions[start_idx: start_idx + l]
            seq_lens = [len(x) for x in cdr3]
            if self.use_ra_encoded:
                ra_encodeds = self.ra_encoded[start_idx: start_idx + l]
                return SID, FID, labels, cdr3s, bert_encodeds, ra_encodeds, seq_lens, cloneFractions
            else:
                return SID, FID, labels, cdr3s, bert_encodeds, seq_lens, cloneFractions

 
    
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    
    def get_noisy_labels(self, n_l):
        self.noisy_labels = n_l
        self.use_noisy_labels = True
    
    def use_purified_labels(self, purified_labels, non_cancer_label = 3):
        cancer_indices = (self.labels != non_cancer_label)
        non_cancer_indices = (self.labels == non_cancer_label)
        # if self.exclude_indices is not None:
        #     purified_labels = purified_labels[self.exclude_indices[:len(purified_labels)]]
            
        self.labels[cancer_indices] = purified_labels
        self.labels[non_cancer_indices] = 0

    
    def __len__(self):
        if self.is_seq:
            if self.is_divided is False:
                return self.seq_cnt
            else:
                return len(self.x_idx)
        else:
            return self.patient_cnt

    
    def __getitem__(self, idx):
        
        if self.is_divided is False:
            if self.use_ra_encoded is False:
                
                if self.use_noisy_labels is False:
                    SID, FID, label, cdr3, bert_encoded, bert_mask,seq_len, cloneFraction = self.get_corpus_line(idx)
                    return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32),\
                        torch.tensor(bert_mask, dtype=torch.float32),\
                        seq_len, torch.tensor(cloneFraction, dtype=torch.float32)
                else:
                    SID, FID, noisy_label, label, cdr3, bert_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                    return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(noisy_label, dtype=torch.long),torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                    
            else:
                if self.use_noisy_labels is False:
                    SID, FID, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                    return int(SID) if SID is not None else self.magic_number,\
                        int(FID) if FID is not None else self.magic_number,\
                        cdr3, torch.tensor(label, dtype=torch.long),\
                            torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                            torch.tensor(cloneFraction, dtype=torch.float32)
                    
                else:
                    SID, FID, noisy_label, label, cdr3, bert_encoded, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                    return int(SID) if SID is not None else self.magic_number,\
                        int(FID) if FID is not None else self.magic_number,\
                        cdr3, torch.tensor(noisy_label, dtype=torch.long), torch.tensor(label, dtype=torch.long),\
                            torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                            torch.tensor(cloneFraction, dtype=torch.float32)
        
        else:
            SID_x, SID_u, FID_x, FID_u, label_x, label_u, bert_encoded_x, bert_encoded_u, bert_mask_x, bert_mask_u, cdr3_x, cdr3_u, prob_x = self.get_corpus_line(idx)
            return int(SID_x) if SID_x is not None else self.magic_number,\
                int(SID_u) if SID_u is not None else self.magic_number,\
                int(FID_x) if FID_x is not None else self.magic_number,\
                int(FID_u) if FID_u is not None else self.magic_number,\
                torch.tensor(label_x, dtype=torch.long), torch.tensor(label_u, dtype=torch.long),\
                    torch.tensor(bert_encoded_x, dtype=torch.float32), torch.tensor(bert_encoded_u, dtype=torch.float32), \
                    torch.tensor(bert_mask_x, dtype=torch.float32), torch.tensor(bert_mask_u, dtype=torch.float32), cdr3_x, cdr3_u, prob_x
                    
                    
                    
                    
                    
                    
class B62_embs_dataset(Dataset):
    def __init__(self,h5_file_path, is_seq=True, is_full = False, use_ra_encoded = False, exclude_k = None):
        self.is_seq = is_seq
        self.h5_file = h5_file_path
        self.magic_number = 114514
        self.is_full = is_full
        self.use_ra_encoded = use_ra_encoded
        self.use_noisy_labels = False
        self.exclude_indices = None
        self.load_h5_file(exclude_k)


    
    def load_h5_file(self, k=None):
      with h5py.File(self.h5_file, 'r') as f:
        self.clone_fractions = f['clone_fractions'][:]
        self.labels = f['labels'][:]
        self.bert_encoded = f['cdr3_bert_encoded'][:]
        self.bert_mask = f['cdr3_bert_mask'][:]
        self.patient_id = f['patient_id'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.sequence_id = f['sequence_id'][:]
        if self.is_full:
            self.bert_encoded = f['cdr3_full_bert_encoded'][:]
        if self.use_ra_encoded:
            self.ra_encoded = f['ra_encoded'][:]
        
        if k is not None:
            # 处理单个数值或列表输入
            if isinstance(k, (int, float)):
                indices = self.labels != k
            else:  # k是列表
                indices = ~np.zeros_like(self.labels, dtype=bool)
                for label in k:
                    indices = np.logical_and(indices, self.labels != label)
            
            self.exclude_indices = indices
            
            # 应用过滤
            self.clone_fractions = self.clone_fractions[indices]
            self.labels = self.labels[indices]
            self.bert_encoded = self.bert_encoded[indices]
            self.bert_mask = self.bert_mask[indices]
            self.patient_id = self.patient_id[indices]
            self.cdr3_seq = self.cdr3_seq[indices]
            if self.is_full:
                self.bert_encoded = self.bert_encoded[indices]
            if self.use_ra_encoded:
                self.ra_encoded = self.ra_encoded[indices]
            
            
        
        self.patient_cnt = len(set(self.patient_id))
        
        self.seq_cnt = len(self.labels)
        self.seq_per_patient = self.seq_cnt // self.patient_cnt
  
    
    def get_values_with_step(self, A, d, B):
        # 为每个A中的数生成两个连续索引
        indices = np.concatenate([(A*d)[:, None] + np.arange(d), ], axis=1).flatten()
        # 从B中取值
        return B[indices]
    
    def get_independent_dataset(self, idxs):
        if self.is_seq:
            raise ValueError("Not supported for sequence format")
        else:
            mask = idxs < self.patient_cnt
            idxs = idxs[mask]
            self.clone_fractions = self.get_values_with_step(idxs, self.seq_per_patient, self.clone_fractions)
            self.labels = self.get_values_with_step(idxs, self.seq_per_patient, self.labels)
            self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            self.bert_mask = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_mask)
            self.patient_id = self.get_values_with_step(idxs, self.seq_per_patient, self.patient_id)
            self.cdr3_seq = self.get_values_with_step(idxs, self.seq_per_patient, self.cdr3_seq)
            if self.is_full:
                self.bert_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.bert_encoded)
            if self.use_ra_encoded:
                self.ra_encoded = self.get_values_with_step(idxs, self.seq_per_patient, self.ra_encoded)

        
            self.patient_cnt = len(set(self.patient_id))
            self.seq_cnt = len(self.labels)
            self.seq_per_patient = self.seq_cnt // self.patient_cnt
            
    def get_corpus_line(self, idx):
        
        if self.is_seq:
            SID = idx
            FID = self.patient_id[idx]
            label = self.labels[idx]
            
            cdr3 = (self.cdr3_seq[idx]).decode('utf-8')
            bert_encoded = self.bert_encoded[idx]
            bert_mask = self.bert_mask[idx]
            cloneFraction = self.clone_fractions[idx]
            seq_len = len(cdr3)
            if self.use_ra_encoded:
                ra_encoded = self.ra_encoded[idx]
                if self.use_noisy_labels:
                    noisy_label = self.noisy_labels[idx]
                    return SID, FID, noisy_label, label, cdr3, bert_encoded, bert_mask, ra_encoded, seq_len, cloneFraction
                else:
                    return SID, FID, label, cdr3, bert_encoded ,bert_mask , ra_encoded, seq_len, cloneFraction
            else:
                if self.use_noisy_labels:
                    noisy_label = self.noisy_labels[idx]
                    return SID, FID, noisy_label, label, cdr3, bert_encoded, bert_mask, seq_len, cloneFraction
                else:
                    return SID, FID, label, cdr3, bert_encoded, bert_mask, seq_len, cloneFraction
        else:
            FID = idx
            start_idx = idx * self.seq_per_patient
            l = self.seq_per_patient
            SID = None
            labels = self.labels[start_idx: start_idx + l]
            cdr3 = self.cdr3_seq[start_idx: start_idx + l]
            cdr3s = [x.decode('utf-8') for x in cdr3]
            bert_encodeds = self.bert_encoded[start_idx: start_idx + l]
            bert_masks = self.bert_mask[start_idx: start_idx + l]
            cloneFractions = self.clone_fractions[start_idx: start_idx + l]
            seq_lens = [len(x) for x in cdr3]
            if self.use_ra_encoded:
                ra_encodeds = self.ra_encoded[start_idx: start_idx + l]
                return SID, FID, labels, cdr3s, bert_encodeds, bert_masks, ra_encodeds, seq_lens, cloneFractions
            else:
                return SID, FID, labels, cdr3s, bert_encodeds, bert_masks, seq_lens, cloneFractions

 
    
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    
    def get_noisy_labels(self, n_l):
        self.noisy_labels = n_l
        self.use_noisy_labels = True
    
    def use_purified_labels(self, purified_labels, non_cancer_label = 3):
        cancer_indices = (self.labels != non_cancer_label)
        non_cancer_indices = (self.labels == non_cancer_label)
        # if self.exclude_indices is not None:
        #     purified_labels = purified_labels[self.exclude_indices[:len(purified_labels)]]
            
        self.labels[cancer_indices] = purified_labels
        self.labels[non_cancer_indices] = 0

    
    def __len__(self):
        if self.is_seq:
            return self.seq_cnt
        else:
            return self.patient_cnt

    
    def __getitem__(self, idx):
        
        if self.use_ra_encoded is False:
            
            if self.use_noisy_labels is False:
                SID, FID, label, cdr3, bert_encoded, bert_mask, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long),\
                    torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(bert_mask, dtype=torch.float32), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
            else:
                SID, FID, noisy_label, label, cdr3, bert_encoded, bert_mask, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(noisy_label, dtype=torch.long),torch.tensor(label, dtype=torch.long),\
                    torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(bert_mask, dtype=torch.float32), seq_len, \
                    torch.tensor(cloneFraction, dtype=torch.float32)
                   
        else:
            if self.use_noisy_labels is False:
                SID, FID, label, cdr3, bert_encoded, bert_mask, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(bert_mask, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                
            else:
                SID, FID, noisy_label, label, cdr3, bert_encoded, bert_mask, ra_encoded, seq_len, cloneFraction = self.get_corpus_line(idx)
                return int(SID) if SID is not None else self.magic_number,\
                    int(FID) if FID is not None else self.magic_number,\
                    cdr3, torch.tensor(noisy_label, dtype=torch.long), torch.tensor(label, dtype=torch.long),\
                        torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(bert_mask, dtype=torch.float32), torch.tensor(ra_encoded, dtype=torch.float32), seq_len, \
                        torch.tensor(cloneFraction, dtype=torch.float32)
                        
                        
                        
                        
                        
                        
class Meta_B62_embs_dataset(Dataset):
    def __init__(self,h5_file_path, is_full = False, is_seq = True, topk = 100, is_use_Sparse = False):

        self.h5_file = h5_file_path
        self.magic_number = 114514
        self.is_full = is_full
        self.is_seq = is_seq
        self.topk = topk
        self.is_use_Sparse = is_use_Sparse
        self.load_h5_file()
        
    
    def load_h5_file(self):
      with h5py.File(self.h5_file, 'r') as f:
        self.labels = f['labels'][:]
        self.bert_encoded = f['cdr3_bert_encoded'][:]
        self.bert_mask = f['cdr3_bert_mask'][:]
        self.cdr3_seq = f['cdr3_seq'][:]
        self.sequence_id = f['sequence_id'][:]
        self.seq_cnt = len(self.labels)
        if self.is_full:
            self.bert_encoded = f['cdr3_full_bert_encoded'][:]
        self.patien_cnt = self.seq_cnt // self.topk
        if self.is_use_Sparse:
            self.sparse_encoded = f['sparse_encoded'][:]
        

    def get_corpus_line(self, idx):
        if self.is_seq:
            SID = idx
            FID = None
            label = self.labels[idx]
            cdr3 = (self.cdr3_seq[idx]).decode('utf-8')
            bert_encoded = self.bert_encoded[idx]
            bert_mask = self.bert_mask[idx]
            seq_len = len(cdr3)
            if self.is_use_Sparse is False:
                return SID, FID, label, cdr3, bert_encoded, bert_mask, seq_len
            else:
                sparse_encoded = self.sparse_encoded[idx]
                return SID, FID, label, cdr3, bert_encoded, bert_mask, sparse_encoded, seq_len
        else:
            FID = idx
            start_idx = idx * self.topk
            l = self.topk
            SID = None
            labels = self.labels[start_idx: start_idx + l]
            cdr3 = self.cdr3_seq[start_idx: start_idx + l]
            cdr3s = [x.decode('utf-8') for x in cdr3]
            bert_encodeds = self.bert_encoded[start_idx: start_idx + l]
            bert_masks = self.bert_mask[start_idx: start_idx + l]
            seq_lens = [len(x) for x in cdr3]
            if self.is_use_Sparse is False:
                return SID, FID, labels, cdr3s, bert_encodeds, bert_masks, seq_lens
            else:
                sparse_encodeds = self.sparse_encoded[start_idx: start_idx + l]
                return SID, FID, labels, cdr3s, bert_encodeds, bert_masks, sparse_encodeds, seq_lens
    
    
    
    def set_seqOrNot(self, is_seq):
        self.is_seq = is_seq
    
    def __len__(self):
        if self.is_seq:
            return self.seq_cnt
        else:
            return self.patien_cnt

    def __getitem__(self, idx):
        
        if self.is_use_Sparse is False:  
            SID, FID, label, cdr3, bert_encoded, bert_mask, seq_len = self.get_corpus_line(idx)
            return int(SID) if SID is not None else self.magic_number, \
            int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long),\
                torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(bert_mask, dtype=torch.float32), seq_len
        else:
            SID, FID, label, cdr3, bert_encoded, bert_mask, sparse_encoded, seq_len = self.get_corpus_line(idx)
            return int(SID) if SID is not None else self.magic_number, \
            int(FID) if FID is not None else self.magic_number,\
                cdr3, torch.tensor(label, dtype=torch.long),\
                torch.tensor(bert_encoded, dtype=torch.float32), torch.tensor(bert_mask, dtype=torch.float32), torch.tensor(sparse_encoded, dtype=torch.float32), seq_len