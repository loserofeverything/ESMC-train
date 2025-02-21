import os
import time
from dataset import *
from torch.utils.data import DataLoader
import tqdm
import h5py
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig
from esm.models.esmc import ESMC


class LabelEncoder:
    def __init__(self):
        self.class_to_index = {}
        self.index_to_class = {}

    def fit(self, label_list):

        self.class_to_index = {label: index for index, label in enumerate(label_list)}
        self.index_to_class = {index: label for label, index in self.class_to_index.items()}

    def transform(self, label_list):
        return [self.class_to_index[label] for label in label_list]


def fill_sublists(A):
    # 找到最长子列表的长度
    max_length = max(len(sublist) for sublist in A)

    # 填充每个子列表
    for sublist in A:
        sublist.extend(['X'] * (max_length - len(sublist)))
    return A


def collate_fn_loadencoded(batch):
    sids, fids, cdr3s, labels, ra_encoded, seq_lens, cloneFractions = zip(*batch)
    # 提取每个张量的唯一值（实际上就是每个张量的第一个元素）
    # unique_labels = torch.tensor([tensor[0] for tensor in labels])
    labels = torch.cat([t.view(-1) for t in labels])
    sids = torch.tensor(sids)
    fids = torch.tensor(fids)
    cloneFractions = torch.cat([t.view(-1) for t in cloneFractions])

    return sids, fids, cdr3s, labels, cloneFractions


def collate_fn(batch):
    sids, fids, cdr3s, labels, seq_lens, cloneFractions = zip(*batch)
    # 提取每个张量的唯一值（实际上就是每个张量的第一个元素）
    # unique_labels = torch.tensor([tensor[0] for tensor in labels])
    labels = torch.cat([t.view(-1) for t in labels])
    sids = torch.tensor(sids)
    fids = torch.tensor(fids)
    cloneFractions = torch.cat([t.view(-1) for t in cloneFractions])

    return sids, fids, cdr3s, labels, cloneFractions


# 示例使用
if __name__ == "__main__":
    
    max_Seq_l = 25
    chosen_labels = ('Breast','Lung','Colorectal', 'Healthy', 'Lihc')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_label_names_sub = ['Breast','Lung','Colorectal', 'Healthy', 'Lihc','Prostate', 'Pancreas']
    _type = "_".join(total_label_names_sub)
    label_names_sub = ['Breast','Lung','Colorectal', 'Healthy', 'Lihc']
    label_encoder = LabelEncoder()
    label_encoder.fit(label_names_sub)
    print(label_encoder.transform(label_names_sub))
    
    
    
    split = 5
    topk_used = 5000
    is_load_encoded = False
    is_start_making_h5 = True
    is_load = True

    
    
    start_time = time.time()
    test_dataset = dataset(None, label_encoder=label_encoder, mode = 'test', \
        topk = topk_used, is_load=is_load, split=split,h5_root_dir="/root/autodl-tmp/12_25_balanced_long", classname='lab', \
            is_load_encoded=is_load_encoded)
    end_time = time.time()
    print("Time taken to load test dataset: ", end_time - start_time)
    
    _h5 = '/root/autodl-tmp/ESM/data/esmc-l{}_dataset'.format(max_Seq_l)
    h5_root_dir = os.path.join(_h5, '12_25balanced_long-without-benign-meanPooledOut')
    h5_folder = "{}split_topk{}_dataset_{}".format(split, int(topk_used), 'lab')
    h5_path = os.path.join(h5_root_dir, h5_folder)
    os.makedirs(h5_path, exist_ok=True)
    # if not os.path.exists(h5_path):
    #     os.makedirs(h5_path)
    # train_h5_file = os.path.join(h5_path, "{}_data.h5".format("train"))
    test_h5_file = os.path.join(h5_path, "{}_data.h5".format("test"))     
    
    is_use_fullBertoutputs = False
    is_view_data = False
    # bert_folder = "bert-base"
    # model = BertTCR.from_pretrained(bert_folder)
    # print(model.config)
    
    if is_start_making_h5 and \
        os.path.exists(test_h5_file) is False:
        bert_folder = "bert-base"
        model = ESMC.from_pretrained("esmc_300m")
        model = model.to(device)
        # train_dataset.set_seqOrNot(True)
        test_dataset.set_seqOrNot(True)
        if is_load_encoded:
            # train_loader = DataLoader(train_dataset, batch_size=200, shuffle=False,collate_fn=collate_fn_loadencoded, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False,collate_fn=collate_fn_loadencoded, num_workers=4)
        else:
            # train_loader = DataLoader(train_dataset, batch_size=200, shuffle=False,collate_fn=collate_fn, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False,collate_fn=collate_fn, num_workers=4)

        sids = []
        fids = []
        cdr3s = []
        labels = []
        cloneFractions = []
        Esm_embs = []
        full_bert_embs = []
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            model.eval()

            for sid, fid, cdr3, label, cloneFraction in tqdm.tqdm(test_loader, \
            desc="embedding test set", total=len(test_loader)):
                batch_size = label.shape[0]
                input_ids = model._tokenize(cdr3)
                output = model(input_ids)
                logits, embeddings, hiddens = (
                    output.sequence_logits,
                    output.embeddings,
                    output.hidden_states,
                )
                esm_emb = torch.mean(embeddings, dim=1)
                esm_emb = esm_emb.to(torch.float32).cpu().detach().numpy()
                Esm_emb = []

                for idx, seq in enumerate(cdr3):
                    if seq[0] == "X":
                        Esm_emb.append(np.zeros((960, ), dtype=np.float32))
                    else:
                        Esm_emb.append(esm_emb[idx])
         
                Esm_emb = np.array(Esm_emb)
                Esm_embs.append(Esm_emb)
                sids.extend(sid.numpy().tolist())
                fids.extend(fid.numpy().tolist())
                cdr3s.extend(cdr3)
                labels.extend(label.numpy().tolist())
                cloneFractions.extend(cloneFraction.numpy().tolist())



            Esm_embs = np.vstack(Esm_embs)

            with h5py.File(test_h5_file, 'w') as f:

                f.create_dataset('clone_fractions', data=np.array(cloneFractions, dtype=np.float32))
                f.create_dataset('cdr3_bert_encoded', data=np.array(Esm_embs, dtype=np.float32))
                f.create_dataset('labels', data=np.array(labels, dtype=np.int32))
                f.create_dataset('patient_id', data=np.array(fids, dtype=np.int32))
                f.create_dataset('sequence_id', data=np.array(sids, dtype=np.int32))
                f.create_dataset('cdr3_seq', data=np.array(cdr3s, dtype=h5py.string_dtype(encoding='utf-8')))
        

    if is_view_data:
        with h5py.File(test_h5_file, 'r') as f:
            clone_fractions = f['clone_fractions'][:]
            labels = f['labels'][:]
            cdr3_encoded = f['cdr3_bert_encoded'][:]
            patient_id = f['patient_id'][:]
            cdr3_seq = f['cdr3_seq'][:]
            sequence_id = f['sequence_id'][:]

       
        
        
    