import pandas as pd
import dgl.function as fn
import numpy as np
import torch

def cosine_sim(alpha):
    data_path = '.\data\wuhan431\/'
    #余弦相似度计算结果
    sim = pd.read_csv(data_path + 'embedding\wuhan_emb_200_with_label_75.1_cosine.csv').iloc[:, 1:]
    count = 0
    sim_beg = list()
    sim_end = list()
    for row_index, row in sim.iterrows():
        print("computing:", row_index, end='\r')
        for col_index, col_values in row.iteritems():
            if col_values > alpha:
                count += 1
                sim_beg.append(row_index)
                sim_end.append(int(col_index)-1)
    pic_sim = pd.concat([pd.DataFrame(sim_beg, columns=['pic_beg']), pd.DataFrame(sim_end, columns=['pic_end'])], axis=1)
    pic_sim.to_csv(data_path + 'pic_sim_'+str(alpha)+'.csv')
    print("[pic_sim generate done!]")

def init_parcel(hetero_graph, embedding):
    funcs = dict()
    #一阶邻居均值初始化
    funcs['fall'] = (fn.copy_u('feature', 'm'), fn.mean('m', 'h'))
    hetero_graph.multi_update_all(funcs, 'copy')
    hetero_graph.nodes['parcel'].data['feature'] = hetero_graph.nodes['parcel'].data['h']
    parcel = pd.DataFrame(hetero_graph.nodes['parcel'].data['feature'].numpy())
    pd.DataFrame(hetero_graph.nodes['parcel'].data['feature'].numpy()).to_csv('.\data\wuhan431\/mean_two_comparative.csv')

    #nonSVI-parcel 随机初始化
    nonzero_list = []
    for i, row in enumerate(parcel.values):
        if np.sum(row) != 0:
            nonzero_list.append(i)
    parcel_list = np.array(parcel).tolist()

    init_p_list = []
    for line in range(len(parcel.index)):
        if line in nonzero_list: #SVI-parcels
            init_p_list.append(parcel_list[line])
        else:
            non_SVIs = np.random.randn(1, embedding).squeeze().tolist()
            init_p_list.append(non_SVIs)
    pd.DataFrame(init_p_list).to_csv(".\data\wuhan431\/init_parcel.csv",index=None)

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)['parcel']
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item()*1.0 / len(labels)