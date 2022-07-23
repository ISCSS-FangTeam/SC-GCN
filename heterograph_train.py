import numpy as np
import torch
import dgl
import torch.nn.functional as F
from heterograph_model import scgcn
import pandas as pd
import argparse
import time
from tensorboardX import SummaryWriter
from util import cosine_sim, init_parcel, evaluate

def main(args):
    # parcel and img
    n_parcel = 1317
    n_img = 1810

    n_hetero_features = dimension
    n_parcel_classes = 5 #分类类别
    data_path = '.\data\wuhan431\/'

    train_idx_parcel = torch.tensor(np.array(pd.read_csv(data_path+"train_mask.csv")))
    test_idx_parcel = torch.tensor(np.array(pd.read_csv(data_path+'test_mask.csv')))
    train_idx_parcel = train_idx_parcel.squeeze()
    test_idx_parcel = test_idx_parcel.squeeze()

    fall = pd.read_csv(data_path + 'fall.csv')

    fall_src = fall[['fall_src']].values.squeeze()
    fall_dst = np.array(fall[['fall_dst']]).squeeze()

    #for parcel-parcel edges
    adjoin = pd.read_csv(data_path + 'adjoin.csv')
    adjoin_src = adjoin[['IN_id']].values.squeeze()
    adjoin_dst = adjoin[['NEAR_id']].values.squeeze()

    cosine_sim(args.alpha)
    pic_sim = pd.read_csv(data_path + 'pic_sim_'+str(args.alpha)+'.csv')
    sim_beg = pic_sim[['pic_beg']].values.squeeze()
    sim_end = pic_sim[['pic_end']].values.squeeze()

    #异构图
    hetero_graph = dgl.heterograph({
        ('img', 'fall', 'parcel'): (fall_src, fall_dst),
        ('parcel', 'fall-by', 'img'): (fall_dst, fall_src),
        ('parcel', 'adjoin', 'parcel'): (adjoin_src, adjoin_dst),
        ('parcel', 'adjoin-by', 'parcel'): (adjoin_dst, adjoin_src),
        ('img', 'same', 'img'): (sim_beg, sim_end)},
        {'parcel': n_parcel, 'img': n_img})

    feats = pd.read_csv(data_path+'embedding\/feats_' + str(dimension) + '.csv')
    img_feats = feats.iloc[:, 6:dimension + 6]
    img_label = feats.loc[:, 'gt']

    hetero_graph.nodes['img'].data['feature'] = torch.from_numpy(img_feats.values)
    hetero_graph.nodes['img'].data['label'] = torch.from_numpy(img_label.values)
    hetero_graph.nodes['parcel'].data['feature'] = torch.randn(n_parcel, dimension)

    parcel = pd.read_csv('.\data\wuhan\/gt.csv')
    parcel_label = parcel[['old_landuse']].values.squeeze()

    hetero_graph.nodes['parcel'].data['label'] = torch.from_numpy(parcel_label)
    hetero_graph.nodes['parcel'].data['feature'] = torch.from_numpy(pd.read_csv(data_path + 'init_parcel.csv').values)
    print("parcel_feats: ", hetero_graph.nodes['parcel'].data['feature'])

    #地块、街景特征获取如下：
    parcel_feats = torch.tensor(hetero_graph.nodes['parcel'].data['feature'], dtype=torch.float32)
    img_feats = torch.tensor(hetero_graph.nodes['img'].data['feature'], dtype=torch.float32)

    #parcel label
    labels = hetero_graph.nodes['parcel'].data['label']
    #img label
    labels_img = hetero_graph.nodes['img'].data['label']
    print(labels_img)

    #前向传播计算
    node_feature = {'img': img_feats, 'parcel': parcel_feats}
    print(hetero_graph.etypes)
    model = scgcn(n_hetero_features, hid_feats=80, out_feats=n_parcel_classes, rel_names=hetero_graph.etypes)
    writer_loss = SummaryWriter(comment='Loss'+str(args.lr))
    writer_acc = SummaryWriter(comment='Accuracy_train'+str(args.lr))
    print("hetero_graph", hetero_graph)


#scgcn training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("starting training...")
    dur = []
    best_parcel_acc = 0.
    loss_t = 0.
    acc_t = 0.
    for epoch in range(args.n_epochs):
        if epoch > 5:
            t0 = time.time()
        model.train()
        logits_parcel = model(hetero_graph, node_feature)['parcel']
        loss_parcel = F.cross_entropy(logits_parcel[train_idx_parcel], labels[train_idx_parcel])
        loss = loss_parcel
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        acc = evaluate(model, hetero_graph, node_feature, labels, test_idx_parcel)

        loss_t += loss
        acc_t += acc

        if epoch % 10 == 0:
            writer_loss.add_scalar('CE-Loss', loss_t / 10, global_step=epoch)
            writer_acc.add_scalar('train_acc', acc_t/10, global_step=epoch)
            loss_t = 0.
            acc_t = 0.
        if epoch > 5:
            dur.append(t1-t0)
        if args.model_path is not None:
            torch.save(model.state_dict(), args.model_path)
        if acc > best_parcel_acc:

            best_parcel_acc = acc
            best_checkpoint_path = ".\checkpoint\emb_"+str(dimension)+"\/best_checkpoint.pth"
            torch.save(model.state_dict(), best_checkpoint_path)

        print("Epoch {:05d} | Parcel acc: {:.4f} | Parcel best acc: {:.4f} | Train loss: {:.4f} | Time: {:.4f}".
            format(epoch, acc, best_parcel_acc, loss, np.average(dur)))

#test
    model.eval()
    with torch.no_grad():
        logits = model.forward(hetero_graph, node_feature)['parcel']
        tmp = pd.DataFrame(logits.data.numpy())
        tmp['predict'] = np.argmax(tmp.values[:, 0:5], axis=1)
        tmp = pd.concat([pd.DataFrame(parcel.iloc[:, 1:-1]), tmp], axis=1)

        parcel_loss = F.cross_entropy(logits[train_idx_parcel], labels[train_idx_parcel])
        parcel_test_acc = torch.sum(logits[test_idx_parcel].argmax(dim=1) == labels[test_idx_parcel]).item() / len(test_idx_parcel)
        parcel_total_acc = torch.sum(logits[:].argmax(dim=1) == labels[:]).item() / (n_parcel)

        print("Parcel test Acc: {:.4f} | Parcel total Acc: {:.4f} | Parcel loss: {:.4f}".
            format(parcel_test_acc, parcel_total_acc, parcel_loss))


if __name__ == '__main__':
    dimension = 200
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument('-e', '--n-epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='leaning rate')
    parser.add_argument('--model_path', type=str, default=".\checkpoint\emb_"+str(dimension)+"\checkpoint.pth",
                        help='path for save the model')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='the threshold of picture similarity')

    args = parser.parse_args()
    main(args)