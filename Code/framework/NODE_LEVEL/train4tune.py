import sys
import numpy as np
import torch
import utils
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
from datasets import get_dataset
from model import NetworkGNN as Network
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


import logging
from sklearn.metrics import pairwise_distances
from torch_scatter import scatter_mean,scatter_sum



def main(exp_args, run=0):
    global train_args
    train_args = exp_args

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    #np.random.seed(train_args.seed)
    torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(train_args.seed)

    split = True
    if train_args.data in ['Reddit', 'arxiv', 'flickr', 'arxiv_full']:
        split = False
    # print('split_data:', split)
    dataset_name = train_args.data
    feature_engineerings = exp_args.FEATURE_ENGINEERING
    data, num_features, num_classes = get_dataset(dataset_name, feature_engineerings, split=split, run=run)
    # print(data.x.size(), data.edge_index.size(), num_classes, num_features)
    data = data.to(device)

    if train_args.data in ['ogbn-proteins', 'genius']:
        criterion = torch.nn.BCEWithLogitsLoss()
        metric = 'rocauc'
    else:
        criterion = nn.CrossEntropyLoss()
        metric = 'acc'  

    # criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    genotype = train_args.arch
    hidden_size = train_args.hidden_size

    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=train_args.num_layers, dropout=train_args.dropout,
                    act=train_args.activation, args=train_args)

    model = model.cuda()
    num_parameters = np.sum(np.prod(v.size()) for name, v in model.named_parameters())
    # print('params size:', num_parameters)
    logging.info("genotype=%s, param size = %fMB, args=%s", genotype, utils.count_parameters_in_MB(model), train_args.__dict__)

    if train_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
        )
    elif train_args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            train_args.learning_rate,
            momentum=train_args.momentum,
            weight_decay=train_args.weight_decay
        )
    elif train_args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs), eta_min=train_args.min_lr)
    best_val_acc = best_test_acc = 0
    results = []
    best_line = 0


    # for epoch in tqdm(range(train_args.epochs),  position=0):
    for epoch in range(train_args.epochs):
        train_loss, train_acc, train_mad = train_trans(data, model, criterion, optimizer, metric)

        if train_args.cos_lr:
            scheduler.step()

        valid_loss, valid_acc, val_mad, test_loss, test_acc, test_mad = infer_trans(data, model, criterion, metric)
        results.append([valid_loss, valid_acc, val_mad, test_loss, test_acc, test_mad])

        if epoch%10 == 0:
            # print(f"epoch:{epoch},train_loss:{train_loss:.04f},valid_acc:{valid_acc:.04f},test_acc:{test_acc:.04f}")
            print(f"[Epoch:{epoch}] Train Loss:{train_loss:.04f} | Valid Acc:{valid_acc:.04f} | Test Acc:{test_acc:.04f}")
            
            


        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_test_acc = test_acc
            best_line = epoch

        
        # logging.info(
        #     'epoch=%s, lr=%s, train_loss=%s, train_acc=%f, valid_acc=%s, test_acc=%s, best_val_acc=%s, best_test_acc=%s, train_mad= %s, val_mad=%s, test_mad=%s',
        #     epoch, scheduler.get_last_lr(), train_loss, train_acc, valid_acc, test_acc, best_val_acc, best_test_acc, train_mad, val_mad, test_mad)
    print(
        # 'Best_results: epoch={}, val_loss={:.04f}, valid_acc={:.04f}, test_loss:{:.04f},test_acc={:.04f}, val_mad:{:.04f},test_mad:{:.04f},'.format(
        #     best_line, results[best_line][0], results[best_line][1], results[best_line][3], results[best_line][4], results[best_line][2], results[best_line][5])
            'Best results: epoch={}, valid loss={:.04f}, valid acc={:.04f}, test loss:{:.04f},test acc={:.04f}'.format(
            best_line, results[best_line][0], results[best_line][1], results[best_line][3], results[best_line][4]))

    return best_val_acc, best_test_acc, train_args

def get_perf(metric, logits, data, mask):
    if metric == 'acc':
        accuracy = 0
        # print("Logits shape:", logits.shape)
        # print("Mask shape:", mask.shape)
        accuracy += logits[mask].max(1)[1].eq(data.y[mask]).sum().item() / mask.sum().item()
    
    elif metric == 'rocauc':
        rocauc_list = []
        y_true = data.y[mask].cpu().detach()
        y_pred = logits[mask].cpu().detach()



        # print('size:', logits.size(), data.y.size(), data.y[0], data.y[mask].shape)
        # for i in range(y_true.shape[1]):
        #     #AUC is only defined when there is at least one positive data.
        #     if torch.sum(y_true[:,i] == 1) > 0 and torch.sum(y_true[:,i] == 0) > 0:
        #         is_labeled = y_true[:,i] == y_true[:,i]
        #         rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        
        y_true = y_true.detach().cpu().numpy()
        if y_true.shape[1] == 1:
            # use the predicted class for single-class classification
            y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
        else:
            y_pred = y_pred.detach().cpu().numpy()

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_labeled = y_true[:, i] == y_true[:, i]
                score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

                rocauc_list.append(score)   



        accuracy = sum(rocauc_list)/len(rocauc_list)
    return accuracy

def train_trans(data, model, criterion, optimizer, metric='acc'):
    model.train()
    total_loss = 0
    accuracy = 0

    # zero grad
    optimizer.zero_grad()

    # output, loss, accuracy
    mask = data.train_mask
    logits = model(data)

    train_loss = criterion(logits[mask], data.y[mask])
    
    total_loss += train_loss.item()
    # update w
    train_loss.backward()
    optimizer.step()


    accuracy = get_perf(metric, logits, data, mask)
    
    return train_loss.item(), accuracy, 0
    # return train_loss.item(), accuracy, madgap[mask].mean()

def infer_trans(data, model, criterion, metric='acc'):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    preds = logits.max(1)[1]


    mask = data.val_mask.bool()
    val_loss = criterion(logits[mask], data.y[mask]).item()

    val_acc = get_perf(metric, logits, data, mask)

    mask = data.test_mask.bool()
    test_loss = criterion(logits[mask], data.y[mask]).item()
    test_acc = get_perf(metric, logits, data, mask)
    return val_loss, val_acc, 0, test_loss, test_acc, 0


if __name__ == '__main__':
    main()


