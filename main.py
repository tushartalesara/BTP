import torch
import torch.optim as optim
import os
import numpy as np
from model import LightGCN
from batch_test import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def early_stopping(log_value, best_value, stopping_step, flag_step=100):
    # early stopping strategy:

    if log_value >= best_value:
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

if __name__ == '__main__':

    params={}
    params['device']=torch.device("cuda")
    params['node_dropout'] = 0.1
    params['pretrain']=False
    params['epoch']=500
    params['node_dropout_flag']=False
    params['batch_size']=1024 # for gowalla
    # params['batch_size']=2048 # for amazon-book
    params['save_flag']=1
    params['embed_size']=128
    params['layer_size']='[128,128,128]'
    params['regs']='1e-4'
    params['seed']=2020

    set_seed(params['seed'])
    plain_adj, norm_adj = data_generator.create_adj_mat()
    
    model = LightGCN(data_generator.n_users,data_generator.n_items,norm_adj,params).to(params['device'])
    
    # weights_file = './model-gowalla/119.pkl'
    # if os.path.exists(weights_file):
    #     model.load_state_dict(torch.load(weights_file))
    if params['pretrain']:
        weights_file = './trained_models/model-gowalla/119.pkl'
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))

            users_to_test = list(data_generator.test_set.keys())
            ret = test(model, users_to_test, drop_flag=False)
            cur_best_pre_0 = ret['recall'][0]

            pretrained_results = 'recall=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % (ret['recall'][0], ret['recall'][-1], ret['ndcg'][0], ret['ndcg'][-1])
            print(f'Pretrained model (Epoch 499):', pretrained_results)
            exit(1)
        else:
            print('Specified weights file not found')
            exit(1)

    ################################ Train ##################################33#
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger,test_epochs = [], [], [], [], [],[]
    for epoch in range(params['epoch']):
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // params['batch_size'] + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings,u_emb_0,p_mb_0,n_emb_0 = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=params['node_dropout_flag'])

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings,u_emb_0,p_mb_0,n_emb_0)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            print('Epoch %d: training loss==[%.5f]' % (epoch, loss))
            continue

        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)


        loss_loger.append(loss.cpu().detach().numpy())
        rec_loger.append(ret['recall'])
        ndcg_loger.append(ret['ndcg'])
        test_epochs.append(epoch)
        print('Epoch %d: training loss==[%.5f], recall=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % (epoch, loss, ret['recall'][0], ret['recall'][-1], ret['ndcg'][0], ret['ndcg'][-1]))

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, flag_step=5)

        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and params['save_flag'] == 1:
            if os.path.isdir("./model-gowalla"):
                torch.save(model.state_dict(), './model-gowalla/' + str(epoch) + '.pkl')
                print('save the weights in path: ', './model-gowalla/' + str(epoch) + '.pkl')
            else:
                os.mkdir("./model-gowalla")
                torch.save(model.state_dict(), './model-gowalla/' + str(epoch) + '.pkl')
                print('save the weights in path: ', './model-gowalla/' + str(epoch) + '.pkl')
    
    # loss_loger.to(params['device'])
    loss=np.array(loss_loger)
    recs = np.array(rec_loger)
    ndcgs = np.array(ndcg_loger)
    test_epochs = np.array(test_epochs)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    print("\trecall=[%s], ndcg=[%s]" % ( '\t'.join(['%.5f' % r for r in recs[idx]]),'\t'.join(['%.5f' % r for r in ndcgs[idx]])))


    # Plotting Loss vs. Epoch
    plt.figure()
    plt.plot(test_epochs, loss_loger, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.show()

    # Plotting Recall vs. Epoch
    plt.figure()
    plt.plot(test_epochs, rec_loger, label='Recall@20', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall vs. Epoch')
    plt.legend()
    plt.show()

    # Plotting NDCG vs. Epoch
    plt.figure()
    plt.plot(test_epochs, ndcg_loger, label='NDCG@20', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG')
    plt.title('NDCG vs. Epoch')
    plt.legend()
    plt.show()