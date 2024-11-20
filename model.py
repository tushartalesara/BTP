import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


class LightGCN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, params):
        super(LightGCN, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = params['device']
        self.emb_size = params['embed_size']
        self.batch_size = params['batch_size']
        self.node_dropout = params['node_dropout']

        self.norm_adj = norm_adj
        # self.item_popularity_inv= self.compute_item_popularity().to(self.device)
        # self.user_popularity_inv= self.compute_user_popularity().to(self.device)
        self.item_popularity_inv, self.user_popularity_inv = self.compute_item_popularity(), self.compute_user_popularity()
        self.layers = eval(params['layer_size'])
        self.decay = eval(params['regs'])

        self.embedding_dict = self.init_weight()

        self.sparse_norm_adj = self.convert_sparse_matrix_to_sparse_tensor(self.norm_adj).to(self.device)
        self.popularity_weight = nn.Parameter(torch.tensor(1.0, device=self.device))


    def init_weight(self):
        # initializer = nn.init.xavier_uniform_
        initializer= nn.init.normal_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size),std=0.1)),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size),std=0.1))
        })

        # weight for the embedding enhancement layer


        return embedding_dict

    def compute_item_popularity(self):
        item_popularity = torch.tensor(self.norm_adj[:self.n_user, self.n_user:].sum(axis=0)).float().view(-1)
        
        item_popularity_inv = 1 / item_popularity
        item_popularity_inv[torch.isinf(item_popularity_inv)] = 0
        # max_val = torch.max(item_popularity_inv)
        # item_popularity_inv=max_val*item_popularity_inv
        return item_popularity_inv.to(self.device)

    def compute_user_popularity(self):
        user_popularity = torch.tensor(self.norm_adj[:self.n_user, self.n_user:].sum(axis=1)).float().view(-1)
        
        user_popularity_inv = 1 / user_popularity
        user_popularity_inv[torch.isinf(user_popularity_inv)] = 0
        # max_val = torch.max(user_popularity_inv)
        # user_popularity_inv=max_val*user_popularity_inv
        return user_popularity_inv.to(self.device)

    def convert_sparse_matrix_to_sparse_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = rate + torch.rand(x._values().size()).to(x.device)
        dropout_mask = random_tensor.floor().bool()
        i = x._indices()[:, dropout_mask]
        v = x._values()[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.size()).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items, user_emb_0, pos_emb_0, neg_emb_0):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = torch.mean(
            torch.nn.functional.softplus(neg_scores - pos_scores))

        regularizer = (1/2)*(user_emb_0.norm(2).pow(2) +
                             pos_emb_0.norm(2).pow(2) +
                             neg_emb_0.norm(2).pow(2))/float(len(users))
        emb_loss = self.decay * regularizer

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        embeddings = torch.cat(
            [self.embedding_dict['user_emb'], self.embedding_dict['item_emb']])

        all_embeddings = [embeddings]

        for k,layers in enumerate(self.layers):
            embeddings = torch.sparse.mm(A_hat, embeddings)
            if k == 0:
                embeddings[:self.n_user, :] = embeddings[:self.n_user, :] * self.user_popularity_inv.view(-1, 1)
                embeddings[self.n_user:, :] = embeddings[self.n_user:, :] * self.item_popularity_inv.view(-1, 1)
            all_embeddings.append(self.popularity_weight*embeddings)

        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.mean(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, self.embedding_dict['user_emb'][users], self.embedding_dict['item_emb'][pos_items], self.embedding_dict['item_emb'][neg_items]