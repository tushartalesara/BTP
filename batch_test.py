from load_data import *
import multiprocessing
import heapq

cores = multiprocessing.cpu_count() // 2

data_generator = Data()
batch_size = 1024 #for gowalla
# batch_size = 2048 #for amazon-book


def ranklist_by_sorted(user_pos_test, test_items, rating):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max_item_score = heapq.nlargest(20, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r

def get_performance(user_pos_test, r):
    recall, ndcg = [], []

    # Calculating Recall
    r=np.asfarray(r)[:20]
    recall.append(np.sum(r) / len(user_pos_test))

    # Calculating NDCG
    test_r=np.zeros(20)
    length=20 if 20<len(user_pos_test) else len(user_pos_test)
    test_r[:length]=1
    idcg=np.sum(test_r * 1./np.log2(np.arange(2, 22)))
    dcg=np.sum(r*(1./np.log2(np.arange(2, 22))))
    if idcg==0.:
        idcg=1.
    ndcg_=dcg/idcg
    if np.isnan(ndcg_):
        ndcg_=0.
    ndcg.append(ndcg_)

    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(data_generator.n_items))

    test_items = list(all_items - set(training_items))
    r = ranklist_by_sorted(user_pos_test, test_items, rating)

    return get_performance(user_pos_test, r)


def test(model, users_to_test, drop_flag=False):
    result = {'recall': np.zeros(1), 'ndcg': np.zeros(1)}

    pool = multiprocessing.Pool(cores)

    u_batch_size = batch_size * 2

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        item_batch = range(data_generator.n_items)

        u_g_embeddings, pos_i_g_embeddings, _,__,___,____ = model(user_batch,
                                                        item_batch,
                                                        [],
                                                        drop_flag)
        rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users

    assert count == n_test_users
    pool.close()
    return result
