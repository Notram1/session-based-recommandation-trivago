from config import *
from data import *
from utils import *
from constant import *
from nn import *
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import os
import logging

def get_prediction(loader, net):
    net.eval()
    all_scores = []
    losses = []
    for batch_id, data in enumerate(loader):
        with torch.no_grad():
            item_ids = Variable(data[0]).to(device=device_type)
            targets = Variable(data[1]).to(device=device_type)
            past_interactions = Variable(data[2]).to(device=device_type)

            past_interaction_masks = (data[3])

            price_rank = Variable(data[4]).to(device=device_type)
            city = Variable(data[5]).to(device=device_type)
            last_item =  Variable(data[6]).to(device=device_type)
            impression_index = Variable(data[7]).to(device=device_type)
            continuous_features = Variable(data[8]).to(device=device_type)

            star = Variable(data[9]).to(device=device_type)
            
            past_interactions_sess = Variable(data[10]).to(device=device_type)
            past_actions_sess = Variable(data[11]).to(device=device_type)
            
            last_click_item = Variable(data[12]).to(device=device_type)
            last_click_impression = Variable(data[13]).to(device=device_type)
            last_interact_index = Variable(data[14]).to(device=device_type)
            neighbor_prices = Variable(data[15]).to(device=device_type)
            other_item_ids = Variable(data[16]).to(device=device_type)
            city_platform = Variable(data[17]).to(device=device_type)

            prediction = net(item_ids, past_interactions, past_interaction_masks, price_rank, city, 
                        last_item, impression_index, continuous_features, star, past_interactions_sess, 
                        past_actions_sess, last_click_item, last_click_impression, last_interact_index, 
                        neighbor_prices, other_item_ids, city_platform)
            loss = crit(prediction,targets).item()
            prediction = prediction.detach().cpu().numpy().tolist()
            all_scores += prediction
            losses.append(loss)
    loss = np.mean(losses)
    return all_scores, loss

def evaluate_valid(val_loader, val_df, net, model_name):                
    val_df['score'], val_loss = get_prediction(val_loader, net)
    
    grouped_val = val_df.groupby('session_id')
    rss = [] # residual sum of squares?
    rss_group = {i:[] for i in range(1,26)}
    incorrect_session = {}
    for session_id, group in grouped_val:        
        scores = group['score']
        sorted_arg = np.flip(np.argsort(scores))

        if group['label'].values[sorted_arg][0] != 1:
            incorrect_session[session_id] = (sorted_arg.values, group['label'].values[sorted_arg])

        rss.append( group['label'].values[sorted_arg])
        rss_group[len(group)].append(group['label'].values[sorted_arg])

    mrr = compute_mean_reciprocal_rank(rss)
    mrr_group = {i:(len(rss_group[i]), compute_mean_reciprocal_rank(rss_group[i])) for i in range(1,26)}
    # print(mrr_group)
    pickle.dump( incorrect_session, open(f'../output/{model_name}_val_incorrect_order.p','wb'))

    return mrr, mrr_group, val_loss

if __name__ =='__main__':
    model_name = 'nn_xnn_time_diff_v2'
    preproc_data_name = 'preproc_data'

    torch.backends.cudnn.deterministic = True
    seed_everything(42)

    logging.basicConfig(filename= f'../output/{model_name}_{time.strftime("%Y%m%d-%H%M%S")}.log',
                        filemode='a',
                        # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger('__main__')

    config = NNConfiguration()
    config.logger = logger
    logger.info(config.get_attributes())

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    if config.sub_sample is not None:
        model_name += '_{:.3f}'.format(config.sub_sample)
        preproc_data_name += '_{:.3f}'.format(config.sub_sample)
    else:
        model_name += '_all'
        preproc_data_name += '_all'

    if config.use_test:
        model_name += '_ut'
        preproc_data_name += '_ut'

    if config.debug:
        model_name += '_db'
        preproc_data_name += '_db'

    model_name += f'_{config.device_id}'
    weight_path = f"../weights/{model_name}.model"

    # Create or load dataset
    if config.load_preproc_data:
        with open(f'{input_dir}/{preproc_data_name}.p', 'rb') as f:
            data_gen = pickle.load(f)
        config.append_diff(data_gen.config)
        logger.info('Preprocessed data has been loaded!')
    else:
        data_gen = NNDataGenerator(config)
        with open(f'{input_dir}/{preproc_data_name}.p', 'wb') as f:
            pickle.dump(data_gen, f, protocol=4)
        logger.info('Preprocessed data has been saved!')
    logger.info(config.get_attributes())
    valid_data = data_gen.val_data
    train_data= data_gen.train_data

    device_type = torch.device('cuda') if (torch.cuda.is_available() and config.use_cuda==True) else torch.device('cpu')
    net = Net(config).to(device=device_type)
    optim = use_optimizer(net, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',min_lr=0.0005, factor=0.7, verbose=True)
    logger.info(net)
    
    crit = config.loss()
    best_mrr = 0
    early_stopping = config.early_stopping
    not_improve_round = 0
    val_loader = data_gen.evaluate_data_valid()
    test_loader =data_gen.instance_a_test_loader()
    train_loader = data_gen.instance_a_train_loader()
    n_iter = 0
    stopped = False
    for i in range(config.num_epochs):        
        net.train()
        for batch_id, data in enumerate(tqdm(train_loader)):
            optim.zero_grad()
            n_iter += 1

            item_ids = Variable(data[0]).to(device=device_type)
            targets = Variable(data[1]).to(device=device_type)
            past_interactions = Variable(data[2]).to(device=device_type)
            
            past_interaction_masks = (data[3])
            
            price_rank = Variable(data[4]).to(device=device_type)
            city = Variable(data[5]).to(device=device_type)
            last_item = Variable(data[6]).to(device=device_type)
            impression_index = Variable(data[7]).to(device=device_type)
            continuous_features = Variable(data[8]).to(device=device_type)
            star = Variable(data[9]).to(device=device_type)
            
            past_interactions_sess = Variable(data[10]).to(device=device_type)
            past_actions_sess = Variable(data[11]).to(device=device_type)
            
            last_click_item = Variable(data[12]).to(device=device_type)
            last_click_impression = Variable(data[13]).to(device=device_type)
            last_interact_index = Variable(data[14]).to(device=device_type)
            neighbor_prices = Variable(data[15]).to(device=device_type)
            other_item_ids = Variable(data[16]).to(device=device_type)
            city_platform = Variable(data[17]).to(device=device_type)

            prediction = net(item_ids, past_interactions, past_interaction_masks, price_rank, city, 
                        last_item, impression_index, continuous_features, star, past_interactions_sess, 
                        past_actions_sess, last_click_item, last_click_impression, last_interact_index, 
                        neighbor_prices, other_item_ids, city_platform)
            
            loss = crit(prediction,targets)
            loss.backward()
            optim.step()    
        mrr, mrr_group, val_loss = evaluate_valid(val_loader, valid_data, net, model_name)
        scheduler.step(val_loss)

        if mrr > best_mrr:
            logger.info(f"improve from {best_mrr} to {mrr}")
            best_mrr = mrr
            not_improve_round = 0
            torch.save(net.state_dict(), weight_path)
        else:
            logger.info(f"didn't improve from {best_mrr} to {mrr}")
            not_improve_round += 1
        if not_improve_round >= early_stopping:
            break
    net.load_state_dict(torch.load(weight_path))    
    logger.info(f"BEST mrr: {best_mrr}")

    if config.debug:
        exit(0)
            
    test_df = data_gen.test_data
    test_df['score'], _ = get_prediction(test_loader, net)

    with open(f'../output/{model_name}_test_score.p', 'wb') as f:
        pickle.dump( test_df.loc[:,['score', 'session_id', 'step']],f, protocol=4)
        
    grouped_test = test_df.groupby('session_id')
    predictions = []
    session_ids = []
    for session_id, group in grouped_test:        
        scores = group['score']
        sorted_arg = np.flip(np.argsort(scores))
        sorted_item_ids = group['item_id'].values[sorted_arg]
        sorted_item_ids = data_gen.cat_encoders['item_id'].reverse_transform(sorted_item_ids)
        sorted_item_string = ' '.join([str(i) for i in sorted_item_ids])
        predictions.append(sorted_item_string)
        session_ids.append(session_id)

    prediction_df = pd.DataFrame()
    prediction_df['session_id'] = session_ids
    prediction_df['item_recommendations'] = predictions

    logger.debug(f"pred df shape: {prediction_df.shape}")
    sub_df = pd.read_csv('../input/submission_popular.csv')
    sub_df.drop('item_recommendations', axis=1, inplace=True)
    sub_df = sub_df.merge(prediction_df, on="session_id")
    # sub_df['item_recommendations'] = predictions

    sub_df.to_csv(f'../output/{model_name}.csv', index=None)