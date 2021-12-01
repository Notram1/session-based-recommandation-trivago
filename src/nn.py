import torch
from torch import nn
from torch.nn.functional import dropout
from utils import *
# from data import *


class GruNet1(torch.nn.Module):
    def __init__(self, config):
        super(GruNet1, self).__init__()
        self.config = config
        self.all_cat_columns = self.config.all_cat_columns
        self.categorical_emb_dim = config.categorical_emb_dim
        self.hidden_dims = config.hidden_dims
        self.num_embeddings = config.num_embeddings

        # embedding part
        self.emb_dict = torch.nn.ModuleDict()
        for cat_col in self.config.all_cat_columns:
            if cat_col =='item_id':
                
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                            embedding_dim=self.categorical_emb_dim, padding_idx = self.config.transformed_dummy_item)
            else:    
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                            embedding_dim=self.categorical_emb_dim)
        # gru for extracting session and user interest
        self.gru_sess = torch.nn.GRU(input_size = self.categorical_emb_dim *2, hidden_size = self.categorical_emb_dim//2, bidirectional=True , num_layers=2, batch_first=True)
        self.other_item_gru = torch.nn.GRU(input_size = self.categorical_emb_dim, hidden_size = self.categorical_emb_dim//2, bidirectional=True , num_layers=1, batch_first=True)

        # hidden layerrs
        self.hidden1 = torch.nn.Linear(self.categorical_emb_dim*17 , self.hidden_dims[0])
        self.hidden2 = torch.nn.Linear(self.hidden_dims[0] + config.continuous_size*2 + 3 + config.neighbor_size, self.hidden_dims[1] )
            
        # output layer
        self.output = torch.nn.Linear(self.hidden_dims[1]   , 1)
        
        # batch normalization
        self.bn = torch.nn.BatchNorm1d(self.categorical_emb_dim*17)
        self.bn_hidden = torch.nn.BatchNorm1d(self.hidden_dims[0] + config.continuous_size*2+ 3 + config.neighbor_size )
        
    def forward(self, item_id, past_interactions, mask, price_rank, city, last_item, impression_index, cont_features, 
            star, past_interactions_sess, past_actions_sess, last_click_item, last_click_impression, last_interact_index, 
            neighbor_prices, other_item_ids, city_platform):
        
        # embedding of all categorical features
        emb_item = self.emb_dict['item_id'](item_id)
        emb_past_interactions = self.emb_dict['item_id'](past_interactions)
        emb_price_rank = self.emb_dict['price_rank'](price_rank)
        emb_city = self.emb_dict['city'](city)
        emb_last_item = self.emb_dict['item_id'](last_item)
        emb_impression_index = self.emb_dict['impression_index'](impression_index)
        emb_star = self.emb_dict['star'](star)
        emb_past_interactions_sess = self.emb_dict['item_id'](past_interactions_sess)
        emb_past_actions_sess = self.emb_dict['action'](past_actions_sess)
        emb_last_click_item = self.emb_dict['item_id'](last_click_item)
        emb_last_click_impression = self.emb_dict['impression_index'](last_click_impression)
        emb_last_interact_index = self.emb_dict['impression_index'](last_interact_index)
        emb_city_platform = self.emb_dict['city_platform'](city_platform)
        emb_other_item_ids = self.emb_dict['item_id'](other_item_ids)
        
        # other items processed by gru
        emb_other_item_ids_gru, _ = self.other_item_gru(emb_other_item_ids)
        pooled_other_item_ids = F.max_pool1d(emb_other_item_ids_gru.permute(0,2,1), kernel_size=emb_other_item_ids_gru.size(1)).squeeze(2)

        # user's past clicked-out item
        emb_past_interactions = emb_past_interactions.permute(0,2,1)
        pooled_interaction = F.max_pool1d(emb_past_interactions, kernel_size=self.config.sequence_length).squeeze(2)
                
        # concatenate sequence of item ids and actions to model session dynamics
        emb_past_interactions_sess = torch.cat( [emb_past_interactions_sess, emb_past_actions_sess], dim=2)
        emb_past_interactions_sess , _ = self.gru_sess(emb_past_interactions_sess)
        emb_past_interactions_sess = emb_past_interactions_sess.permute(0,2,1)
        pooled_interaction_sess = F.max_pool1d(emb_past_interactions_sess, kernel_size=self.config.sess_length).squeeze(2)
                
        # categorical feature interactions
        item_interaction =  emb_item * pooled_interaction
        item_last_item = emb_item * emb_last_item
        item_last_click_item = emb_item * emb_last_click_item
        imp_last_idx = emb_impression_index * emb_last_interact_index
                
        # efficiently compute the aggregation of feature interactions 
        emb_list = [emb_item, pooled_interaction, emb_price_rank, emb_city, emb_last_item, emb_impression_index, emb_star]
        emb_concat = torch.cat(emb_list, dim=1)
        sum_squared = torch.pow( torch.sum( emb_concat, dim=1) , 2).unsqueeze(1)
        squared_sum = torch.sum( torch.pow( emb_concat, 2) , dim=1).unsqueeze(1)
        second_order = 0.5 * (sum_squared - squared_sum)
        
        # compute the square of continuous features
        squared_cont = torch.pow(cont_features, 2)

        # DNN part
        concat = torch.cat([emb_item, pooled_interaction, emb_price_rank, emb_city, emb_last_item, emb_impression_index, 
                        item_interaction, item_last_item, emb_star, pooled_interaction_sess, emb_last_click_item, 
                        emb_last_click_impression, emb_last_interact_index, item_last_click_item, imp_last_idx, 
                        pooled_other_item_ids, emb_city_platform] , dim=1)
        concat = self.bn(concat)
        hidden = torch.nn.ReLU()(self.hidden1(concat))

        hidden = torch.cat( [cont_features, hidden, sum_squared, squared_sum, second_order, squared_cont, neighbor_prices] , dim=1)        
        hidden = self.bn_hidden(hidden)
        hidden = torch.nn.ReLU()(self.hidden2(hidden))        

        output = torch.sigmoid(self.output(hidden)).squeeze()        
        return output


class GruNet2(torch.nn.Module):
    def __init__(self, config):
        super(GruNet2, self).__init__()
        self.config = config
        self.all_cat_columns = self.config.all_cat_columns
        self.categorical_emb_dim = config.categorical_emb_dim
        self.hidden_dims = config.hidden_dims
        self.num_embeddings = config.num_embeddings

        # embedding part
        self.emb_dict = torch.nn.ModuleDict()
        for cat_col in self.config.all_cat_columns:
            if cat_col =='item_id':
                
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                            embedding_dim=self.categorical_emb_dim, padding_idx = self.config.transformed_dummy_item)
            else:    
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                            embedding_dim=self.categorical_emb_dim)
        # gru for extracting session and user interest
        self.gru_sess = torch.nn.GRU(input_size = self.categorical_emb_dim *2, hidden_size = self.categorical_emb_dim//2, bidirectional=True , num_layers=2, batch_first=True)
        self.other_item_gru = torch.nn.GRU(input_size = self.categorical_emb_dim, hidden_size = self.categorical_emb_dim//2, bidirectional=True , num_layers=1, batch_first=True)
        
        # hidden layerrs
        self.hidden1 = torch.nn.Linear(self.categorical_emb_dim*13 , self.hidden_dims[0])
        self.hidden2 = torch.nn.Linear(self.hidden_dims[0] + config.continuous_size + config.neighbor_size, self.hidden_dims[1] )
            
        # output layer
        self.output = torch.nn.Linear(self.hidden_dims[1]   , 1)
        
        # batch normalization
        self.bn = torch.nn.BatchNorm1d(self.categorical_emb_dim*13)
        self.bn_hidden = torch.nn.BatchNorm1d(self.hidden_dims[0] + config.continuous_size + config.neighbor_size )
        
    def forward(self, item_id, past_interactions, mask, price_rank, city, last_item, impression_index, cont_features, 
            star, past_interactions_sess, past_actions_sess, last_click_item, last_click_impression, last_interact_index, 
            neighbor_prices, other_item_ids, city_platform):
        
        # embedding of all categorical features
        emb_item = self.emb_dict['item_id'](item_id)
        emb_past_interactions = self.emb_dict['item_id'](past_interactions)
        emb_price_rank = self.emb_dict['price_rank'](price_rank)
        emb_city = self.emb_dict['city'](city)
        emb_last_item = self.emb_dict['item_id'](last_item)
        emb_impression_index = self.emb_dict['impression_index'](impression_index)
        emb_star = self.emb_dict['star'](star)
        emb_past_interactions_sess = self.emb_dict['item_id'](past_interactions_sess)
        emb_past_actions_sess = self.emb_dict['action'](past_actions_sess)
        emb_last_click_item = self.emb_dict['item_id'](last_click_item)
        emb_last_click_impression = self.emb_dict['impression_index'](last_click_impression)
        emb_last_interact_index = self.emb_dict['impression_index'](last_interact_index)
        emb_city_platform = self.emb_dict['city_platform'](city_platform)
        emb_other_item_ids = self.emb_dict['item_id'](other_item_ids)
        
        # other items processed by gru
        emb_other_item_ids_gru, _ = self.other_item_gru(emb_other_item_ids)
        pooled_other_item_ids = F.max_pool1d(emb_other_item_ids_gru.permute(0,2,1), kernel_size=emb_other_item_ids_gru.size(1)).squeeze(2)

        # user's past clicked-out item
        emb_past_interactions = emb_past_interactions.permute(0,2,1)
        pooled_interaction = F.max_pool1d(emb_past_interactions, kernel_size=self.config.sequence_length).squeeze(2)
                
        # concatenate sequence of item ids and actions to model session dynamics
        emb_past_interactions_sess = torch.cat( [emb_past_interactions_sess, emb_past_actions_sess], dim=2)
        emb_past_interactions_sess , _ = self.gru_sess(emb_past_interactions_sess)
        emb_past_interactions_sess = emb_past_interactions_sess.permute(0,2,1)
        pooled_interaction_sess = F.max_pool1d(emb_past_interactions_sess, kernel_size=self.config.sess_length).squeeze(2)

        # DNN part
        concat = torch.cat([emb_item, pooled_interaction, emb_price_rank, emb_city, emb_last_item, emb_impression_index, 
                            emb_star, pooled_interaction_sess, emb_last_click_item, emb_last_click_impression, 
                            emb_last_interact_index, pooled_other_item_ids, emb_city_platform] , dim=1)
        concat = self.bn(concat)
        hidden = torch.nn.ReLU()(self.hidden1(concat))

        hidden = torch.cat( [cont_features, hidden, neighbor_prices] , dim=1)        
        hidden = self.bn_hidden(hidden)
        hidden = torch.nn.ReLU()(self.hidden2(hidden))        

        output = torch.sigmoid(self.output(hidden)).squeeze()        
        return output


class TransformerNet1(torch.nn.Module):
    def __init__(self, config):
        super(TransformerNet1, self).__init__()
        self.config = config
        self.all_cat_columns = self.config.all_cat_columns
        self.categorical_emb_dim = config.categorical_emb_dim
        self.hidden_dims = config.hidden_dims
        self.num_embeddings = config.num_embeddings

        # embedding part
        self.emb_dict = torch.nn.ModuleDict()
        for cat_col in self.config.all_cat_columns:
            if cat_col =='item_id':                
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                embedding_dim=self.categorical_emb_dim, padding_idx = self.config.transformed_dummy_item)
            elif cat_col == 'action':
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                embedding_dim=self.categorical_emb_dim, padding_idx = self.config.transformed_dummy_action)
            elif cat_col == 'impression_index':
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                embedding_dim=self.categorical_emb_dim, padding_idx = DUMMY_IMPRESSION_INDEX)
            else:    
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                embedding_dim=self.categorical_emb_dim)
        # positional encoding to account for sequence order information in transformer blocks
        self.pos_encoder = PositionalEncoding(d_model= self.categorical_emb_dim, dropout= self.config.dropout_rate)

        # multi-head transformer blocks for extracting session and user interest
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.categorical_emb_dim, nhead=4, dim_feedforward=self.hidden_dims[0], 
                                                    dropout=self.config.dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        
        # hidden layers
        self.hidden1 = torch.nn.Linear(self.categorical_emb_dim*17 , self.hidden_dims[0])
        self.hidden2 = torch.nn.Linear(self.hidden_dims[0] + config.continuous_size*2 + 3 + config.neighbor_size, self.hidden_dims[1] )
            
        # output layer
        self.output = torch.nn.Linear(self.hidden_dims[1], 1)
        
        # batch normalization
        self.bn = torch.nn.BatchNorm1d(self.categorical_emb_dim*17)
        self.bn_hidden = torch.nn.BatchNorm1d(self.hidden_dims[0] + config.continuous_size*2+ 3 + config.neighbor_size )
        
    def forward(self, item_id, past_interactions, mask, price_rank, city, last_item, impression_index, cont_features, 
            star, past_interactions_sess, past_actions_sess, last_click_item, last_click_impression, last_interact_index, 
            neighbor_prices, other_item_ids, city_platform):
        
        # embedding of all categorical features
        emb_item = self.emb_dict['item_id'](item_id) # contains DUMMY_ITEM?
        emb_past_interactions = self.emb_dict['item_id'](past_interactions) # contains DUMMY_ITEM
        emb_price_rank = self.emb_dict['price_rank'](price_rank)
        emb_city = self.emb_dict['city'](city)
        emb_last_item = self.emb_dict['item_id'](last_item) # contains DUMMY_ITEM
        emb_impression_index = self.emb_dict['impression_index'](impression_index)
        emb_star = self.emb_dict['star'](star) # contains 0 star
        emb_past_interactions_sess = self.emb_dict['item_id'](past_interactions_sess) # contains DUMMY_ITEM
        emb_past_actions_sess = self.emb_dict['action'](past_actions_sess) # contains DUMMY_ACTION
        emb_last_click_item = self.emb_dict['item_id'](last_click_item) # contains DUMMY_ITEM
        emb_last_click_impression = self.emb_dict['impression_index'](last_click_impression) # contains DUMMY_IMPRESSION_INDEX
        emb_last_interact_index = self.emb_dict['impression_index'](last_interact_index) # contains DUMMY_IMPRESSION_INDEX
        emb_city_platform = self.emb_dict['city_platform'](city_platform)
        emb_other_item_ids = self.emb_dict['item_id'](other_item_ids) # contains DUMMY_ITEM
        
        # sequential categorical features processed by transformer
        emb_sequential = torch.cat( [emb_other_item_ids, emb_past_interactions, emb_past_interactions_sess, 
                                emb_past_actions_sess], dim=1) * np.sqrt(self.categorical_emb_dim * 4)
        emb_sequential = self.pos_encoder(emb_sequential)
        transformed_sequential = self.transformer_encoder(emb_sequential)

        # other items
        emb_other_item_ids_transf = transformed_sequential[:, :emb_other_item_ids.size(1),:]
        pooled_other_item_ids = F.max_pool1d(emb_other_item_ids_transf.permute(0,2,1), kernel_size=emb_other_item_ids_transf.size(1)).squeeze(2)

        # user's past clicked-out item
        emb_past_interactions_transf = transformed_sequential[:, emb_other_item_ids.size(1):emb_other_item_ids.size(1)+emb_past_interactions.size(1), :]
        pooled_interaction = F.max_pool1d(emb_past_interactions_transf.permute(0,2,1), kernel_size=self.config.sequence_length).squeeze(2)
                
        # concatenate sequence of item ids and actions to model session dynamics
        emb_past_interactions_sess_transf = transformed_sequential[:, emb_other_item_ids.size(1)+emb_past_interactions.size(1):, :]
        pooled_interaction_sess = F.max_pool1d(emb_past_interactions_sess_transf.permute(0,2,1), kernel_size=2*self.config.sess_length).squeeze(2)
               
        # categorical feature interactions
        item_interaction =  emb_item * pooled_interaction
        item_last_item = emb_item * emb_last_item
        item_last_click_item = emb_item * emb_last_click_item
        imp_last_idx = emb_impression_index * emb_last_interact_index
                
        # efficiently compute the aggregation of feature interactions 
        emb_list = [emb_item, pooled_interaction, emb_price_rank, emb_city, emb_last_item, emb_impression_index, emb_star]
        emb_concat = torch.cat(emb_list, dim=1)
        sum_squared = torch.pow( torch.sum( emb_concat, dim=1) , 2).unsqueeze(1)
        squared_sum = torch.sum( torch.pow( emb_concat, 2) , dim=1).unsqueeze(1)
        second_order = 0.5 * (sum_squared - squared_sum)
        
        # compute the square of continuous features
        squared_cont = torch.pow(cont_features, 2)

        # DNN part
        concat = torch.cat([emb_item, pooled_interaction, emb_price_rank, emb_city, emb_last_item, emb_impression_index, 
                        item_interaction, item_last_item, emb_star, pooled_interaction_sess, emb_last_click_item, 
                        emb_last_click_impression, emb_last_interact_index, item_last_click_item, imp_last_idx, 
                        pooled_other_item_ids, emb_city_platform] , dim=1)
        concat = self.bn(concat)        
        hidden = torch.nn.ReLU()(self.hidden1(concat))

        hidden = torch.cat( [cont_features, hidden, sum_squared, squared_sum, second_order, squared_cont, neighbor_prices] , dim=1)        
        hidden = self.bn_hidden(hidden)
        hidden = torch.nn.ReLU()(self.hidden2(hidden))
    
        output = torch.sigmoid(self.output(hidden)).squeeze()
                
        return output
    
class TransformerNet2(torch.nn.Module):
    def __init__(self, config):
        super(TransformerNet2, self).__init__()
        self.config = config
        self.all_cat_columns = self.config.all_cat_columns
        self.categorical_emb_dim = config.categorical_emb_dim
        self.hidden_dims = config.hidden_dims
        self.num_embeddings = config.num_embeddings

        # embedding part
        self.emb_dict = torch.nn.ModuleDict()
        for cat_col in self.config.all_cat_columns:
            if cat_col =='item_id':                
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                embedding_dim=self.categorical_emb_dim, padding_idx = self.config.transformed_dummy_item)
            elif cat_col == 'action':
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                embedding_dim=self.categorical_emb_dim, padding_idx = self.config.transformed_dummy_action)
            elif cat_col == 'impression_index':
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                embedding_dim=self.categorical_emb_dim, padding_idx = DUMMY_IMPRESSION_INDEX)
            else:    
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                embedding_dim=self.categorical_emb_dim)

        # positional encoding to account for sequence order information in transformer blocks
        self.pos_encoder = PositionalEncoding(d_model= self.categorical_emb_dim, dropout= self.config.dropout_rate)

        # multi-head transformer blocks 
        self.transformer = nn.Transformer(d_model=self.categorical_emb_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                                    dim_feedforward=self.hidden_dims[0], dropout=self.config.dropout_rate, batch_first=True)
        
        # hidden layers
        self.hidden = torch.nn.Linear(self.categorical_emb_dim * 104 + config.continuous_size + config.neighbor_size, self.hidden_dims[1])
            
        # output layer
        self.output = torch.nn.Linear(self.hidden_dims[1], 1)
        
        
    def forward(self, item_id, past_interactions, mask, price_rank, city, last_item, impression_index, cont_features, 
            star, past_interactions_sess, past_actions_sess, last_click_item, last_click_impression, last_interact_index, 
            neighbor_prices, other_item_ids, city_platform):
        
        # embedding of all categorical features
        emb_item = self.emb_dict['item_id'](item_id).unsqueeze(1) # contains DUMMY_ITEM?
        emb_past_interactions = self.emb_dict['item_id'](past_interactions) # contains DUMMY_ITEM
        emb_price_rank = self.emb_dict['price_rank'](price_rank).unsqueeze(1)
        emb_city = self.emb_dict['city'](city).unsqueeze(1)
        emb_last_item = self.emb_dict['item_id'](last_item).unsqueeze(1) # contains DUMMY_ITEM
        emb_impression_index = self.emb_dict['impression_index'](impression_index).unsqueeze(1)
        emb_star = self.emb_dict['star'](star).unsqueeze(1) # contains 0 star
        emb_past_interactions_sess = self.emb_dict['item_id'](past_interactions_sess) # contains DUMMY_ITEM
        emb_past_actions_sess = self.emb_dict['action'](past_actions_sess) # contains DUMMY_ACTION
        emb_last_click_item = self.emb_dict['item_id'](last_click_item).unsqueeze(1) # contains DUMMY_ITEM
        emb_last_click_impression = self.emb_dict['impression_index'](last_click_impression).unsqueeze(1) # contains DUMMY_IMPRESSION_INDEX
        emb_last_interact_index = self.emb_dict['impression_index'](last_interact_index).unsqueeze(1) # contains DUMMY_IMPRESSION_INDEX
        emb_city_platform = self.emb_dict['city_platform'](city_platform).unsqueeze(1)
        emb_other_item_ids = self.emb_dict['item_id'](other_item_ids) # contains DUMMY_ITEM
        
        # Concatenate features processed by transformer
        src = torch.cat( [emb_item, emb_past_interactions, emb_price_rank, emb_city, emb_last_item, emb_impression_index,
                        emb_star, emb_past_interactions_sess, emb_past_actions_sess, emb_last_click_item, emb_last_click_impression,
                        emb_last_interact_index, emb_city_platform, emb_other_item_ids], dim=1) * np.sqrt(self.categorical_emb_dim)
        src = self.pos_encoder(src)
        tgt = torch.zeros_like(src)

        transformer_out = self.transformer(src, tgt)
        
        hidden = torch.cat( [transformer_out.view(transformer_out.size(0), -1), cont_features, neighbor_prices] , dim=1)
        hidden = torch.nn.ReLU()(self.hidden(hidden))
        
        output = torch.sigmoid(self.output(hidden)).squeeze()        
        return output
    

# Taken from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
