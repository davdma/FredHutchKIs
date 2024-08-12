import os
from os.path import join
import shutil
import random
from configargparse import ArgumentParser
import time
import logging
import numpy as np
from time import gmtime, strftime
import pandas as pd
import pickle
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, rand
from sklearn.metrics import roc_auc_score, matthews_corrcoef, r2_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
from lifelines.utils import concordance_index
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from prosmith.utils.modules import (
    MM_TN,
    MM_TNConfig)
from prosmith.utils.data_utils import SMILESProteinDataset
from prosmith.utils.train_utils import *
from prosmith.utils.gen_utils import save_config
from prosmith.training.training import get_training_parser
from prosmith.preprocessing.preprocessing import get_preprocessing_parser

def get_xgboost_arguments():
    parser = ArgumentParser()
    # centralize into one folder with all the files: train_file, val_file, test_file, embed_path, save_pred_path
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="The path where preprocessed data for the model is saved.",
    )
    # optional pred save path (otherwise defaults to save_path/pred/)
    parser.add_argument(
        "--save_pred_path",
        type=str,
        default="",
        help="The path where predictions are saved.",
    )
    parser.add_argument(
        "--num_iter",
        default=2000,
        type=int,
        help="Total number of iterations to search for best set of hyperparameters.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="model_chkpt.pkl",
        help="Name of trained Transformer Network file.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="train_config.toml",
        help="Name of training config file.",
    )
    parser.add_argument(
        '--port',
        default=12558, # 12557
        type=int,
        help='Port for tcp connection for multiprocessing'
    )
    parser.add_argument(
        '--log_name',
        default="xgboost",
        type=str,
        help='Will be added to the file name of the log file'
    )
    return parser

def extract_repr(args, model, dataloader, device, logger):
    print("device: %s" % device)
    # evaluate the model on validation set
    model.eval()
    logger.info(f"Extracting repr...")

    if is_cuda(device):
        model = model.to(device)
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # move batch to device
            batch = [r.to(device) for r in batch]
            smiles_emb, smiles_attn, protein_emb, protein_attn, labels, indices = batch
            _, cls_repr = model(
                smiles_emb=smiles_emb, 
                smiles_attn=smiles_attn, 
                protein_emb=protein_emb,
                protein_attn=protein_attn,
                device=device,
                gpu=0,
                get_repr=True
            )

            protein_attn = int(sum(protein_attn.cpu().detach().numpy()[0]))
            smiles_attn = int(sum(smiles_attn.cpu().detach().numpy()[0]))

            smiles = smiles_emb[0][:smiles_attn].mean(0).cpu().detach().numpy()
            esm = protein_emb[0][:protein_attn].mean(0).cpu().detach().numpy()
            cls_rep = cls_repr[0].cpu().detach().numpy()

            if step == 0:
                cls_repr_all = cls_rep.reshape(1,-1)
                esm_repr_all = esm.reshape(1,-1)
                smiles_repr_all = smiles.reshape(1,-1)
                labels_all = labels[0]
                logging.info(indices.cpu().detach().numpy())
                original_indices = list(indices.cpu().detach().numpy())
            else:
                cls_repr_all = np.concatenate((cls_repr_all, cls_rep.reshape(1,-1)), axis=0)
                smiles_repr_all = np.concatenate((smiles_repr_all, smiles.reshape(1,-1)), axis=0)
                esm_repr_all = np.concatenate((esm_repr_all, esm.reshape(1,-1)), axis=0)
                labels_all = torch.cat((labels_all, labels[0]), dim=0)
                original_indices = original_indices + list(indices.cpu().detach().numpy())

            if step % 10 == 0:
                logger.info('Step [{}/{}]'.format(step+1, len(trainloader)))

    logger.info(f"Repr extraction complete.")
    return cls_repr_all, esm_repr_all, smiles_repr_all, labels_all.cpu().detach().numpy(), original_indices

depth_array = [6,7,8,9,10,11,12,13,14]
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", depth_array),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 30, 1000),
    "weight": hp.uniform("weight", 0.01,0.99)}

def trainer(gpu, args, device):
    log_file = args.save_path / ("log_" + args.log_name + '_layers' + str(args.num_hidden_layers) + '_xgboost.log')
    logger = logging.getLogger('process_log')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    fhandler = logging.FileHandler(filename=str(log_file), mode='a')
    logger.addHandler(fhandler)
    logger.info(args)

    if is_cuda(device):
        setup(gpu, args.world_size, str(args.port))
        torch.manual_seed(0)
        torch.cuda.set_device(gpu)
    
    config = MM_TNConfig.from_dict({
        "s_hidden_size":600,
        "p_hidden_size": args.embedding_size,
        "hidden_size": 768,
        "max_seq_len":1276,
        "num_hidden_layers" : args.num_hidden_layers,
        "task" : args.task,
        "num_classes": args.multi_num_classes if args.task == 'multiclass' else 1
    })

    logger.info(f"Loading dataset to {device}:{gpu}")
    train_dataset = SMILESProteinDataset(
        data_path=args.save_path / 'train.csv',
        embed_dir=args.save_path / 'embeddings',
        train=True,
        device=device, 
        gpu=gpu,
        random_state=0,
        task=args.task,
        extraction_mode=True) 

    val_dataset = SMILESProteinDataset(
        data_path=args.save_path / 'val.csv',
        embed_dir=args.save_path / 'embeddings',
        train=False, 
        device=device, 
        gpu=gpu,
        random_state=0,
        task=args.task,
        extraction_mode=True)
        
    test_dataset = SMILESProteinDataset(
        data_path=args.save_path / 'test.csv',
        embed_dir=args.save_path / 'embeddings',
        train=False, 
        device=device, 
        gpu=gpu,
        random_state=0,
        task=args.task,
        extraction_mode=True)

    trainsampler = DistributedSampler(train_dataset, 
                                      shuffle=False, 
                                      num_replicas=args.world_size, 
                                      rank=gpu, 
                                      drop_last=True)
    valsampler = DistributedSampler(val_dataset, 
                                    shuffle=False, 
                                    num_replicas=args.world_size, 
                                    rank=gpu, 
                                    drop_last=True)
    testsampler = DistributedSampler(test_dataset, 
                                     shuffle=False, 
                                     num_replicas=args.world_size, 
                                     rank=gpu, 
                                     drop_last=True)

    logger.info(f"Loading dataloader")
    trainloader = DataLoader(train_dataset, 
                             batch_size=1, 
                             shuffle=False, 
                             num_workers=1, 
                             sampler=trainsampler)
    valloader = DataLoader(val_dataset, 
                           batch_size=1, 
                           shuffle=False, 
                           num_workers=1, 
                           sampler=valsampler)
    testloader = DataLoader(test_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=1, 
                            sampler=testsampler)

    logger.info(f"Loading model")
    model = MM_TN(config)
    
    if is_cuda(device):
        model = model.to(gpu)
        model = DDP(model, device_ids=[gpu])

    if (args.save_path / ('model/' + args.pretrained_model)).exists():
        pretrained_path = args.save_path / ('model/' + args.pretrained_model)
        logger.info(f"Loading model")
        try:
            state_dict = torch.load(pretrained_path)
            new_model_state_dict = model.state_dict()
            for key in new_model_state_dict.keys():
                if key in state_dict.keys():
                    try:
                        new_model_state_dict[key].copy_(state_dict[key])
                        #logging.info("Updatete key: %s" % key)
                    except:
                        None
            model.load_state_dict(new_model_state_dict)
            logger.info("Successfully loaded pretrained model")
        except:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict[key.replace("module.", "")] = value
            model.load_state_dict(new_state_dict)
            logger.info("Successfully loaded pretrained model (V2)")

    else:
        logger.info("Model path is invalid, cannot load pretrained Interbert model")
        raise Exception('Invalid model path.')

    logger.info('Extracting representations...')
    val_cls, val_esm, val_smiles, val_labels, _ = extract_repr(args, model, valloader, device, logger)
    test_cls, test_esm, test_smiles, test_labels, test_indices = extract_repr(args, model, testloader, device, logger)
    train_cls, train_esm, train_smiles, train_labels, _ = extract_repr(args, model, trainloader, device, logger)

    logger.info('Number of test labels: ' + str(len(test_labels)))
    logger.info(f"Extraction complete")
    
    def get_predictions(param, dM_train, dM_val):
        param, num_round, dM_train = set_param_values_V2(param=param, dtrain=dM_train)
        bst = xgb.train(param,  dM_train, num_round)
        y_val_pred = bst.predict(dM_val)
        return(y_val_pred)
        
    def get_performance_metrics(pred, true):
        if args.task == 'binary':
            acc = np.mean(np.round(pred) == np.array(true))
            roc_auc = roc_auc_score(np.array(true), pred)
            mcc = matthews_corrcoef(np.array(true), np.round(pred))
            # add precision, recall, f1
            f1 = f1_score(np.array(true), np.round(pred))
            recall = recall_score(np.array(true), np.round(pred))
            precision = precision_score(np.array(true), np.round(pred))
            logging.info(f"accuracy: {acc}, ROC AUC: {roc_auc}, MCC: {mcc}")
            logging.info(f"precision: {precision}, recall: {recall}, f1: {f1}")
        elif args.task == 'regression':
            mse = mean_squared_error(true, pred)
            CI = concordance_index(true, pred)
            rm2 = get_rm2(ys_orig = true, ys_line = pred)
            R2 = r2_score(true, pred)
            mae = mean_absolute_error(true, pred)
            rmse = np.sqrt(mse)
            logging.info("MSE: %s,R2: %s, rm2: %s, CI: %s, MAE: %s, RMSE: %s" % (mse, R2, rm2, CI, mae, rmse))
        else:
            # multiclass
            # log acc, precision, recall, f1 for each class
            labels = [i for i in range(args.multi_num_classes)]
            acc = multi_accuracy(pred, np.array(true), labels=labels)
            precision = precision_score(np.array(true), pred, labels=labels, average=None)
            recall = recall_score(np.array(true), pred, labels=labels, average=None)
            f1 = f1_score(np.array(true), pred, labels=labels, average=None)
            for i in range(args.multi_num_classes):
                logging.info(f'Performance metrics for class {i}:')
                logging.info(f'ACC: {acc[i]}, PRECISION: {precision[i]}, RECALL: {recall[i]}, F1: {f1[i]}')
    
    def set_param_values(param):
        num_round = int(param["num_rounds"])
        param["tree_method"] = "hist"
        param["device"] = f"cuda:{dist.get_rank()}"
        param["sampling_method"] = "gradient_based"
        if args.task == 'regression':
            param['objective'] = 'reg:squarederror'
            weights = None
        elif args.task == 'binary':
            param['objective'] = 'binary:logistic'
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
            dtrain.set_weight(weights)
        else:
            param['objective'] = 'multi:softmax'
            param['num_class'] = 3
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
            dtrain.set_weight(weights)

        del param["num_rounds"]
        del param["weight"]
        return(param, num_round)

    def set_param_values_all_cls(param):
        num_round = int(param["num_rounds"])
        
        param["tree_method"] = "gpu_hist"
        param["sampling_method"] = "gradient_based"
        if args.task == 'regression':
            param['objective'] = 'reg:squarederror'
            weights = None
        elif args.task == 'binary':
            param['objective'] = 'binary:logistic'
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain_all_cls.get_label()])
            dtrain_all_cls.set_weight(weights)
        else:
            # multiclass
            param['objective'] = 'multi:softmax'
            param['num_class'] = 3
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
            dtrain_all_cls.set_weight(weights)
            
        del param["num_rounds"]
        del param["weight"]
        return(param, num_round)

    def set_param_values_cls(param):
        num_round = int(param["num_rounds"])
        
        param["tree_method"] = "gpu_hist"
        param["sampling_method"] = "gradient_based"
        if args.task == 'regression':
            param['objective'] = 'reg:squarederror'
            weights = None
        elif args.task == 'binary':
            param['objective'] = 'binary:logistic'
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain_cls.get_label()])
            dtrain_cls.set_weight(weights)
        else:
            param['objective'] = 'multi:softmax'
            param['num_class'] = 3
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
            dtrain_cls.set_weight(weights)
            
        del param["num_rounds"]
        del param["weight"]
        return(param, num_round)

    def set_param_values_V2(param, dtrain):
        num_round = int(param["num_rounds"])
        param["max_depth"] = int(depth_array[param["max_depth"]])
        param["tree_method"] = "gpu_hist"
        param["sampling_method"] = "gradient_based"
        if args.task == 'regression':
            param['objective'] = 'reg:squarederror'
            weights = None
        elif args.task == 'binary':
            param['objective'] = 'binary:logistic'
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
            dtrain.set_weight(weights)
        else:
            param['objective'] = 'multi:softmax'
            param['num_class'] = 3
            weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
            dtrain.set_weight(weights)
            
        del param["num_rounds"]
        del param["weight"]
        return(param, num_round, dtrain)
        
    def get_performance(pred, true):
        # metric used for hyperoptimization
        if args.task == 'binary':
            MCC = matthews_corrcoef(true, np.round(pred))
            return (-MCC)
        elif args.task == 'regression':
            MSE = mean_squared_error(true, pred)
            return (MSE)
        else:
            # multiclass - assume no need to do rounding
            weighted_F1 = f1_score(true, pred, 
                                   labels=[i for i in range(args.multi_num_classes)],
                                   average='weighted')
            return (-weighted_f1)

    ############# ESM+ChemBERTa2
    train_X_all = np.concatenate([train_esm, train_smiles], axis=1)
    test_X_all = np.concatenate([test_esm, test_smiles], axis=1)
    val_X_all = np.concatenate([val_esm, val_smiles], axis=1)

    dtrain = xgb.DMatrix(np.array(train_X_all), label=np.array(train_labels).astype(float))
    dtest = xgb.DMatrix(np.array(test_X_all), label=np.array(test_labels).astype(float))
    dvalid = xgb.DMatrix(np.array(val_X_all), label=np.array(val_labels).astype(float))
    dtrain_val = xgb.DMatrix(np.concatenate([np.array(train_X_all), np.array(val_X_all)], axis=0),
        label=np.concatenate([np.array(train_labels).astype(float),np.array(val_labels).astype(float)], axis=0))
    
    def train_xgboost_model_all(param):
        param, num_round = set_param_values(param)
        #Training:
        bst = xgb.train(param, dtrain, num_round)
        return (get_performance(pred=bst.predict(dvalid), true=val_labels))

    trials = Trials()
    best = fmin(fn=train_xgboost_model_all, space=space_gradient_boosting,
                algo=rand.suggest, max_evals=args.num_iter, trials=trials)

    #predictions for validation and test set on test set:
    logger.info("ESM+ChemBERTa2")
    logger.info("Validation set:")
    y_val_pred_all = get_predictions(param=trials.argmin, dM_train=dtrain, dM_val=dvalid)
    get_performance_metrics(pred=y_val_pred_all, true=val_labels)
    logger.info("Test set:")
    y_test_pred_all = get_predictions(param=trials.argmin, dM_train=dtrain_val, dM_val=dtest)
    get_performance_metrics(pred=y_test_pred_all, true=test_labels)
    
    ############# ESM+ChemBERTa +cls
    train_X_all_cls = np.concatenate([np.concatenate([train_esm, train_smiles], axis = 1), train_cls], axis=1)
    test_X_all_cls = np.concatenate([np.concatenate([test_esm, test_smiles], axis = 1), test_cls], axis=1)
    val_X_all_cls = np.concatenate([np.concatenate([val_esm, val_smiles], axis = 1), val_cls], axis=1)

    dtrain_all_cls = xgb.DMatrix(np.array(train_X_all_cls), label = np.array(train_labels).astype(float))
    dtest_all_cls = xgb.DMatrix(np.array(test_X_all_cls), label = np.array(test_labels).astype(float))
    dvalid_all_cls = xgb.DMatrix(np.array(val_X_all_cls), label = np.array(val_labels).astype(float))
    dtrain_val_all_cls = xgb.DMatrix(np.concatenate([np.array(train_X_all_cls), np.array(val_X_all_cls)], axis=0), 
            label=np.concatenate([np.array(train_labels).astype(float),np.array(val_labels).astype(float)], axis = 0))
    
    def train_xgboost_model_all_cls(param):
        param, num_round = set_param_values_all_cls(param)
        #Training:
        bst = xgb.train(param,  dtrain_all_cls, num_round)
        return(get_performance(pred=bst.predict(dvalid_all_cls), true=val_labels))

    trials = Trials()
    best = fmin(fn = train_xgboost_model_all, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = args.num_iter, trials = trials)

    # predictions for validation and test set on test set
    logging.info("ESM+ChemBERTa2+cls-token")
    logging.info("Validation set:")
    y_val_pred_all_cls = get_predictions(param = trials.argmin, dM_train = dtrain_all_cls, dM_val = dvalid_all_cls)
    get_performance_metrics(pred=y_val_pred_all_cls, true=val_labels)
    
    logging.info("Test set:")
    y_test_pred_all_cls = get_predictions(param  = trials.argmin, dM_train = dtrain_val_all_cls, dM_val = dtest_all_cls)
    get_performance_metrics(pred = y_test_pred_all_cls, true = test_labels)

    ############# cls token
    dtrain_cls = xgb.DMatrix(np.array(train_cls), label = np.array(train_labels).astype(float))
    dvalid_cls = xgb.DMatrix(np.array(val_cls), label = np.array(val_labels).astype(float))
    dtest_cls = xgb.DMatrix(np.array(test_cls), label = np.array(test_labels).astype(float))
    dtrain_val_cls = xgb.DMatrix(np.concatenate([np.array(train_cls), np.array(val_cls)], axis = 0),
                                label = np.concatenate([np.array(train_labels).astype(float),np.array(val_labels).astype(float)], axis = 0))
    
    def train_xgboost_model_cls(param):
        param, num_round = set_param_values_cls(param)
        #Training:
        bst = xgb.train(param,  dtrain_cls, num_round)
        return(get_performance(pred = bst.predict(dvalid_cls), true =val_labels))

    trials = Trials()
    best = fmin(fn = train_xgboost_model_cls, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = args.num_iter, trials = trials)
                
    #predictions for validation and test set on test set:
    logger.info("cls-token")
    logger.info("Validation set:")
    y_val_pred_cls = get_predictions(param = trials.argmin, dM_train = dtrain_cls, dM_val = dvalid_cls)
    get_performance_metrics(pred = y_val_pred_cls, true = val_labels)
    
    logger.info("Test set:")
    y_test_pred_cls = get_predictions(param = trials.argmin, dM_train = dtrain_val_cls, dM_val = dtest_cls)
    get_performance_metrics(pred = y_test_pred_cls, true = test_labels)
    
    #############
    best_weighted_f1, best_mcc, best_mse = 0, 0, 1000
    best_i, best_j, best_k = 0,0,0
    for i in [k/100 for k in range(0,100)]:
        for j in [k/100 for k in range(0,100)]:
            if i+j <=1:
                k = (1-i-j)
                y_val_pred = i*y_val_pred_all_cls + j*y_val_pred_all  + k*y_val_pred_cls
                if args.task == 'binary':
                    mcc = matthews_corrcoef(val_labels, np.round(y_val_pred))
                    if mcc > best_mcc:
                        best_mcc = mcc
                        best_i, best_j, best_k = i, j, k
                elif args.task == 'regression':
                    mse = mean_squared_error(val_labels, y_val_pred)
                    if mse < best_mse:
                        best_mse = mse
                        best_i, best_j, best_k = i, j, k
                else:
                    # multiclass - confirm we need to round pred?
                    weighted_f1 = f1_score(val_labels, y_val_pred, 
                                           labels=[i for i in range(args.multi_num_classes)],
                                           average='weighted')
                    if weighted_f1 > best_weighted_f1:
                        best_weighted_f1 = weighted_f1
                        best_i, best_j, best_k = i, j, k
        
    y_test_pred = best_i*y_test_pred_all_cls + best_j*y_test_pred_all + best_k*y_test_pred_cls
    logger.info("Three models combined:")
    logger.info("ESM+ChemBERTa2+cls: %s, ESM+ChemBERTa2: %s, cls-token: %s" %(best_i, best_j, best_k))
    get_performance_metrics(pred = y_test_pred, true = test_labels)

    # save model predictions:
    if args.save_pred_path == "":
        pred_path = args.save_path / 'xgboost_predictions'
    else:
        pred_path = args.save_pred_path
        
    try:
        os.mkdir(str(pred_path))
    except:
        pass 
    np.save(join(str(pred_path), "y_test_pred.npy"), y_test_pred)
    np.save(join(str(pred_path), "test_indices.npy"), np.array(test_indices))

    if args.world_size != -1:
        cleanup()

### TO DO: fix multiclass option for xgboost, and add metrics
if __name__ == '__main__':
    parser = get_xgboost_arguments()
    args = parser.parse_args()

    if not args.save_path.exists() or not (args.save_path / 'model').exists():
        raise Exception('Invalid save path.')
    
    n_gpus = len(list(range(torch.cuda.device_count())))

    # get the preprocessing and config args
    preprocessing_parser = get_preprocessing_parser(
         default_config_files=[str(args.save_path / 'config.toml')]
     )
    preprocessing_args, unknown = preprocessing_parser.parse_known_args()
    training_parser = get_training_parser(
         default_config_files=[str(args.save_path / args.train_config)]
    )
    training_args, unknown = training_parser.parse_known_args()
    args.num_hidden_layers = training_args.num_hidden_layers
    args.task = training_args.task
    args.multi_num_classes = training_args.multi_num_classes
    args.embedding_size = preprocessing_args.embedding_size
    args.dataset = preprocessing_args.dataset
    args.smiles = preprocessing_args.smiles
    args.full_sequence = preprocessing_args.full_sequence
    args.context_size = preprocessing_args.context_size
    args.esm_version = preprocessing_args.esm_version
    
    # Set up the device
    # Check if multiple GPUs are available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        gpus = len(device_ids)
        args.world_size = gpus
    else:
        device = torch.device('cpu')
        args.world_size = -1

    save_config(parser, args, args.save_path / 'train_xgboost_config.toml')
    
    try:
        if torch.cuda.is_available():
            mp.spawn(trainer, nprocs=args.world_size, args=(args, device))
        else:
            # trainer(0, args, device)
            raise Exception('No GPU detected.')
    except Exception as e:
        print(e)
