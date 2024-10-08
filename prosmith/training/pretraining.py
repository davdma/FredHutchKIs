import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from prosmith.utils.modules import (
    MM_TN,
    MM_TNConfig,
)
from prosmith.utils.data_utils import SMILESProteinFastDataset
from prosmith.utils.train_utils import *
from prosmith.utils.gen_utils import save_config

import os
import pandas as pd
import shutil
import random
from configargparse import ArgumentParser
import time
import logging
import numpy as np
from pathlib import Path
from time import gmtime, strftime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, precision_score, recall_score, f1_score
from lifelines.utils import concordance_index
import wandb
from prosmith.preprocessing.preprocessing import get_preprocessing_parser

def get_pretraining_parser(**kwargs):
    parser = ArgumentParser(**kwargs)
    # centralize into one folder with all the files
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="The path where preprocessed data for the model is saved.",
    )
    parser.add_argument(
        "--esm_version",
        default='2',
        choices=['1b', '2'],
        help=f"ESM model version: 1b, 2 (default: 2)"
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=1280,
        choices=[1280, 2560, 5120],
        help=f"ESM model embedding dim: 1280, 2560, 5120 (default: 1280)."
    )
    # train_file, val_file, embed_path, save_model_path
    parser.add_argument(
        "--batch_size",
        default=12,
        type=int,
        help="Batch size per GPU",
    )
    # to cut down training time
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Use a subset of the training set for each epoch.",
    )
    parser.add_argument(
        "--task",
        default='regression',
        choices=['binary', 'multiclass', 'regression'],
        help=f"Task type: regression, binary, multiclass (default: regression)"
    )
    parser.add_argument(
        "--multi_num_classes",
        type=int,
        default=3,
        help=f"Number of classes for multiclassification (default: 3)."
    )
    parser.add_argument(
        "--early_stopping",
        action='store_true',
        help=f"Enable early stopping."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help=f"Number of epochs without improvement before early stopping."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs", 
        default=50,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=6,
        type=int,
        help="The num_hidden_layers size of MM_TN",
    )
    parser.add_argument(
        '--port',
        default=12557,
        type=int,
        help='Port for tcp connection for multiprocessing'
    )
    parser.add_argument(
        '--log_name',
        default="default",
        type=str,
        help='Will be added to the file name of the log file'
    )
    parser.add_argument(
        '--project',
        default="FredHutchPretrain",
        type=str,
        help='Wandb project name.'
    )
    parser.add_argument(
        "--checkpoint",
        action='store_true',
        help=f"Run from last checkpoint."
    )
    return parser

def train(args, model, trainloader, optimizer, criterion, device, gpu, epoch, logger):
    model.train()
    train_loss = 0.
    
    logger.info(f"Training for epoch {epoch+1}")
    for step, batch in enumerate(trainloader):
        # logging.info(f"Batch: {step}, Time ={np.round(time.time()-start_time)}")
        smiles_emb, smiles_attn, protein_emb, protein_attn, labels, _ = batch
        smiles_emb = smiles_emb.to(device)
        smiles_attn = smiles_attn.to(device)
        protein_emb = protein_emb.to(device)
        protein_attn = protein_attn.to(device)
        labels = labels.to(device)
        
        # zero the gradients
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(
            smiles_emb=smiles_emb, 
            smiles_attn=smiles_attn, 
            protein_emb=protein_emb,
            protein_attn=protein_attn,
            device=device, gpu=gpu
        )
        
        loss = criterion(outputs, labels.float())
        
        # backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # logging.info training loss after every n steps
        if step % 10 == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, args.num_train_epochs, step+1, len(trainloader), loss.item()))
        train_loss += loss.item()

        # end training early
        if step + 1 >= args.subset * len(trainloader):
            break
            
    return train_loss / (step + 1) # len(trainloader)

def evaluate(args, model, valloader, criterion, device, gpu, logger):
    # evaluate the model on validation set
    model.eval()
    y_true, y_pred = [], []
    val_loss = 0.
    
    logger.info(f"Evaluating")
    
    with torch.no_grad():
        for step, batch in enumerate(valloader):
            # move batch to device
            smiles_emb, smiles_attn, protein_emb, protein_attn, labels, _ = batch
            smiles_emb = smiles_emb.to(device)
            smiles_attn = smiles_attn.to(device)
            protein_emb = protein_emb.to(device)
            protein_attn = protein_attn.to(device)
            labels = labels.to(device)
            
            # forward pass
            outputs = model(
                smiles_emb=smiles_emb, 
                smiles_attn=smiles_attn, 
                protein_emb=protein_emb,
                protein_attn=protein_attn,
                device=device,
                gpu=gpu
            )
            
            loss = criterion(outputs, labels.float())
            val_loss += loss

            if args.task == 'binary':
                # need to apply sigmoiding to the output logits
                preds = nn.functional.sigmoid(outputs)
                y_true.extend(labels.cpu().bool())
                y_pred.extend(preds.cpu())
            elif args.task == 'multiclass':
                preds = outputs 
                y_true.extend(labels.cpu())
                y_pred.extend(preds.cpu())
            else:
                # regression
                preds = outputs
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

    # calculate evaluation metrics
    val_metrics = {}
    if args.task == 'regression':
        MSE = mean_squared_error(y_true, y_pred)
        MAE = mean_absolute_error(y_true, y_pred)
        val_loss /= len(valloader)
        R2 = r2_score(y_true, y_pred)
        CI = concordance_index(y_true, y_pred)
        logger.info('Val MSE: {:.4f}, Val Loss: {:.4f}, Val R2: {:.4f}, Val CI: {:.4f}'.format(MSE, val_loss, R2, CI))
        
        val_metrics['MSE'] = MSE
        val_metrics['MAE'] = MAE
        val_metrics['RMSE'] = np.sqrt(MSE)
        val_metrics['CI'] = CI
        val_metrics['R2'] = R2
        return val_loss, val_metrics
    elif args.task == 'binary':
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        y_pred = np.round(y_pred).astype(int) # rounding to int for binary classes
        acc = accuracy(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        val_loss /= len(valloader)
        logger.info('Val Accuracy: {:.4f}, Val AUC: {:.4f}, Val precision: {:.4f}, Val recall: {:.4f}, Val f1: {:.4f}, Val Loss: {:.4f}'.format(acc, auc_score, precision, recall, f1, val_loss))

        val_metrics['Accuracy'] = acc
        val_metrics['Precision'] = precision
        val_metrics['Recall'] = recall
        val_metrics['F1'] = f1
        val_metrics['ROC AUC'] = auc_score
        return val_loss, val_metrics
    else:
        # multiclass - want accuracy, precision, recall, f1 for each of the num classes!
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred)
        y_pred = np.argmax(y_pred, dim=1)
        labels = [i for i in range(args.multi_num_classes)]
        acc = accuracy(y_true, y_pred, labels=labels, average=None)
        precision = precision_score(y_true, y_pred, labels=labels, average=None)
        recall = recall_score(y_true, y_pred, labels=labels, average=None)
        f1 = f1_score(y_true, y_pred, labels=labels, average=None)
        for i in range(args.multi_num_classes):
            val_metrics[f'Accuracy (Class {i})'] = acc[i]
            val_metrics[f'Precision (Class {i})'] = precision[i]
            val_metrics[f'Recall (Class {i})'] = recall[i]
            val_metrics[f'F1 (Class {i})'] = f1[i]
            
        val_loss /= len(valloader)
        return val_loss, val_metrics

# Define the main function for training the model
def trainer(gpu, args, device):
    if is_cuda(device):
        setup(gpu, args.world_size, str(args.port))
        torch.manual_seed(0)
        torch.cuda.set_device(gpu)

    is_distributed = args.world_size > 1
    # set up logging
    log_file = args.save_path / ("log_" + args.log_name + '_' + str(dist.get_rank()) + '_' + str(args.world_size) + '.log')
    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    fhandler = logging.FileHandler(filename=str(log_file), mode='a')
    logger.addHandler(fhandler)
    logger.info(args)

    config = MM_TNConfig.from_dict({
        "s_hidden_size":600,
        "p_hidden_size": args.embedding_size,
        "hidden_size": 768,
        "max_seq_len": 1276,
        "num_hidden_layers": args.num_hidden_layers,
        "task": "regression_no_scaling" if args.task == 'regression' else args.task,
        "num_classes": args.multi_num_classes if args.task == 'multiclass' else 1
    })

    # initialize wandb run for main process
    if is_main_process():
        logger.info('Running the main process...')
        run = wandb.init(
            project=args.project,
            config={ # later can incorporate config parameters from save path
                'esm_version': args.esm_version,
                'embedding_size': args.embedding_size,
                'save_path': str(args.save_path),
                'batch_size': args.batch_size,
                'subset': args.subset,
                'task': args.task,
                'num_classes': args.multi_num_classes if args.task == 'multiclass' else 1,
                'early_stopping': args.early_stopping,
                'patience': args.patience,
                'learning_rate': args.learning_rate,
                'num_train_epochs': args.num_train_epochs,
                'num_hidden_layers': args.num_hidden_layers,
                'log_name': args.log_name,
                'checkpoint': args.checkpoint
            },
        )

    logger.info(f"Loading model")
    model = MM_TN(config)
    
    if is_cuda(device):
        model = model.to(gpu)
        model = DDP(model, device_ids=[gpu])

    if args.checkpoint and (args.save_path / f'model/model_{args.embedding_size}_v{args.esm_version}_chkpt.pkl').exists():
        logger.info(f'Loading model from last checkpoint')
        try:
            state_dict = torch.load(str(args.save_path / f'model/model_{args.embedding_size}_v{args.esm_version}_chkpt.pkl'))
            model.load_state_dict(state_dict)
            logger.info("Successfully loaded from checkpoint")
        except Exception as err:
            logger.debug("Failed to load from checkpoint.")
            raise err
    else:
        logger.info("No checkpoint loaded.")

    # change loss type for binary and multiclass model!!!
    # MSELoss for regression, BCELoss for binary, CE cross entropy for multiclass
    if args.task == 'regression':
        criterion = nn.MSELoss()
    elif args.task == 'binary':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # train the model
    logger.info(f"Start training")
    if args.early_stopping:
        early_stopper = EarlyStopper(patience=args.patience)
        flag_tensor = torch.zeros(1).to(device)
    else:
        best_val_loss = float('inf')
        best_val_metrics = None

    # create the dataloader
    logger.info(f"Loading dataset to {device}:{gpu}")
    train_dataset = SMILESProteinFastDataset(data_path=args.save_path / 'cleaned_BindingDB_IC50_train.csv',
                                        embed_dir=args.save_path / f'embeddings_{args.embedding_size}_v{args.esm_version}')
    val_dataset = SMILESProteinFastDataset(data_path=args.save_path / 'cleaned_BindingDB_IC50_test.csv',
                                       embed_dir=args.save_path / f'embeddings_{args.embedding_size}_v{args.esm_version}')

    if is_distributed:
        # distributed training
        logger.info(f"Distributed training...")
        trainsampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=args.world_size, rank=gpu, drop_last=True)
        valsampler = DistributedSampler(val_dataset, shuffle=True, num_replicas=args.world_size, rank=gpu, drop_last=True)
    else: 
        trainsampler = None
        valsampler = None

    logger.info(f"Loading dataloader")
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(trainsampler is None), num_workers=1, sampler=trainsampler, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=(valsampler is None), num_workers=1, sampler=valsampler, pin_memory=True)
        
    for epoch in range(args.num_train_epochs):
        start_time = time.time()
        logger.info(f"Training started for epoch: {epoch+1}")
        if is_distributed:
            trainsampler.set_epoch(epoch)
            valsampler.set_epoch(epoch)
            
        train_loss = train(args, model, trainloader, optimizer, criterion, device, gpu, epoch, logger)
        train_end_time = time.time()
        logger.info(f"Training loop complete: finished in {train_end_time - start_time} seconds.")

        # wandb logging
        if is_main_process():
            wandb.log({"train loss": train_loss})

        logger.info(f"Evaluating...")
        eval_start_time = time.time()
        # val loss is reported loss, val_metrics should be dictionary containing additional evaluation metrics
        val_loss, val_metrics = evaluate(args, model, valloader, criterion, device, gpu, logger)
        eval_end_time = time.time()
        logger.info(f"Evaluation complete: finished in {eval_end_time - eval_start_time} seconds.")
        logger.info('-' * 80)
        logger.info(f'| Device id: {gpu} | End of epoch: {(epoch+1)} | Train Loss {train_loss} | Val Loss: {val_loss}')

        val_log = {"val loss": val_loss}
        for metric, metric_value in val_metrics.items():
            logger.info(f'Additional validation metric ({metric}): {str(metric_value)}')
            val_log["val " + metric] = metric_value
        logger.info('-' * 80)

        # wandb logging
        if is_main_process():
            wandb.log(val_log)

        if args.early_stopping:
            if is_main_process():
                # early stopping only in main process
                early_stopper.step(val_loss)
                if early_stopper.is_stopped():
                    flag_tensor += 1
                    
                if early_stopper.is_best_epoch():
                    early_stopper.store_metric(val_metrics)
                    torch.save(model.state_dict(), os.path.join(str(args.save_path / 'model'), f'model_{args.embedding_size}_v{args.esm_version}_chkpt.pkl'))

            dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM) # all gpus wait
            if flag_tensor == 1:
                logger.info('Training has been early stopped.')
                break
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = val_metrics
                if is_main_process():
                    torch.save(model.state_dict(), os.path.join(str(args.save_model_path / 'model'), f'model_{args.embedding_size}_v{args.esm_version}_chkpt.pkl'))

        end_time = time.time()
        logger.info(f'Epoch total time: {end_time - start_time}s')

    # log best metrics on wandb
    logger.info('All training complete.')
    if is_main_process():
        if args.early_stopping:
            for metric, metric_value in early_stopper.get_metric().items():
                run.summary["Best " + metric] = metric_value
        else:
            for metric, metric_value in best_val_metrics:
                run.summary["Best " + metric] = metric_value
        run.finish()
        
    if args.world_size != -1:
        cleanup()

if __name__ == '__main__':
    parser = get_pretraining_parser()
    args = parser.parse_args()
    
    n_gpus = len(list(range(torch.cuda.device_count())))
    eff_bs = n_gpus*args.batch_size # effective batch size is number of gpus (want more gpus for DDP!)
    args.port = args.port
    
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

    if not args.save_path.exists():
        raise Exception('Invalid save path.')
    
    try:
        model_path = args.save_path / 'model'
        model_path.mkdir(exist_ok=True, parents=True)
    except:
    	raise Exception('Unable to make save directory.')

    save_config(parser, args, args.save_path / f'pretrain_{args.embedding_size}_v{args.esm_version}_config.toml')

    if not wandb.login():
        raise Exception('Failed to login to wandb.')
        
    try:
        if torch.cuda.is_available():
            mp.spawn(trainer, nprocs=args.world_size, args=(args, device))
        else:
            # trainer(0, args, device)
            raise Exception('No GPU detected.')
    except Exception as err:
        raise err # why print only?
