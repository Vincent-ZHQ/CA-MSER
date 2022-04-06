import sys
import argparse
import pickle
from data_utils import SERDataset
import torch
import numpy as np
# from model import SER_AlexNet, SER_AlexNet_GAP, SER_CNN
from models.ser_model import Ser_Model
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import os
import random
from tqdm import tqdm
from collections import Counter
from torch.backends import cudnn

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


colors_per_class = {
    0 : [0, 0, 0],
    1 : [255, 107, 107],
    2 : [100, 100, 255],
    3 : [16, 172, 132],
}
def main(args):
    
    # Aggregate parameters
    params={
            #model & features parameters
            'ser_task': 'SLM',

            #training
            'repeat_idx': args.repeat_idx,
            'val_id': args.val_id,
            'test_id': args.test_id,
            'num_epochs':args.num_epochs,
            'early_stop':args.early_stop,
            'batch_size':args.batch_size,
            'lr':args.lr,
            'random_seed':args.seed,
            'use_gpu':args.gpu,
            'gpu_ids': args.gpu_ids,
            
            #best mode
            'save_label': args.save_label,
            
            #parameters for tuning
            'oversampling': args.oversampling,
            'pretrained': args.pretrained
            }

    print('*'*40)
    print(f"\nPARAMETERS:\n")
    print('*'*40)
    print('\n')
    for key in params:
        print(f'{key:>15}: {params[key]}')
    print('*'*40)
    print('\n')

    #set random seed
    seed_everything(params['random_seed'])
    # Load dataset
    with open(args.features_file, "rb") as fin:
        features_data = pickle.load(fin)
    ser_dataset = SERDataset(features_data,
                               val_speaker_id=args.val_id,
                               test_speaker_id=args.test_id,
                               oversample=args.oversampling
                               )
    # Train
    train_stat = train(ser_dataset, params, save_label=args.save_label)

    return train_stat


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a SER  model in an iterative-based manner with "
                    "pyTorch and IEMOCAP dataset.")

    #Features
    parser.add_argument('features_file', type=str,
        help='Features extracted from `extract_features.py`.')
    
    #Training
    parser.add_argument('--repeat_idx', type=str, default='0',
        help='ID of repeat_idx')
    parser.add_argument('--val_id', type=str, default='1F',
        help='ID of speaker to be used as validation')
    parser.add_argument('--test_id', type=str, default='1M',
        help='ID of speaker to be used as test')
    parser.add_argument('--num_epochs', type=int, default=200,
        help='Number of training epochs.') 
    parser.add_argument('--early_stop', type=int, default=4,
        help='Number of early stopping epochs.') 
    parser.add_argument('--batch_size', type=int, default=32,
        help='Mini batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, 
        help='Learning rate.')
    parser.add_argument('--seed', type=int, default=100,
        help='Random seed for reproducibility.')
    parser.add_argument('--gpu', type=int, default=1,
        help='If 1, use GPU')
    parser.add_argument('--gpu_ids', type=list, default=[0],
        help='If 1, use GPU')        
    
    #Best Model
    parser.add_argument('--save_label', type=str, default=None,
        help='Label for the current run, used to save the best model ')

    #Parameters for model tuning
    parser.add_argument('--oversampling', action='store_true',
        help='By default, no oversampling is applied to training dataset.'
             'Set this to true to apply random oversampling to balance training dataset')
    
    parser.add_argument('--pretrained', action='store_true',
        help='By default, SER_AlexNet or SER_AlexNet_GAP model weights are'
             'initialized randomly. Set this flag to initalize with '
             'ImageNet pre-trained weights.')

    return parser.parse_args(argv)



def test(mode, params, model, criterion_ce, criterion_mml, test_dataset, batch_size, device,
         return_matrix=False):

    """Test an SER model.

    Parameters
    ----------
    model
        PyTorch model
    criterion
        loss_function
    test_dataset
        The test dataset
    batch_size : int
    device
    return_matrix : bool
        Whether to return the confusion matrix.

    Returns
    -------
    loss, weighted accuracy (WA), unweighted accuracy (UA), confusion matrix 
       

    """
    total_loss = 0
    test_preds_segs = []
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # we'll store the features as NumPy array of size num_images x feature_size and the labels
    sne_features1 = None
    sne_features2 = None
    sne_features3 = None
    sne_features4 = None
    
    sne_features5 = None
    sne_features6 = None
    sne_features7 = None
    sne_features8 = None
    
    sne_features9 = None
    sne_features10 = None
    sne_features11 = None
    sne_features12 = None
    
    sne_features13 = None
    sne_features14 = None
    sne_features15 = None
    sne_features16 = None
    
    sne_features17 = None
    sne_features18 = None
    sne_features19 = None
    sne_features20 = None
    
    # sne_features = [sne_features1, sne_features2, sne_features3, sne_features4, sne_features5, sne_features6, sne_features7, sne_features8, sne_features9, sne_features10, sne_features11, sne_features12, sne_features13, sne_features14, sne_features15, sne_features16, sne_features17, sne_features18, sne_features19, sne_features20]
    
    sne_features = [None, None, None, None]
    out_features = None
    sne_labels = []
        
    model.eval()

    # for i, test_batch in enumerate(test_loader):
    with tqdm(test_loader) as td:
        for test_batch in td:
                
            # Send data to correct device
            test_data_spec_batch = test_batch['seg_spec'].to(device)
            test_data_mfcc_batch = test_batch['seg_mfcc'].to(device)
            test_data_audio_batch = test_batch['seg_audio'].to(device)
            test_labels_batch =  test_batch['seg_label'].to(device,dtype=torch.long)
        
            labels = test_batch['seg_label'].cpu().detach().numpy()
            sne_labels += list(labels)
        
            # Forward
            test_outputs = model(test_data_spec_batch, test_data_mfcc_batch, test_data_audio_batch)
            test_preds_segs.append(f.log_softmax(test_outputs['M'], dim=1).cpu())

            #test loss
            test_loss_ce = criterion_ce(test_outputs['M'], test_labels_batch)
            # test_loss_mml = criterion_mml(test_outputs['M'], test_labels_batch)
            test_loss = test_loss_ce#  + test_loss_mml
           
            total_loss += test_loss.item()
            
            '''
            # VISULAIZATION
            for index in range(4):
                str_idx = 'F' + str(index+1)
                current_features = test_outputs[str_idx].cpu().numpy()
                if sne_features[index] is not None:
                    sne_features[index] = np.concatenate((sne_features[index], current_features))
                else:
                    sne_features[index] = current_features     
            '''           
    '''
    # VISULAIZATION
    if mode == 'TEST':
        for index in range(4):
            tsne = TSNE(n_components=2).fit_transform(sne_features[index])
            visualize_tsne_2(str(index), tsne, sne_labels, params)    
    '''
    # Average loss
    test_loss = total_loss / len(test_loader)

    # Accumulate results for val data
    test_preds_segs = np.vstack(test_preds_segs)
    test_preds = test_dataset.get_preds(test_preds_segs)
    
    # Make sure everything works properly
    assert len(test_preds) == test_dataset.n_actual_samples
    test_wa = test_dataset.weighted_accuracy(test_preds)
    test_ua = test_dataset.unweighted_accuracy(test_preds)
    test_cor = test_dataset.confusion_matrix_iemocap(test_preds)

    results = (test_loss, test_wa*100, test_ua*100)
    
    if return_matrix:
        test_conf = test_dataset.confusion_matrix_iemocap(test_preds)
        return results, test_conf
    else:
        return results

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
    
def visualize_tsne_points_2(name, tx, ty, labels, params):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, s=1, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.show()
    
    t = round(time.time()*1000)
    t_str = time.strftime('%H_%M_%S',time.localtime(t/1000))

    img_path = './results/t-SNE/' + t_str + '_' + name + '_' + params['repeat_idx'] + '_' + params['test_id'] + '.png'
    # finally, show the plot
    fig.savefig(img_path, dpi=fig.dpi)
    
def visualize_tsne_points_3(name, tx, ty, tz, labels, params):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        current_tz = np.take(tz, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, current_tz, s=4, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    
    t = round(time.time()*1000)
    t_str = time.strftime('%H_%M_%S',time.localtime(t/1000))

    img_path = './results/t-SNE/' + t_str + '_' + name + '_' + params['repeat_idx'] + '_' + params['test_id'] + '.png'
    print(img_path)
    # finally, show the plot
    fig.savefig(img_path, dpi=fig.dpi)
        
def visualize_tsne_2(name, tsne, labels, params):

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points_2(name, tx, ty, labels, params)

def visualize_tsne_3(name, tsne, labels, params):

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tz = tsne[:, 2]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    tz = scale_to_01_range(tz)

    # visualize the plot: samples as colored points
    visualize_tsne_points_3(name, tx, ty, tz, labels, params)
    
def train(dataset, params, save_label='default'):

    #get dataset
    train_dataset = dataset.get_train_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=params['batch_size'], 
                                shuffle=True)  
                                
    val_dataset = dataset.get_val_dataset()
    test_dataset = dataset.get_test_dataset()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("pytorch version: ", torch.__version__)
    print("cuda version: ", torch.version.cuda)
    print("cudnn version: ", torch.backends.cudnn.version())
    print("gpu name: ", torch.cuda.get_device_name())
    print("gpu index: ", torch.cuda.current_device())
    
    #select device
    if params['use_gpu'] == 1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Construct model, optimizer and criterion
    batch_size = params['batch_size']
    
    # print(type(Ser_Model()))
    model = Ser_Model().to(device) 
    
    print(model.eval())
    print(f"Number of trainable parameters: {count_parameters(model.train())}")
    print('\n')

    #Set loss criterion and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mml = nn.MultiMarginLoss(margin=0.5) 

    loss_format = "{:.04f}"
    acc_format = "{:.02f}%"
    acc_format2 = "{:.02f}"
    best_val_wa = 0
    best_val_ua = 0
    save_path = save_label + '.pth'
    best_val_loss = 1e8
    best_val_acc = -1e8
    
    all_train_loss =[]
    all_train_wa =[]
    all_train_ua=[]
    all_val_loss=[]
    all_val_wa=[]
    all_val_ua=[]
    train_preds = []
    
    print("Start Training!!!")
    
    for epoch in range(params['num_epochs']):
        
        y_pred = {'M': [], 'A': []}
        y_true = {'M': [], 'A': []}
        
        # adjust_learning_rate(params['lr'], optimizer, epoch)
        
        #get current learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        
        # Train one epoch
        total_loss = 0
        train_preds = []
        target=[]
        model.train()
        
        # for i, train_batch in enumerate(train_loader):
        with tqdm(train_loader) as td:
            for train_batch in td:
                
                # Clear gradients
                optimizer.zero_grad()
            
                # Send data to correct device
                train_data_spec_batch = train_batch['seg_spec'].to(device)
                train_data_mfcc_batch = train_batch['seg_mfcc'].to(device)
                train_data_audio_batch = train_batch['seg_audio'].to(device)
                train_labels_batch =  train_batch['seg_label'].to(device,dtype=torch.long)
            
                # Forward pass
                outputs = model(train_data_spec_batch, train_data_mfcc_batch, train_data_audio_batch)
            
                #for m in params['ser_task']:
                #    y_pred[m].append(f.log_softmax(outputs[m], dim=1).cpu().detach().numpy())
                #    y_true.append
                
                train_preds.append(f.log_softmax(outputs['M'], dim=1).cpu().detach().numpy())
                
                # Compute the loss, gradients, and update the parameters
                train_loss_ce = criterion_ce(outputs['M'], train_labels_batch)
                # train_loss_mml = criterion_mml(outputs['M'], train_labels_batch)
                train_loss = train_loss_ce# + train_loss_mml

                train_loss.backward()
                total_loss += train_loss.item()
                optimizer.step()
            
        # Evaluate training data
        train_loss = total_loss / len(train_loader)
        # Accumulate results for train data
        train_preds = np.vstack(train_preds)
        train_preds = train_dataset.get_preds(train_preds)
        
        # Make sure everything works properly
        train_wa = train_dataset.weighted_accuracy(train_preds) * 100
        train_ua = train_dataset.unweighted_accuracy(train_preds) * 100
        #train_cor = train_dataset.confusion_matrix_iemocap(train_preds)
        
        all_train_loss.append(loss_format.format(train_loss))
        all_train_wa.append(acc_format2.format(train_wa))
        all_train_ua.append(acc_format2.format(train_ua))
    
    
        #Validation
        with torch.no_grad():
            val_result = test('VAL', params,
                model, criterion_ce, criterion_mml, val_dataset, 
                batch_size=64, 
                device=device)

            val_loss = val_result[0]
            val_wa = val_result[1]
            val_ua = val_result[2]

            # Update best model based on validation UA
            # if val_loss < (best_val_loss - 1e-6):
            if val_wa + val_ua > best_val_acc:
                
                print("True")
                best_val_ua = val_ua
                best_val_wa = val_wa
                best_val_loss = val_loss
                best_val_acc = val_wa + val_ua
                best_epoch = epoch
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
        print(best_epoch, epoch)

        all_val_loss.append(loss_format.format(val_loss))
        all_val_wa.append(acc_format2.format(val_wa))
        all_val_ua.append(acc_format2.format(val_ua))
        
        print(f"Epoch {epoch+1}  (lr = {current_lr})\
        Loss: {loss_format.format(train_loss)} - {loss_format.format(val_loss)} - WA: {acc_format.format(val_wa)} <{acc_format.format(best_val_wa)}> - UA: {acc_format.format(val_ua)} <{acc_format.format(best_val_ua)}>")
        
        # early stop
        if (epoch - best_epoch >= params['early_stop']) and (epoch > 5):
            break    
        #break
        
    # Test on best model
    with torch.no_grad():
        model.load_state_dict(torch.load(save_path))

        test_result, confusion_matrix = test('TEST', params,
            model, criterion_ce, criterion_mml, test_dataset, 
            batch_size=64, #params['batch_size'],
            device=device, return_matrix=True)

        print("*" * 40)
        print("RESULTS ON TEST SET:")
        print("Loss:{:.4f}\tWA: {:.2f}\tUA: "
              "{:.2f}".format(test_result[0], test_result[1], test_result[2]))
        print("Confusion matrix:\n{}".format(confusion_matrix[1]))   
        

    return(epoch, best_epoch, all_train_loss, all_train_wa, all_train_ua,
            all_val_loss, all_val_wa, all_val_ua,
            loss_format.format(test_result[0]), 
            acc_format2.format(test_result[1]),
            acc_format2.format(test_result[2]),
            confusion_matrix[0])


# seeding function for reproducibility
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #cudnn.benchmark=True
    #cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(lr_0, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_0 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
# to count the number of trainable parameter in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
