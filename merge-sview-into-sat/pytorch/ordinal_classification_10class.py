from __future__ import division
from __future__ import print_function
#import os.path

# all libraries:
import numpy as np
import time as tm
import sys
sys.path.insert(1, '../../tools')
import pandas as pd
import scipy.stats as stats
import h5py
from scipy.linalg import toeplitz
import pickle
from sklearn.metrics import confusion_matrix
import partitioning
import argparse
import os
from skimage.io import imsave
import datasets_sview_into_satellite as datasets
import datasets_sview as datasets_sview
import tifffile as tiff

# deep learning part: 
import torch 
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#================== PARSING INPUTS ==========================
#== TO FINISH THE PARSING AND FEEDING INTO THE MODEL WITH VARIOUS CHOICES ==#
parser = argparse.ArgumentParser(description='ordinal_classification_sview')
parser.add_argument("--mode", "-m", help="Training format >> 0:test, 1:train, 2:fine_tune_test, 3: fine_tune_train", default=1, type=int, choices=[0,1,2,3])
parser.add_argument("--imgfile", "-i", help="hdf5 image file", default=None, type=str)
parser.add_argument("--labfile", "-l", help="label pickle file", default=None, type=str)
parser.add_argument("--satfile", "-s", help="satellite file", default=None, type=str)
parser.add_argument("--satlabfile", help="label file corresponding to the satellite hdf5 data", default=None, type=str)
parser.add_argument("--satpatch", help="size of the satellite patches to extract", default=200, type=int)
parser.add_argument("--labelName", "-n", help="label name", default=None, type=str)
parser.add_argument("--sviewLabelName", help="sview label name", default=None, type=str)
parser.add_argument("--modelName", help="trained model name", default=None, type=str)
parser.add_argument("--gen_part", help="generate or use partition", action="store_true")
parser.add_argument("--part_file", help="satellite partition file", default=None, type=str)
parser.add_argument("--sview_part_file", help="sview partition file", default=None, type=str)
parser.add_argument("--validation_flag", help="cross validation (0) | train-test-validation split (1) | train-test split (2) | train-test per class split (3)", default=1, type=int, choices=[0,1,2,3])
parser.add_argument("--part_kn", help="total number of partitions in cross-validation", default=5, type=int)
parser.add_argument("--part_kp", help="partition number to work with in cross-validation", default=0, type=int)
parser.add_argument("--train_part", help="training set partition size - between 0 and 1", default=0.6, type=float)
parser.add_argument("--test_part", help="test set partition size - between 0 and 1", default=0.3, type=float)
parser.add_argument("--validation_part", help="validation set partition size - between 0 and 1", default=0.1, type=float)
parser.add_argument("--train_size", help="size of the training set to be used - between 0 and 1", default=1.0, type=float)
parser.add_argument("--city_name", help="name of the city", default="london")
parser.add_argument("--num_epochs", help="number of epochs", default=10, type=int)
parser.add_argument("--batch_size", help="batch size", default=1, type=int)
parser.add_argument("--lrrate", help="learning rate", default=2e-6, type=float)
parser.add_argument("--testsetlabel", help="class for test set partition", default=0, type=int)
parser.add_argument("--gpu_num", help="number of the gpu: 0,1,2", default="0", type=str)
parser.add_argument("--aug_prob", help="augmentation probability", default=0.0, type=float)
parser.add_argument("--pre_train_iter", help="number of iterations the model should be pre-trained using sview-only", default=0, type=int)
parser.add_argument("--finetune_flag", help="boolean flag indicating whether we should train network with UNet", action="store_true")
args = parser.parse_args()

train_format = args.mode # 0: test, 1: train, 2: refine, 3: refine_test,
TRAIN = False
TEST = False
RTRAIN = False
RTEST = False

outformat = 'none'
if train_format == 0:
    TEST = True
    outformat = 'train'
elif train_format == 1:
    TRAIN = True
    outformat = 'train'
elif train_format == 2:
    RTRAIN = True
    outformat = 'refine'
elif train_format == 3:
    RTEST = True
    outformat = 'refine'

img_hdf5_file = args.imgfile
lab_pickle_file = args.labfile
label_name = args.labelName
sview_label_name = args.sviewLabelName
label_set = np.arange(1, 11)
trained_model_name = args.modelName
part_gen = args.gen_part # should I generate partitions for mapping
part_file = args.part_file # name of the file of the k partitions (pickle)
sview_part_file = args.sview_part_file # name of the files that hold the partitions for sview images
validation_flag = args.validation_flag
part_kn = args.part_kn # total number of partitions wanted
part_kp = args.part_kp # the partition number we want to be working with
train_part = args.train_part
test_part = args.test_part
validation_part = args.validation_part
train_size = args.train_size
outformat = '{}_{}'.format(outformat, train_size)
city_name = args.city_name
print('Given batch size is {} but at this point only batch size of one is allowed.'.format(args.batch_size))
batch_size=args.batch_size
#batch_size=1
num_epochs=args.num_epochs
lrrate = args.lrrate
label_test = args.testsetlabel
aug_prob = args.aug_prob
sat_hdf5_file = args.satfile
sat_hdf5_labfile = args.satlabfile
sat_patch = args.satpatch
pre_train_iter = args.pre_train_iter
finetune_flag = args.finetune_flag
# = name that will be used to write the model from now on = #
out_pre = '{}_{}_{}_{}'.format(city_name, outformat, trained_model_name, "%.5f" % lrrate)
# = setting the GPU to use = #
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

## print information
if TRAIN:
    print('Training...')
elif TEST:
    print('Testing...')
elif RTRAIN:
    print('Fine-tuning...')
else:
    print('Testing fine-tuned model...')
print('Image file, Label file: {}'.format(img_hdf5_file))
print('Trained model name: {}'.format(trained_model_name))
print('Label name: {}'.format(label_name))
print('Generating partitions...{}'.format(part_gen))
print('Partition file name...{}'.format(part_file))
print('Partition file sview name...{}'.format(part_file))
print('Satellite file name...{}'.format(sat_hdf5_file))
print('Satellite label file name...{}'.format(sat_hdf5_labfile))
print('Satellite patch size...{}'.format(sat_patch))
print('Current label set...{}'.format(label_set))
print('Label csv file for street-view images...{}'.format(lab_pickle_file))
print('Label name from the sview file...{}'.format(sview_label_name))
print('Pre of the files written...{}'.format(out_pre))
if validation_flag == 0:
    print('Running {} fold cross validation...'.format(part_kn))
elif validation_flag == 1:
    print('Train: {}, Validation: {}, Test: {} divide'.format(train_part, validation_part, test_part))
elif validation_flag == 2:
    print('Train: {}, Test: {} divide'.format(train_part, test_part))
    if train_part + test_part < 1.0:
        print('IMPORTANT: Train and test portions do not add up to 1.')
elif validation_flag == 3:
    print('Test class(es): {}'.format(label_test))
print('Training size...{}'.format(train_size))
print('Final output file name acronym...{}'.format(outformat))
print('City name: {}'.format(city_name))
print('Number of iterations for pre-training only sview: {}'.format(pre_train_iter))
print('Should I train the network along with UNet: {}'.format(finetune_flag))


#==============================================================================
print('loading training dataset...')
# only use with part_gen = False
if part_gen: 
    print('The code does not generate partitions. Please use sview-only code to generate stree-view-level partitions then convert them to satellite partitions.')
    sys.exit(1)

if validation_flag == 0: # meaning we are doing cross validation
    DS = datasets.Dataset_CrossValidation(sat_hdf5_file, sat_hdf5_labfile, 
                                        img_hdf5_file, label_name, sat_patch, 
                                        label_set, lab_pickle_file, sview_label_name)
    DS.pick_label(part_gen, part_file, part_kn=part_kn, part_kp=part_kp, vsize=validation_part)
    DSsview = datasets_sview.Dataset_CrossValidation(img_hdf5_file, lab_pickle_file, sview_label_name, clabel_name=None)
    DSsview.pick_label(part_gen, sview_part_file, part_kn=part_kn, part_kp=part_kp, vsize=validation_part)
elif validation_flag == 1: # meaning we divide the dataset into test / validation / training.
    DS = datasets.Dataset_TVT(sat_hdf5_file, sat_hdf5_labfile, img_hdf5_file, 
                            label_name, sat_patch, label_set, lab_pickle_file, 
                            sview_label_name)
    DS.pick_label(part_gen, part_file, train_part, validation_part, psize=train_size)
    DSsview = datasets_sview.Dataset_TVT(img_hdf5_file, lab_pickle_file, sview_label_name, clabel_name=None)
    DSsview.pick_label(part_gen, sview_part_file, train_part, validation_part, psize=train_size)
elif validation_flag == 2: # meaning we divide the dataset into test and train
    DS = datasets.Dataset_TT(sat_hdf5_file, sat_hdf5_labfile, img_hdf5_file, 
                            label_name, sat_patch, label_set, lab_pickle_file, 
                            sview_label_name)
    DS.pick_label(part_gen, part_file, train_part, psize=train_size)
    DSsview = datasets_sview.Dataset_TT(img_hdf5_file, lab_pickle_file, sview_label_name, clabel_name=None)
    DSsview.pick_label(part_gen, sview_part_file, train_part, psize=train_size)
elif validation_flag == 3: # meaning we divide the dataset into test and train using the class label for test set
    DS = datasets.Dataset_TT_byclass(sat_hdf5_file, sat_hdf5_labfile, img_hdf5_file, 
                                    label_name, sat_patch, label_set, lab_pickle_file, 
                                    sview_label_name, label_test=label_test)
    DS.pick_label(part_gen, part_file, train_part, psize=train_size)
    DSsview = datasets_sview.Dataset_TT_byclass(img_hdf5_file, lab_pickle_file, sview_label_name, clabel_name=None, label_test=label_test)
    DSsview.pick_label(part_gen, sview_part_file, train_part, psize=train_size)

if part_gen:
    sys.exit(1)


print('done.')

## Need to generate the U-Net with pytorch. 
# convert the network output to probabilities
def convert2prob(h): 
    px = F.relu(torch.sigmoid(h)-2e-6)+1e-6
    PC1 = (1-px)**9
    PC2 = PC1 * 9.0  * px / (1-px)
    PC3 = PC2 * 8.0 / 2.0 * px / (1-px)
    PC4 = PC3 * 7.0 / 3.0 * px / (1-px)
    PC5 = PC4 * 6.0 / 4.0 * px / (1-px)
    PC10 = px**9
    PC9 = PC10 * 9.0 * (1-px) / px
    PC8 = PC9 * 8.0 / 2.0 * (1-px) / px
    PC7 = PC8 * 7.0 / 3.0 * (1-px) / px
    PC6 = PC7 * 6.0 / 4.0 * (1-px) / px
    PC = torch.cat([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10], dim=1)
    return PC

class UNet_Short(nn.Module): 
    def __init__(self, n_in, n_out): 
        super(UNet_Short, self).__init__()
        # encoding path
        self.enc_c1 = nn.Conv2d(n_in, 32, 3, padding=1)
        self.enc_c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.enc_c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_c4 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_c5 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_c6 = nn.Conv2d(128, 128, 3, padding=1)

        # decoding path
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec_c6 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec_c5 = nn.Conv2d(128+128, 128, 3, padding=1)
        self.dec_c4 = nn.Conv2d(128, 64, 3, padding=1)
        # concatenation happens here
        self.dec_c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_c2 = nn.Conv2d(64+64, 64, 3, padding=1)
        self.dec_c1 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec_out = nn.Conv2d(32, n_out, 3, padding=1)

    def forward(self, x): 
        x = F.relu(self.enc_c1(x)) #32
        x1_to_connect = F.relu(self.enc_c2(x)) # 64
        x = self.pool(x1_to_connect) # 64
        x = F.relu(self.enc_c3(x)) # 128
        x2_to_connect = F.relu(self.enc_c4(x)) #128
        x = self.pool(x2_to_connect) #128
        x = F.relu(self.enc_c5(x)) #128
        x = F.relu(self.enc_c6(x)) #128
        
        x = self.up(x)
        x = self.dec_c6(x)
        x = torch.cat([x, x2_to_connect], dim=1)
        x = F.relu(self.dec_c5(x))
        x = F.relu(self.dec_c4(x))
        x = self.up(x)
        x = self.dec_c3(x)
        x = torch.cat([x,x1_to_connect], dim=1)
        x = F.relu(self.dec_c2(x))
        x = F.relu(self.dec_c1(x))
        out = self.dec_out(x)
        return convert2prob(out), out


class Net(nn.Module):
    def __init__(self, n_in, n_out): 
        super(Net, self).__init__()
        self.dense1 = nn.Linear(n_in, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, n_out)

    def forward(self, x1, x2, x3, x4): 
        # first layer
        x1 = F.relu(self.dense1(x1))
        x2 = F.relu(self.dense1(x2))
        x3 = F.relu(self.dense1(x3))
        x4 = F.relu(self.dense1(x4))
        # second layer
        x1 = F.relu(self.dense2(x1))
        x2 = F.relu(self.dense2(x2))
        x3 = F.relu(self.dense2(x3))
        x4 = F.relu(self.dense2(x4))
        # third layer
        x1 = F.relu(self.dense3(x1))
        x2 = F.relu(self.dense3(x2))
        x3 = F.relu(self.dense3(x3))
        x4 = F.relu(self.dense3(x4))
        # = aggregation layer = average = #
        x = x1 / 4.0 + x2 / 4.0 + x3 / 4.0 + x4 / 4.0
        # fourth layer
        x = F.relu(self.dense4(x))
        # out layer
        out = self.dense5(x)
        return convert2prob(out), out

def integrate_sview(x_sat, svpred, sat_patch):
    r = torch.nonzero(x_sat[0,-1,:,:] > 0)
    if r.shape[0] > 0: 
        B = torch.sparse.FloatTensor(r.t(), svpred.view(r.shape[0]), x_sat.size()[2:]).to_dense()
        xs = torch.cat([x_sat[:,:4,:,:], B.view(1,1,sat_patch,sat_patch)], dim=1)
    else: 
        xs = torch.cat([x_sat[:,:4,:,:], torch.zeros(1,1,sat_patch,sat_patch).cuda()], dim=1)
    return xs

UNet = UNet_Short(5, 1)
UNet.cuda()
net = Net(4096, 1)
net.cuda()

optimizer = optim.Adam(list(UNet.parameters()) + list(net.parameters()), lr=lrrate, weight_decay=0.0001)
optimizer_sview = optim.Adam(net.parameters(), lr=lrrate, weight_decay=0.0001)
optimizer_sat = optim.Adam(UNet.parameters(), lr=lrrate, weight_decay=0.0001)

# generate necessary folders for model saving and log files:
if not os.path.exists("../log_dirs"):
    os.makedirs("../log_dirs")

if not os.path.exists("../../models"):
    os.makedirs("../../models")

if not os.path.exists("../../models/merge-sview-into-sat"):
    os.makedirs("../../models/merge-sview-into-sat")

if not os.path.exists("../../analysis"):
    os.makedirs("../../analysis")

if not os.path.exists("../../analysis/merge-sview-into-sat"):
    os.makedirs("../../analysis/merge-sview-into-sat")


# do the training
#==============================================================================
#==============================================================================
# make it multiple of 200
if validation_flag == 0:
    log_file = '../log_dirs/logs_{}_{}_out_of_{}_fold'.format(out_pre, part_kp, part_kn)
    save_name = '../../models/merge-sview-into-sat/{}_{}_out_of_{}_fold'.format(out_pre, part_kp, part_kn)
elif validation_flag == 1:
    log_file = '../log_dirs/logs_{}_division_tr{}_vl{}_te{}'.format(out_pre, train_part, validation_part, test_part)
    save_name = '../../models/merge-sview-into-sat/{}_division_tr{}_vl{}_te{}'.format(out_pre, train_part, validation_part,
                                                                                                             test_part)
elif validation_flag == 2:
    log_file = '../log_dirs/logs_{}_division_tr{}_te{}'.format(out_pre, train_part, test_part)
    save_name = '../../models/merge-sview-into-sat/{}_division_tr{}_te{}'.format(out_pre, train_part, test_part)

elif validation_flag == 3:
    log_file = '../log_dirs/logs_{}_division_classlabel'.format(out_pre)
    save_name = '../../models/merge-sview-into-sat/{}_division_classlabel'.format(out_pre)


if TRAIN or RTRAIN:
    #
    iter_per_epoch = np.int(len(DS.train_part) / np.float(batch_size))

    if RTRAIN:
        print('not yet implemented...')
        sys.exit()

    loss_min = 999999
    # compute the MAE error before training...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on: ', device)

    print('=== Running pretraining using sview only to train the network for {} iterations...'.format(pre_train_iter))
    batch_valid = DSsview.get_balanced_validation_batch(batch_size * 10)
    running_loss = 0.0
    for step in range(pre_train_iter):
        batch = DSsview.get_balanced_train_batch(batch_size * 2)
        xsvi1 = torch.from_numpy(batch[0]).float().to(device)
        xsvi2 = torch.from_numpy(batch[1]).float().to(device)
        xsvi3 = torch.from_numpy(batch[2]).float().to(device)
        xsvi4 = torch.from_numpy(batch[3]).float().to(device)
        labels = torch.from_numpy(batch[4]).float().to(device)
        preds = net(xsvi1, xsvi2, xsvi3, xsvi4)[0]
        loss = -1.0*torch.mean(torch.sum(labels * torch.log(preds + 1e-8), 1))
        loss.backward()
        optimizer_sview.step()
        optimizer_sview.zero_grad()
        running_loss += loss
        if step % 200 == 199:
            xsvi1 = torch.from_numpy(batch_valid[0]).float().to(device)
            xsvi2 = torch.from_numpy(batch_valid[1]).float().to(device)
            xsvi3 = torch.from_numpy(batch_valid[2]).float().to(device)
            xsvi4 = torch.from_numpy(batch_valid[3]).float().to(device)
            labels = torch.from_numpy(batch_valid[4]).float().to(device)
            preds = net(xsvi1, xsvi2, xsvi3, xsvi4)[0]
            loss = -1.0*torch.mean(torch.sum(labels * torch.log(preds + 1e-8), 1))
            running_loss = 0.0
            print('Pretraining - validation loss: {} at iteration {}'.format(loss, step+1))
    
    print('=== Running training using sview and satellite...')
    mae_error = 0
    run_num = 1
    runi = 0
    cxloss = 0
    while run_num > 0:
        for i, valid_data in enumerate(DS.validation_iterator(psat_size=1000)): 
            x_sat = torch.from_numpy(valid_data[4].transpose(0,3,1,2)).float().to(device)
            xsvi1 = torch.from_numpy(valid_data[0]).float().to(device)
            xsvi2 = torch.from_numpy(valid_data[1]).float().to(device)
            xsvi3 = torch.from_numpy(valid_data[2]).float().to(device)
            xsvi4 = torch.from_numpy(valid_data[3]).float().to(device)
            svpred = net(xsvi1, xsvi2, xsvi3, xsvi4)[1]
            xs = integrate_sview(x_sat, svpred, 1000)
            preds_ = UNet(xs)[0].to('cpu').detach().numpy()
            mae_error += np.sum(np.abs( np.argmax( preds_, axis=1 ) - np.argmax( valid_data[5], axis=3 ) ) * np.max(valid_data[5], axis=3)) / np.sum(valid_data[5])
            cxloss += np.mean(-np.sum(valid_data[5].transpose(0,3,1,2) * np.log(preds_ + 1e-8), 1))
            runi += 1
        run_num = run_num - 1
    mae_error = mae_error / np.float(runi)
    cxloss = cxloss / np.float(runi)
    print('>>> MAE in validation before training: {}'.format(mae_error))
    print('>>> Xentropy in validation before training: {}'.format(cxloss))
    
    loss_min = mae_error
    mae_error_list = [mae_error]
    cxloss_list = [cxloss]
    running_loss = 0.0
    for epoch in range(0, num_epochs): 
        optimizer.zero_grad()
        optimizer_sat.zero_grad()
        for step in range(0, 500): 
            batch = DS.get_balanced_train_batch(1)
            x_sat = torch.from_numpy(batch[4].transpose(0,3,1,2)).float().to(device)
            xsvi1 = torch.from_numpy(batch[0]).float().to(device)
            xsvi2 = torch.from_numpy(batch[1]).float().to(device)
            xsvi3 = torch.from_numpy(batch[2]).float().to(device)
            xsvi4 = torch.from_numpy(batch[3]).float().to(device)
            svpred = net(xsvi1, xsvi2, xsvi3, xsvi4)[1]
            xs = integrate_sview(x_sat, svpred, sat_patch)
            labels = torch.from_numpy(batch[5].transpose(0,3,1,2)).float().to(device)
            
            preds, h5 = UNet(xs)
            loss = -1.0*torch.mean(torch.sum(labels * torch.log(preds + 1e-8), 1))/batch_size
            loss.backward()
            
            # taking a gradient step every batch_size iterations.
            if step % batch_size == batch_size-1: 
                if finetune_flag:
                    optimizer.step()
                    optimizer.zero_grad()
                else: 
                    optimizer_sat.step()
                    optimizer_sat.zero_grad()


            running_loss += loss.to('cpu').item()*batch_size
            if step % 50 == 49: 
                print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 50))
                running_loss = 0.0
                
            
        mae_error = 0
        cxloss = 0
        run_num = 1
        runi = 0
        while run_num > 0:
            for i, valid_data in enumerate(DS.validation_iterator(psat_size=1000)): 
                x_sat = torch.from_numpy(valid_data[4].transpose(0,3,1,2)).float().to(device)
                xsvi1 = torch.from_numpy(valid_data[0]).float().to(device)
                xsvi2 = torch.from_numpy(valid_data[1]).float().to(device)
                xsvi3 = torch.from_numpy(valid_data[2]).float().to(device)
                xsvi4 = torch.from_numpy(valid_data[3]).float().to(device)
                svpred = net(xsvi1, xsvi2, xsvi3, xsvi4)[1]
                xs = integrate_sview(x_sat, svpred, 1000)
                preds_ = UNet(xs)[0].to('cpu').detach().numpy()
                mae_error += np.sum(np.abs( np.argmax( preds_, axis=1 ) - np.argmax( valid_data[5], axis=3 ) ) * np.max(valid_data[5], axis=3)) / np.sum(valid_data[5])
                cxloss += np.mean(-np.sum(valid_data[5].transpose(0,3,1,2) * np.log(preds_ + 1e-8), 1))
                runi += 1
            run_num = run_num - 1
        mae_error = mae_error / np.float(runi)
        cxloss = cxloss / np.float(runi)
        print('>>> MAE in validation after epoch {} : {}'.format(epoch + 1, mae_error))
        print('>>> Xentropy in validation after epoch {} : {}'.format(epoch + 1, cxloss))
        mae_error_list.append(mae_error)
        cxloss_list.append(cxloss)
        # writing if the current iteration reaches a lower validation error: as average of the last 5, to avoid lucky drops.
        if epoch > 10: 
            if np.mean(mae_error_list[-5:]) < loss_min:
                loss_min = np.mean(mae_error_list[-5:])
                print('*** Got a new minimum validation loss: {} ***', loss_min)
                torch.save(net.state_dict(), save_name + '_sview_network.pth')
                torch.save(UNet.state_dict(), save_name + '_unet.pth')
        if epoch % 100 == 99: 
            # writing intermediate validation results.
            mae_error_name = '../../analysis/merge-sview-into-sat/{}_mae_error_list.txt'.format(out_pre)
            np.savetxt(mae_error_name, mae_error_list)
            xent_error_name = '../../analysis/merge-sview-into-sat/{}_xentropy_error_list.txt'.format(out_pre)
            np.savetxt(xent_error_name, cxloss_list)


    # writing down the mae error in the validation set over the epochs
    mae_error_name = '../../analysis/merge-sview-into-sat/{}_mae_error_list.txt'.format(out_pre)
    np.savetxt(mae_error_name, mae_error_list)
    xent_error_name = '../../analysis/merge-sview-into-sat/{}_xentropy_error_list.txt'.format(out_pre)
    np.savetxt(xent_error_name, cxloss_list)
else: # meaning we are testing the algorithms
    print('Loading: ', save_name + '_unet.pth', save_name + '_sview_network.pth')
    UNet.load_state_dict(torch.load(save_name + '_unet.pth'))
    UNet.eval()
    net.load_state_dict(torch.load(save_name + '_sview_network.pth'))
    net.eval()
    # compute the MAE error before training...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print('Restored network... running through the test set')
    # == assigning the file names == #
    if validation_flag == 0:
        fname = '../../analysis/merge-sview-into-sat/{}_{}_out_of_{}_folds'.format(out_pre,part_kp, part_kn)
        fname_h5 = fname + '_h5_vals'
        fname_pred = fname + '_predictions'
        fname_image = fname + '_image'
        fname_lsoa = fname + '_lsoa'
        fname_label = fname + '_label'
    elif validation_flag == 1:
        fname = '../../analysis/merge-sview-into-sat/{}_division_tr{}_vl{}_te{}'.format(out_pre, train_part, validation_part, test_part)
        fname_h5 = fname + '_h5_vals'
        fname_pred = fname + '_predictions'
        fname_image = fname + '_image'
        fname_lsoa = fname + '_lsoa'
        fname_label = fname + '_label'
    elif validation_flag == 2:
        fname_h5 = '../../analysis/merge-sview-into-sat/{}_division_tr{}_te{}_h5_vals'.format(out_pre, train_part, test_part)
        fname_pred = '../../analysis/merge-sview-into-sat/{}_division_tr{}_te{}_predictions'.format(out_pre, train_part, test_part)
    elif validation_flag == 3:
        fname_h5 = '../../analysis/merge-sview-into-sat/{}_division_classlabel_h5_vals'.format(out_pre)
        fname_pred = '../../analysis/merge-sview-into-sat/{}_division_classlabel_predictions'.format(out_pre)

    def aggregate_mae(labels, preds, h5s, lsoas): 
        lsoa_list = np.unique(lsoas.flatten())
        if any(lsoa_list == 65535):
            lsoa_list = lsoa_list[:-1]
        mae = 0.0
        preds_lsoa_list = []
        label_lsoa_list = []
        h5_lsoa_list = []
        lsoa_list_to_write = []
        for l in lsoa_list: 
            predsl = preds[lsoas == l].mean()
            labelsl = labels[lsoas == l].mean()
            h5sl = h5s[lsoas == l].mean()
            lsoal = np.ones_like(predsl)*l
            preds_lsoa_list.append(predsl)
            label_lsoa_list.append(labelsl)
            h5_lsoa_list.append(h5sl)
            lsoa_list_to_write.append(lsoal)
            mae += np.abs(predsl - labelsl)
        return mae / lsoa_list.size, \
                preds_lsoa_list, \
                h5_lsoa_list, \
                label_lsoa_list, \
                lsoa_list_to_write

    # == read and predicting for test data == #
    mae = 0
    mae_lsoa = 0
    preds_lsoa_list = []
    label_lsoa_list = []
    h5_lsoa_list = []
    lsoa_list = []
    total_test_num = len(DS.test_part)
    for i, test_data_ in enumerate(DS.test_iterator(psat_size=1000)):
        test_data = test_data_[0]
        lsoa_info = test_data_[1]
        x_sat = torch.from_numpy(test_data[4].transpose(0,3,1,2)).float().to(device)
        xsvi1 = torch.from_numpy(test_data[0]).float().to(device)
        xsvi2 = torch.from_numpy(test_data[1]).float().to(device)
        xsvi3 = torch.from_numpy(test_data[2]).float().to(device)
        xsvi4 = torch.from_numpy(test_data[3]).float().to(device)
        svpred = net(xsvi1, xsvi2, xsvi3, xsvi4)[1]
        xs = integrate_sview(x_sat, svpred, 1000)
        preds, h5 = UNet(xs)
        preds_ = preds.to('cpu').detach().numpy()
        h5_ = h5.to('cpu').detach().numpy()[0,...]
        mae += np.sum(np.abs( np.argmax( preds_, axis=1 ) - np.argmax( test_data[5], axis=3 ) ) * np.max(test_data[5], axis=3)) / np.sum(test_data[5])        
        mae_lsoa_, pred_list, h5_list, label_list, lsoa_list_to_write = aggregate_mae(np.argmax(test_data[5], axis=3), np.argmax(preds_, axis=1), h5_, lsoa_info)
        mae_lsoa += mae_lsoa_
        preds_lsoa_list.append(pred_list)
        label_lsoa_list.append(label_list)
        h5_lsoa_list.append(h5_list)
        lsoa_list.append(lsoa_list_to_write)
        tiff_file = fname_h5 + '_{}.tif'.format(i)
        tiff.imsave(tiff_file, h5_[0,...].astype(np.float))
        tiff_file = fname_pred + '_{}.tif'.format(i)
        tiff.imsave(tiff_file, np.argmax(preds_, axis=1)[0,...].astype(np.float))
        tiff_file = fname_label +'_{}.tif'.format(i)
        tiff.imsave(tiff_file, np.argmax(test_data[5], axis=3)[0,...].astype(np.float))
        tiff_file = fname_image + '_{}.tif'.format(i)
        img_ = test_data[4][0,:,:,1:4]
        m = img_.min()
        M = img_.max()
        tiff.imsave(tiff_file, ((img_ - m)/(M-m+1e-8) * 255).astype(np.uint8))
        tiff_file = fname_lsoa + '_{}.tif'.format(i)
        tiff.imsave(tiff_file, lsoa_info.astype(np.float))
        
    print('Test data MAE: {}'.format(mae / np.float(total_test_num)))
    print('Test data MAE - LSOA aggregate: {}'.format(mae_lsoa / np.float(total_test_num)))
    preds_lsoa_name = '{}_preds_lsoa_list.txt'.format(fname)    
    np.savetxt(preds_lsoa_name, np.concatenate(preds_lsoa_list))
    label_lsoa_name = '{}_label_lsoa_list.txt'.format(fname)
    np.savetxt(label_lsoa_name, np.concatenate(label_lsoa_list))
    h5_lsoa_name = '{}_h5_lsoa_list.txt'.format(fname)
    np.savetxt(h5_lsoa_name, np.concatenate(h5_lsoa_list))
    lsoa_name = '{}_lsoa_list.txt'.format(fname)
    np.savetxt(lsoa_name, np.concatenate(lsoa_list).astype(np.int), fmt='%i')
    

