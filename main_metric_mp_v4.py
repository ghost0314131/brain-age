import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from torch.utils.data.distributed import DistributedSampler
import os, shutil, argparse, torchsnooper, random
from utils import Logger, AverageMeter, aug_crop_center, aug_flip, aug_shift,\
    num2vect, KLDivLossFunc, MAELossFunc, save_checkpoint
import models as customized_models

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
input_img_path = '../T1_mwp1/{}_mwp1T1.npy' # path of preprocessed image data
input_info_path = 'MCAD.csv' # path of age data

# Transforming the age to soft label (probability distribution)
bin_range = [18, 90]
bin_step = 2
sigma = 1 
bin_start = bin_range[0]
bin_end = bin_range[1]
bin_length = bin_end - bin_start
bin_number = int(bin_length / bin_step)
bc = bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)


class Parser():
    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        self.parser.add_argument(
            '--batch_size', type=int, default=16,
            help='batch size for training and validation (default: 16)')
        self.parser.add_argument(
            '--num_workers', type=int, default=8,
            help='num of workers of DataLoader (default: 8)')

        # learning
        self.parser.add_argument(
            '--seed', type=int, default=0,
            help='random seed (default: 0)')
        self.parser.add_argument(
            '--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
        self.parser.add_argument(
            '--epochs', type=int, default=400,
            help='number of epochs to train (default: 400)')
        self.parser.add_argument(
            '--optimizer', type=str, default='AdamW',
            help='optimizer (default:AdamW)')
        self.parser.add_argument(
            '--lr', type=float, default=3e-4,
            help='learning rate (default: 3e-4), for SGD recommend 0.01')
        self.parser.add_argument(
            '--dropout', type=float, default=0.5,
            help='dropout probability(default: 0.5)')

        # checkpoint
        self.parser.add_argument('-n1', '--name_train', default='All_train_s1', type=str)
        self.parser.add_argument('-n2', '--name_valid', default='All_valid_s1', type=str)
        self.parser.add_argument('-add', '--address', default='tcp://127.0.0.1:2340', type=str)
        self.parser.add_argument('-c', '--checkpoint', default='ckp', type=str, metavar='PATH',
            help='path to save checkpoint (default: checkpoint)')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')  # checkpoint.pth.tar      model_best.pth.tar
        self.parser.add_argument(
            '--valid', action='store_true', default=False,
            help='validation mode')
        self.parser.add_argument(
            '--export', action='store_true', default=False,
            help='export predicted prob of validation dataset')

        # done
        self.args = self.parser.parse_args()


class Dataset_twoway(Dataset):
    '''
    Dataset class for pair-wise training
    '''
    def __init__(self, name, augmented=False):
        self.name = name
        self.augmented = augmented
        Info = pd.read_csv(input_info_path.format(self.name), sep=',', header=None)
        self.subj_id = Info.iloc[:, 0].values
        self.age = Info.iloc[:, 1].values
        self.pairs = int(len(self.age)*(len(self.age)-1)/2)
        self.lookup = self.tableConstruct()
        assert 2 < len(self.age)

    def get_each(self, idx):
        assert idx < len(self.age)
        img_path = input_img_path.format(int(self.subj_id[idx]))
        nii_img = np.nan_to_num(np.load(img_path))
        if self.augmented:
            if np.random.uniform(0, 1) > 0.75:
                nii_img = aug_flip(nii_img, 0)
            if np.random.uniform(0, 1) > 0.5:
                nii_img = aug_shift(nii_img, np.random.randint(0, 3), np.random.randint(0, 3))
        nii_img = aug_crop_center(nii_img, (110, 128, 110))
        sp = tuple([1]) + nii_img.shape
        nii_img = nii_img.reshape(sp)
        nii_img = torch.tensor(nii_img, dtype=torch.float32)
        y_prob, _ = num2vect(np.array([self.age[idx], ]), bin_range, bin_step, sigma)
        y_prob = torch.tensor(y_prob, dtype=torch.float32)
        return nii_img, y_prob, self.age[idx]

    def tableConstruct(self):
        sampleNum = len(self.age)
        lookupTable = np.full((self.pairs, 2), -1)
        first = 0
        firstCount = 0
        firstTop = sampleNum - 1
        second = 1
        secondDown = 1
        for i in range(self.pairs):
            lookupTable[i] = first, second
            firstCount += 1
            if firstCount >= firstTop:
                firstTop -= 1
                first += 1
                firstCount = 0
            second += 1
            if second >= sampleNum:
                secondDown += 1
                second = secondDown
        return lookupTable

    def __getitem__(self, idx):
        assert idx < self.pairs
        idx = int(random.random()*self.pairs)  # limit
        id1, id2 = self.lookup[idx]
        in1, tar1, lab1 = self.get_each(id1)
        in2, tar2, lab2 = self.get_each(id2)
        return in1, tar1, lab1, in2, tar2, lab2

    def __len__(self):
        #return self.pairs
        return int(len(self.age)/2)  # limit


class Dataset_oneway(Dataset):
    '''
    Dataset class for validating and testing (for just one sample)
    '''
    def __init__(self, name, augmented=False):
        self.name = name
        self.augmented = augmented
        Info = pd.read_csv(input_info_path.format(self.name), sep=',', header=None)
        self.subj_id = Info.iloc[:, 0].values
        self.age = Info.iloc[:, 1].values

    def __getitem__(self, idx):
        assert idx < len(self.age)
        img_path = input_img_path.format(int(self.subj_id[idx]))
        nii_img = np.nan_to_num(np.load(img_path))
        if self.augmented:
            if np.random.uniform(0, 1) > 0.75:
                nii_img = aug_flip(nii_img, 0)
            if np.random.uniform(0, 1) > 0.5:
                nii_img = aug_shift(nii_img, np.random.randint(0, 3), np.random.randint(0, 3))
        nii_img = aug_crop_center(nii_img, (110, 128, 110))
        sp = tuple([1]) + nii_img.shape
        nii_img = nii_img.reshape(sp)
        nii_img = torch.tensor(nii_img, dtype=torch.float32)
        y_prob, _ = num2vect(np.array([self.age[idx], ]), bin_range, bin_step, sigma)
        y_prob = torch.tensor(y_prob, dtype=torch.float32)
        return nii_img, y_prob, self.age[idx]

    def __len__(self):
        return len(self.age)


class DistLoss(torch.nn.Module):
    '''
    Contrastive loss function
    '''
    def __init__(self, margin=5.0):
        super(DistLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 2

        x0_type, x1_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2

    def forward(self, x0, x1, y0, y1):
        self.check_type_forward((x0, x1))
        y = y0 == y1
        y = y.int()
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2) * torch.abs(y0 - y1)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

# @torchsnooper.snoop()
def main(local_rank, nprocs, args):
    '''
    main program, parameters are set for multi-gpu processing
    '''
    # multi-gpu
    dist.init_process_group(backend='nccl', init_method=args.address, world_size=args.nprocs, rank=local_rank)
    torch.cuda.set_device(local_rank)
    args.device = local_rank

    best_mae = 10000.0
    start_epoch = args.start_epoch
    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    # define model structure
    model = customized_models.MetricNet(num_classes=bin_number,
                                          num_blocks=[2, 2, 2, 2],
                                          width_multiplier=[1, 1, 0.5, 0.125],
                                          dropout=args.dropout)
    model.to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    args.batch_size = int(args.batch_size / args.nprocs)

    if args.valid:
        dataset = Dataset_oneway(args.name_valid)
        validloader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    else:
        train_dataset = Dataset_twoway(args.name_train, augmented=True)
        valid_dataset = Dataset_oneway(args.name_valid)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        trainloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        validloader = DataLoader(
            valid_dataset, sampler=valid_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    criterion = KLDivLossFunc().to(local_rank)   # default reduce is true
    measure = MAELossFunc().to(local_rank)
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
    else:
        raise NotImplementedError()

    # Resume from interrupted training
    title = 'Design'
    if args.resume:
        # Load checkpoint.
        args.resume = './'+args.checkpoint+'/'+ args.resume
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        best_mae = checkpoint['best_mae']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        with torch.no_grad():
            valid_loss, valid_mae, valid_true, valid_pred = test_batch(
                args, model, validloader, criterion, measure, bc)
        if args.export:
            np.savez(args.resume + '_output.npz', y_true=valid_true, y_pred=valid_pred)
        if args.device == 0:
            print(' Resume valid Loss:  %.8f, Valid MAE:  %.4f' % (valid_loss, valid_mae))
        if args.valid:
            return
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Loss.Tr', 'Loss.Va', '  MAE.Tr', '  MAE.Va'])

    # main training process
    for epoch in range(start_epoch, args.epochs):
        if args.device == 0:
            print("Epoch %d Starts:" % epoch)
        # set meta parmeters
        lr = optimizer.param_groups[0]['lr']
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        # training phase
        train_loss, train_mae, train_true, train_pred = train_batch(
            args, model, trainloader, criterion, measure, optimizer, bc)
        if args.optimizer == 'SGD':
            scheduler.step()

        with torch.no_grad():
            # validation loss
            valid_loss, valid_mae, valid_true, valid_pred = test_batch(
                args, model, validloader, criterion, measure, bc)

        # print
        if args.device == 0:
            print(' Train Loss:  %.8f, Train MAE:  %.4f' % (train_loss, train_mae))
            print(' Valid Loss:  %.8f, Valid MAE:  %.4f' % (valid_loss, valid_mae))

        # save logs
        valid_mae = valid_mae.cpu()
        is_best = valid_mae < best_mae
        best_mae = min(valid_mae, best_mae)
        if args.device == 0:
            logger.append([lr, train_loss, valid_loss, train_mae, valid_mae])
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_mae,
                'best_mae': best_mae,
                # 'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint, filename='ckp_{}_{:.2f}.pth.tar'.format(epoch+1, valid_mae))
    logger.close()
    print('Best mae:')
    print(best_mae)


def train_batch(args, model, train_loader, criterion, measure, optimizer, bc):
    # switch to train mode
    model.train()
    losses = AverageMeter()
    MAE = AverageMeter()
    y_true = []
    y_pred = []
    for batch_idx, (in1, tar1, lab1, in2, tar2, lab2) in enumerate(train_loader):

        in1, in2, tar1, tar2, lab1, lab2 = in1.to(args.device), in2.to(args.device), tar1.to(args.device), tar2.to(args.device), lab1.to(args.device), lab2.to(args.device)

        out1, out2, vec1, vec2 = model(in1, in2)

        loss_pair = criterion(out1, tar1.reshape(out1.shape)) + criterion(out2, tar2.reshape(out2.shape))
        loss_dist = DistLoss()(vec1, vec2, lab1, lab2)
        loss = loss_pair + loss_dist
        # compute gradient and do BP step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prob1 = torch.exp(out1.squeeze().cpu())
            prob2 = torch.exp(out2.squeeze().cpu())
            preds1 = prob1 @ bc 
            preds2 = prob2 @ bc
            preds1, lab1 = preds1.to(args.device), lab1.to(args.device)  
            preds2, lab2 = preds2.to(args.device), lab2.to(args.device)
            acc1 = measure(preds1, lab1)
            acc2 = measure(preds2, lab2)
            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss_pair, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc2 = reduce_mean(acc2, args.nprocs)

            # measure accuracy and record loss
            losses.update(reduced_loss.item(), 2*len(lab1))
            MAE.update(reduced_acc1, len(lab1))
            MAE.update(reduced_acc2, len(lab2))

            # compose into list
            preds1, preds2 = preds1.cpu().numpy().tolist(), preds2.cpu().numpy().tolist()
            lab1, lab2 = lab1.cpu().numpy().tolist(), lab2.cpu().numpy().tolist()
            if isinstance(lab1, list):
                y_true.extend(lab1)
            else:
                y_true.extend([lab1])
            if isinstance(preds1, list):
                y_pred.extend(preds1)
            else:
                y_pred.extend([preds1])

            if isinstance(lab2, list):
                y_true.extend(lab2)
            else:
                y_true.extend([lab2])
            if isinstance(preds2, list):
                y_pred.extend(preds2)
            else:
                y_pred.extend([preds2])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return losses.avg/2, MAE.avg, y_true.reshape(-1, 1), y_pred.reshape(-1, 1)


def test_batch(args, model, val_loader, criterion, measure, bc):
    # switch to evaluate mode
    model.eval()
    losses = AverageMeter()
    MAE = AverageMeter()
    y_true = []
    y_pred = []
    for batch_idx, (in1, tar1, lab1) in enumerate(val_loader):

        in1, tar1, lab1 = in1.to(args.device), tar1.to(args.device), lab1.to(args.device)
        out1, _, _, _ = model(in1, in1)

        loss = criterion(out1, tar1.reshape(out1.shape))

        prob1 = torch.exp(out1.squeeze().cpu())
        preds1 = prob1 @ bc
        
        preds1, lab1 = preds1.to(args.device), lab1.to(args.device)
        acc1 = measure(preds1, lab1)

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)

        # measure accuracy and record loss
        losses.update(reduced_loss.item(), len(lab1))
        MAE.update(reduced_acc1, len(lab1))
        # compose into list
        lab1 = lab1.cpu().numpy().tolist()
        preds1 = preds1.cpu().numpy().tolist()
        if isinstance(lab1, list):
            y_true.extend(lab1)
        else:
            y_true.extend([lab1])
        if isinstance(preds1, list):
            y_pred.extend(preds1)
        else:
            y_pred.extend([preds1])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if args.export:
       np.savez(args.resume+'_output.npz', y_true=y_true, y_pred=y_pred)
    return losses.avg, MAE.avg, y_true.reshape(-1, 1), y_pred.reshape(-1, 1)


def reduce_mean(tensor, nprocs):
    '''
    multi-gpu synchronize
    '''
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


if __name__ == '__main__':
    args = Parser(description='Pytorch Age Prediction').args
    args.nprocs = torch.cuda.device_count()
    print('show all arguments configuration...')
    print(args)
    if not os.path.exists('./' + args.checkpoint):
        os.system('mkdir ./' + args.checkpoint)
    mp.spawn(main, nprocs=args.nprocs, args=(args.nprocs, args))
    print("Done")
