import os
import argparse
from glob import glob
import numpy as np
import torch.utils.data as data
from model import RetinexNet, Mymodel
from torchvision import transforms
from dataset import Low_Light_Dataset
from trainer import Base_Trainer
import matplotlib.pyplot as plt


def main(epochs, batch_size, patch_size, lr, data_dir, ckpt_dir, gpu_id, vis_dir):

    phase_name = ['Decom', 'Relight']
    Decom_epoch = epochs * 2
    Relight_epoch = epochs
    phase_epoch = [Decom_epoch, Relight_epoch]
    lr = lr * np.ones([Decom_epoch])
    val_every_epoch = 20
    # lr[20:] = lr[0] / 10.0

    train_data_path = os.path.join(data_dir, 'train')
    valid_data_path = os.path.join(data_dir, 'val')

    train_low_data_names = glob(train_data_path + '/low/*.png')
                           # glob(data_dir + '/train/low/*.png')
    train_low_data_names.sort()
    train_high_data_names = glob(train_data_path + '/high/*.png')
                           # glob(data_dir + '/our485/high/*.png')
    train_high_data_names.sort()
    eval_low_data_names  = glob(valid_data_path + '/low/*.*')
    eval_low_data_names.sort()
    eval_high_data_names = glob(valid_data_path + '/high/*.*')
    eval_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    assert len(train_low_data_names) != 0

    transform_low = transforms.Compose(
        [
            transforms.RandomCrop(patch_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    transform_high = transforms.Compose(
        [
            transforms.RandomCrop(patch_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train_dataset = Low_Light_Dataset(train_data_path, transform_low, transform_high)
    valid_dataset = Low_Light_Dataset(valid_data_path, transform_val, transform_val)

    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    print('Number of train data: %d,  Batch of train data: %d' % (len(train_dataset), len(train_dataloader)))

    valid_dataloader = data.DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=4)
    print('Number of valid data: %d,  Batch of valid data: %d' % (len(valid_dataset), len(valid_dataloader)))

    '''for id, item in enumerate(train_dataloader):
        low = item[0]
        high = item[1]
        print(low.shape, high.shape)
        low = low[0].permute(1, 2, 0)
        high = high[0].permute(1, 2, 0)
        print(low.shape, high.shape)
        plt.subplot(2, 1, 1)
        plt.imshow(low)
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(high)
        plt.axis('off')
        plt.show()
        print('1')'''

    model = Mymodel(gpu_id)
    model.summary()

    model_trainer = Base_Trainer(model, ckpt_dir,
                                 train_dataloader, valid_dataloader,
                                 lr, val_every_epoch, gpu_id,
                                 phase_name, phase_epoch, vis_dir)

    model_trainer.valid('Decom', 1)
    # model_trainer.train()

    '''train_low_data_names,
                train_high_data_names,
                eval_low_data_names,
                eval_high_data_names,
                batch_size=batch_size,
                patch_size=patch_size,
                epoch=Decom_epoch,
                lr=lr,
                # vis_dir=vis_dir,
                ckpt_dir=ckpt_dir,
                eval_every_epoch=20,
                train_phase="Decom")

    model_trainer.train(train_low_data_names,
                train_high_data_names,
                eval_low_data_names,
                eval_high_data_names,
                batch_size=batch_size,
                patch_size=patch_size,
                epoch=Relight_epoch,
                lr=lr,
                # vis_dir=vis_dir,
                ckpt_dir=ckpt_dir,
                eval_every_epoch=20,
                train_phase="Relight")'''


if __name__ == '__main__':
    # TODO logger

    parser = argparse.ArgumentParser(description='Learning Low Light Image Enhancement')

    parser.add_argument('--gpu_id', dest='gpu_id', default="6",
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100,
                        help='number of total epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16,
                        help='number of samples in one batch')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=48,
                        help='patch size')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--data_dir', dest='data_dir',
                        default='./data1/',
                        help='directory storing the training data')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpts/data1/',
                        help='directory for checkpoints')

    args = parser.parse_args()

    if args.gpu_id != "-1":
        # Create directories for saving the checkpoints and visuals
        args.vis_dir = args.ckpt_dir + '/visuals/'
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.vis_dir):
            os.makedirs(args.vis_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Create the model
        # model = RetinexNet(args.ckpt_dir).cuda()
        # Train the model
        main(args.epochs, args.batch_size, args.patch_size, args.lr, args.data_dir, args.ckpt_dir, args.gpu_id, args.vis_dir)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
