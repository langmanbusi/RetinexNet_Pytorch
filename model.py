import os
import time
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from utils import MSE, SSIM, PSNR, LPIPS
from torch.utils.tensorboard import SummaryWriter
import atexit
import logging


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:3, :, :])
        L        = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu         = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size,
                                      padding=1, padding_mode='replicate')

        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')

        self.net2_deconv1_1= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_2= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_3= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1,
                                     padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0      = self.net2_conv0_1(input_img)
        out1      = self.relu(self.net2_conv1_1(out0))
        out2      = self.relu(self.net2_conv1_2(out1))
        out3      = self.relu(self.net2_conv1_3(out2))

        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1   = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2   = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3   = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        deconv1_rs= F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs= F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output    = self.net2_output(feats_fus)
        return output


class RetinexNet(nn.Module):
    def __init__(self, ckpt_dir=None):
        super(RetinexNet, self).__init__()

        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()
        if ckpt_dir:
            tsbd_dir = os.path.join(ckpt_dir, 'tensorboard')
            self.writer = SummaryWriter(log_dir=tsbd_dir)
            atexit.register(self.cleanup)

    def cleanup(self):
        self.writer.close()

    def forward(self, input_low, input_high):
        # Forward DecompNet
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda()
        input_high= Variable(torch.FloatTensor(torch.from_numpy(input_high))).cuda()
        R_low, I_low   = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)

        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low)

        # Other variables
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)

        return R_low, I_low, R_high, I_high, I_delta, I_low_3, I_high_3, I_delta_3

        # Compute losses
        self.recon_loss_low  = F.l1_loss(R_low * I_low_3,  input_low)
        self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)
        self.recon_loss_mutal_low  = F.l1_loss(R_high * I_low_3, input_low)
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)
        self.equal_R_loss = F.l1_loss(R_low,  R_high.detach())
        self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)

        self.Ismooth_loss_low   = self.smooth(I_low, R_low)
        self.Ismooth_loss_high  = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        self.loss_Decom = self.recon_loss_low + \
                          self.recon_loss_high + \
                          0.001 * self.recon_loss_mutal_low + \
                          0.001 * self.recon_loss_mutal_high + \
                          0.1 * self.Ismooth_loss_low + \
                          0.1 * self.Ismooth_loss_high + \
                          0.01 * self.equal_R_loss
        self.loss_Relight = self.relight_loss + \
                            3 * self.Ismooth_loss_delta

        self.output_R_low   = R_low.detach().cpu()
        self.output_I_low   = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S       = self.output_R_low * self.output_I_delta
        self.input_high = input_high.detach().cpu()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, eval_high_data_names, vis_dir, train_phase, save_eval=False):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        MSE_output = []
        SSIM_output = []
        PSNR_output = []
        LPIPS_output = []
        N = len(eval_low_data_names)

        for idx in tqdm(range(len(eval_low_data_names))):
            eval_low_img   = Image.open(eval_low_data_names[idx])
            eval_high_img = Image.open(eval_high_data_names[idx])

            eval_low_img   = np.array(eval_low_img, dtype="float32")/255.0
            eval_low_img   = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            eval_high_img = np.array(eval_high_img, dtype="float32") / 255.0
            eval_high_img = np.transpose(eval_high_img, (2, 0, 1))
            input_high_eval = np.expand_dims(eval_high_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_high_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                if save_eval:
                    input    = np.squeeze(input_low_eval)
                    result_1 = np.squeeze(result_1)
                    result_2 = np.squeeze(result_2)
                    cat_image= np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == "Relight":
                self.forward(input_low_eval, input_high_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                result_GT = self.input_high
                if save_eval:
                    input = np.squeeze(input_low_eval)
                    result_1 = np.squeeze(result_1)
                    result_2 = np.squeeze(result_2)
                    result_3 = np.squeeze(result_3)
                    result_4 = np.squeeze(result_4)
                    cat_image= np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
                # TODO 增加评价指标，不保存图片

                # metric
                MSE_output += [MSE(result_4.numpy(), result_GT.numpy())]
                SSIM_output += [SSIM(result_4.numpy(), result_GT.numpy())]
                PSNR_output += [PSNR(result_4.numpy(), result_GT.numpy())]
                # LPIPS_output += [LPIPS(result_4, result_GT)]

            if save_eval:
                cat_image = np.transpose(cat_image, (1, 2, 0))
                # print(cat_image.shape)
                im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
                filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
                           (train_phase, idx + 1, epoch_num))
                im.save(filepath[:-4] + '.jpg')

        if self.train_phase == 'Relight':
            # metric
            MSE_avg = sum(MSE_output) / N * 255 * 255 / 1000
            SSIM_avg = sum(SSIM_output) / N
            PSNR_avg = sum(PSNR_output) / N
            LPIPS_avg = sum(LPIPS_output) / N

            self.writer.add_scalar(self.train_phase + '/MSE', MSE_avg, epoch_num)
            self.writer.add_scalar(self.train_phase + '/SSIM', SSIM_avg, epoch_num)
            self.writer.add_scalar(self.train_phase + '/PSNR', PSNR_avg, epoch_num)
            self.writer.add_scalar(self.train_phase + '/LPIPS', LPIPS_avg, epoch_num)       # TODO 改文件夹

            print('MSE = ', MSE_avg,
                  'SSIM = ', SSIM_avg,
                  'PSNR = ', PSNR_avg,
                  'LPIPS = ', LPIPS_avg)

    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name= save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(), save_name)

    def load(self, ckpt_dir):
        load_dir = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts) > 0:
                load_ckpt = load_ckpts[-1]
                global_step = int(load_ckpt[:-4])
                ckpt_dict = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0  # TODO 异常

    def train(self,
              train_low_data_names,
              train_high_data_names,
              eval_low_data_names,
              eval_high_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)
        num_iter = numBatch * epoch
        # Create the optimizers
        self.train_phase = train_phase
        if self.train_phase == "Decom":
            self.train_op_Decom   = optim.Adam(self.DecomNet.parameters(),
                                               lr=lr[0], betas=(0.9, 0.999))
        elif self.train_phase == "Relight":
            self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                               lr=lr[0], betas=(0.9, 0.999))

        # Initialize a network if its checkpoint is available
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num    = global_step
            start_epoch = global_step // numBatch
            start_step  = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num    = 0
            start_epoch = 0
            start_step  = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id   = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            if self.train_phase == "Decom":
                for param_group in self.train_op_Decom.param_groups:
                    param_group['lr'] = self.lr
            elif self.train_phase == "Relight":
                for param_group in self.train_op_Relight.param_groups:
                    param_group['lr'] = self.lr
            with tqdm(range(start_step, numBatch)) as tepoch:
                for batch_id in tepoch:
                    tepoch.set_description(f"Train {train_phase} : Epoch {epoch}")
                    # Generate training data for a batch
                    batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                    batch_input_high= np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")

                    for patch_id in range(batch_size):
                        # Load images
                        train_low_img = Image.open(train_low_data_names[image_id])
                        train_low_img = np.array(train_low_img, dtype='float32')/255.0
                        train_high_img= Image.open(train_high_data_names[image_id])
                        train_high_img= np.array(train_high_img, dtype='float32')/255.0
                        # Take random crops
                        h, w, _        = train_low_img.shape
                        x = random.randint(0, h - patch_size)
                        y = random.randint(0, w - patch_size)
                        train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                        train_high_img= train_high_img[x: x + patch_size, y: y + patch_size, :]
                        # Data augmentation
                        if random.random() < 0.5:
                            train_low_img = np.flipud(train_low_img)
                            train_high_img= np.flipud(train_high_img)
                        if random.random() < 0.5:
                            train_low_img = np.fliplr(train_low_img)
                            train_high_img= np.fliplr(train_high_img)
                        rot_type = random.randint(1, 4)
                        if random.random() < 0.5:
                            train_low_img = np.rot90(train_low_img, rot_type)
                            train_high_img= np.rot90(train_high_img, rot_type)
                        # Permute the images to tensor format
                        train_low_img = np.transpose(train_low_img, (2, 0, 1))
                        train_high_img= np.transpose(train_high_img, (2, 0, 1))
                        # Prepare the batch
                        batch_input_low[patch_id, :, :, :] = train_low_img
                        batch_input_high[patch_id, :, :, :]= train_high_img
                        self.input_low = batch_input_low
                        self.input_high= batch_input_high

                        image_id = (image_id + 1) % len(train_low_data_names)
                        if image_id == 0:
                            tmp = list(zip(train_low_data_names, train_high_data_names))
                            random.shuffle(list(tmp))
                            train_low_data_names, train_high_data_names = zip(*tmp)

                    # Feed-Forward to the network and obtain loss
                    self.forward(self.input_low,  self.input_high)
                    if self.train_phase == "Decom":
                        self.train_op_Decom.zero_grad()
                        self.loss_Decom.backward()
                        self.train_op_Decom.step()
                        loss = self.loss_Decom.item()
                    elif self.train_phase == "Relight":
                        self.train_op_Relight.zero_grad()
                        self.loss_Relight.backward()
                        self.train_op_Relight.step()
                        loss = self.loss_Relight.item() # TODO 增加网络块

                    iter_num += 1
                    # print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                    #       % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                    tepoch.set_postfix({'Iter': '{:d}/{:d}'.format(iter_num, num_iter),
                                        'Loss': '{:.4f}'.format(loss)})     # TODO 改成tqdm的输出

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                # self.evaluate(epoch + 1, eval_low_data_names, eval_high_data_names,
                #               vis_dir=vis_dir, train_phase=train_phase) # TODO 增加评价指标
                self.save(iter_num, ckpt_dir) # TODO 改ckpt命名格式

            self.writer.add_scalar(self.train_phase + '/Loss', loss, epoch)

        print("Finished training for phase %s." % train_phase)

    def predict(self,
                test_low_data_names,
                test_high_data_names,
                res_dir,
                ckpt_dir,
                save_predict=False):

        # Load the network with a pre-trained checkpoint
        self.train_phase= 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        self.train_phase= 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
             print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False

        # init metric
        MSE_output = []
        SSIM_output = []
        PSNR_output = []
        LPIPS_output = []
        N = len(test_low_data_names)

        # Predict for the test images
        for idx in tqdm(range(N)):
            test_low_img_path = test_low_data_names[idx]
            test_high_img_path = test_high_data_names[idx]
            # show name of result image
            test_img_name = test_low_img_path.split('/')[-1]
            # print('Processing ', test_img_name)
            # change dim
            test_low_img   = Image.open(test_low_img_path)
            test_low_img   = np.array(test_low_img, dtype="float32")/255.0
            test_low_img   = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            test_high_img = Image.open(test_high_img_path)
            test_high_img = np.array(test_high_img, dtype="float32") / 255.0
            test_high_img = np.transpose(test_high_img, (2, 0, 1))
            input_high_test = np.expand_dims(test_high_img, axis=0)

            self.forward(input_low_test, input_high_test)
            result_1 = self.output_R_low
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S
            result_GT = self.input_high

            # metric
            MSE_output += [MSE(result_4.numpy(), result_GT.numpy())]
            SSIM_output += [SSIM(result_4.numpy(), result_GT.numpy())]
            PSNR_output += [PSNR(result_4.numpy(), result_GT.numpy())]
            LPIPS_output += [LPIPS(result_4, result_GT)]

            # prepare for save
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)
            result_GT = np.squeeze(result_GT)

            if save_predict:  # and idx+1 % 10 == 0:
                if save_R_L:
                    cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
                else:
                    cat_image = result_4.numpy()
                    # cat_image = np.concatenate([input, result_4, result_GT], axis=2)

                cat_image = np.transpose(cat_image, (1, 2, 0))
                # print(cat_image.shape)
                im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
                filepath = res_dir + '/' + test_img_name
                im.save(filepath[:-4] + '.jpg')

        # metric
        MSE_avg = sum(MSE_output) / N * 255 * 255 / 1000
        SSIM_avg = sum(SSIM_output) / N
        PSNR_avg = sum(PSNR_output) / N
        LPIPS_avg = sum(LPIPS_output) / N

        print('MSE = ', MSE_avg,
              'SSIM = ', SSIM_avg,
              'PSNR = ', PSNR_avg,
              'LPIPS = ', LPIPS_avg)


class Mymodel(nn.Module):
    def __init__(self, gpu):
        super(Mymodel, self).__init__()

        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()
        if gpu is not None:
            gpu = 0
            self.gpu = torch.device('cuda:' + str(gpu))

    def forward(self, input_low, input_high):
        # Forward DecompNet
        input_low = input_low.to(self.gpu)
        input_high = input_high.to(self.gpu)
        R_low, I_low = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)

        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low)

        # Other variables
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)

        return input_low, input_high, R_low, I_low, R_high, I_high, I_delta, I_low_3, I_high_3, I_delta_3

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters: {}'.format(params))
        print(self)
