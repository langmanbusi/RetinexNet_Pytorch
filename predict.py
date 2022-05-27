import os
import argparse
from glob import glob
import numpy as np
from model import RetinexNet, Mymodel
import torch
from tqdm import tqdm
from PIL import Image
from utils import MSE, SSIM, PSNR, LPIPS


def main(data_dir, ckpt_dir, res_dir, gpu_id):

    test_low_data_names  = glob(data_dir + 'test/low/' + '*.*')
    test_high_data_names = glob(data_dir + 'test/high/' + '*.*')
    test_low_data_names.sort()
    test_high_data_names.sort()
    print('Number of evaluation images: %d' % len(test_low_data_names))

    model = Mymodel(gpu_id)
    predict(model, test_low_data_names,
                  test_high_data_names,
                  res_dir=res_dir,
                  ckpt_dir=ckpt_dir,
                  save_predict=True, gpu_id=gpu_id)


def predict(model, test_low_data_names, test_high_data_names, res_dir, ckpt_dir, save_predict=False, gpu_id = None):
    with torch.no_grad():
        if gpu_id is not None:
            gpu_id = 0
            gpu_id = torch.device('cuda:' + str(gpu_id))
            model = model.to(gpu_id)

        # Load the network with a pre-trained checkpoint
        train_phase = 'Decom'
        load_model_status, _ = load(model, ckpt_dir, train_phase)
        if load_model_status:
            print(train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        train_phase = 'Relight'
        load_model_status, _ = load(model, ckpt_dir, train_phase)
        if load_model_status:
             print(train_phase, ": Model restore success!")
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
            input_low_test = torch.from_numpy(np.expand_dims(test_low_img, axis=0))

            test_high_img = Image.open(test_high_img_path)
            test_high_img = np.array(test_high_img, dtype="float32") / 255.0
            test_high_img = np.transpose(test_high_img, (2, 0, 1))
            input_high_test = torch.from_numpy(np.expand_dims(test_high_img, axis=0))

            input_low, input_high, R_low, _, _, _, _, I_low_3, _, I_delta_3 = \
                model.forward(input_low_test, input_high_test)

            output = I_delta_3 * R_low

            # metric
            MSE_output += [MSE(output.cpu().numpy(), input_high.cpu().numpy())]
            SSIM_output += [SSIM(output.cpu().numpy(), input_high.cpu().numpy())]
            PSNR_output += [PSNR(output.cpu().numpy(), input_high.cpu().numpy())]
            LPIPS_1 = LPIPS(output.cpu(), input_high.cpu())
            LPIPS_2 = torch.squeeze(LPIPS_1)
            LPIPS_4 = float(LPIPS_2)
            # LPIPS_4 = float(LPIPS_3)
            # LPIPS_batch = float(torch.squeeze(LPIPS(output.cpu(), input_high.cpu())))
            LPIPS_output += [LPIPS_4]

            # prepare for save
            input_low = np.squeeze(input_low.cpu().numpy())
            result_1 = np.squeeze(R_low.cpu().numpy())
            result_2 = np.squeeze(I_low_3.cpu().numpy())
            result_3 = np.squeeze(I_delta_3.cpu().numpy())
            result_4 = np.squeeze(output.cpu().numpy())
            result_GT = np.squeeze(input_high.cpu().numpy())

            if save_predict:  # and idx+1 % 10 == 0:
                if save_R_L:
                    cat_image_1 = np.concatenate([input_low, result_4, result_GT], axis=2)
                    cat_image_2 = np.concatenate([result_1, result_2, result_3], axis=2)
                    cat_image = np.concatenate([cat_image_1, cat_image_2], axis=1)
                else:
                    cat_image = np.concatenate([input_low, result_4, result_GT], axis=2)

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


def load(model, ckpt_dir, train_phase):
    load_dir   = ckpt_dir + '/' + train_phase + '/'
    if os.path.exists(load_dir):
        load_ckpts = os.listdir(load_dir)
        load_ckpts.sort()
        load_ckpts = sorted(load_ckpts, key=len)
        if len(load_ckpts)>0:
            load_ckpt  = load_ckpts[-1]
            global_step= int(load_ckpt[:-4])
            ckpt_dict  = torch.load(load_dir + load_ckpt)
            if train_phase == 'Decom':
                model.DecomNet.load_state_dict(ckpt_dict)
            elif train_phase == 'Relight':
                model.RelightNet.load_state_dict(ckpt_dict)
            return True, global_step
        else:
            return False, 0
    else:
        return False, 0  # TODO 异常


if __name__ == '__main__':
    # TODO logger

    parser = argparse.ArgumentParser(description='Learning Low Light Image Enhancement')

    parser.add_argument('--gpu_id', dest='gpu_id',
                        default="6",
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--data_dir', dest='data_dir',
                        default='./data1/',
                        help='directory storing the test data')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir',
                        default='./ckpts/data1/',
                        help='directory for checkpoints')
    parser.add_argument('--res_dir', dest='res_dir',
                        default='./results/test1/low/',
                        help='directory for saving the results')

    args = parser.parse_args()
    if args.gpu_id != "-1":
        # Create directories for saving the results
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Test the model
        main(args.data_dir, args.ckpt_dir, args.res_dir, args.gpu_id)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
