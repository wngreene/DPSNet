from models import PSNet as PSNet

import argparse
import time
import csv
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision as tv
import custom_transforms
from utils import tensor2array
from loss_functions import compute_errors_test
from sequence_folders import SequenceFolder

import stereo_dataset.stereo_dataset as sd
from stereo_dataset.gta_sfm_dataset import GTASfMStereoDataset
from stereo_dataset.stereo_sequence_folder import StereoSequenceFolder

import os
from path import Path
# from scipy.misc import imsave
from PIL import Image

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=2)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--pretrained-dps', dest='pretrained_dps', default=None, metavar='PATH',
                    help='path to pre-trained dpsnet model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--output-dir', default='result', type=str, help='Output directory for saving predictions in a big 3D numpy file')
parser.add_argument('--ttype', default='test.txt', type=str, help='Text file indicates input data')
parser.add_argument('--nlabel', type=int ,default=64, help='number of label')
parser.add_argument('--mindepth', type=float ,default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float ,default=10, help='maximum depth')
parser.add_argument('--output-print', action='store_true', help='print output depth')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency')

parser.add_argument("--normalize_depths", action="store_true", help="Normalize poses/depths by groundtruth mean depth.")
parser.add_argument("--stereo_dataset", action="store_true", help="Use StereoDataset data.")
parser.add_argument("--roll_right_image_180", action="store_true", help="Roll right images by 180 degrees.")

def write_images(output_dir, image_idx, idepthmap_est, idepthmap_true):
    """Save colormapped depthmap images for debugging.
    """
    cmap = plt.get_cmap("magma")

    vmin = 0.0
    vmax = np.max(idepthmap_true)

    debug = np.squeeze(cmap((idepthmap_est - vmin) / (vmax - vmin)))
    debug_image = Image.fromarray(np.uint8(debug[:, :, :3] * 255))
    debug_image.save(os.path.join(output_dir, "idepthmap_{}_est.jpg".format(image_idx)))

    debug = np.squeeze(cmap((idepthmap_true - vmin) / (vmax - vmin)))
    debug_image = Image.fromarray(np.uint8(debug[:, :, :3] * 255))
    debug_image.save(os.path.join(output_dir, "idepthmap_{}_true.jpg".format(image_idx)))

    return

def main():
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    if args.stereo_dataset:
        roll_transform = None
        if args.roll_right_image_180:
            right_roll = lambda sample: sd.roll_right_image_180(sample)
            roll_transform = tv.transforms.Compose([tv.transforms.Lambda(right_roll)])
        val_stereo_dataset = GTASfMStereoDataset(
            args.data, "./stereo_dataset/gta_sfm_overlap0.5_test.txt", 0, roll_transform, True)
        val_set = StereoSequenceFolder(val_stereo_dataset, transform=valid_transform)
        print('{} samples found in val_set'.format(len(val_set)))
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            ttype=args.ttype,
            sequence_length=args.sequence_length
        )
        print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    dpsnet = PSNet(args.nlabel, args.mindepth).cuda()
    weights = torch.load(args.pretrained_dps)
    dpsnet.load_state_dict(weights['state_dict'])
    dpsnet.eval()

    num_parameters = sum(p.numel() for p in dpsnet.parameters() if p.requires_grad)
    print("Num parameters: {}".format(num_parameters))

    output_dir= Path(args.output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    left_filenames = []
    errors = np.zeros((2, 8, int(len(val_loader)/args.print_freq)+1), np.float32)
    with torch.no_grad():
        for ii, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, left_filename) in enumerate(val_loader):
            if ii % args.print_freq == 0:
                i = int(ii / args.print_freq)
                tgt_img_var = Variable(tgt_img.cuda())
                ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
                ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses]
                intrinsics_var = Variable(intrinsics.cuda())
                intrinsics_inv_var = Variable(intrinsics_inv.cuda())
                tgt_depth_var = Variable(tgt_depth.cuda())

                mean_depth = 1.0
                if args.normalize_depths:
                    # Scale poses and gt depths by mean depth. Network output should be
                    # multiplied by mean_depth to get back to metric units. We only need
                    # metric depths when computing metrics.
                    assert(len(ref_poses_var) == 1)
                    mean_depth = torch.mean(tgt_depth[tgt_depth < 1000])
                    tgt_depth /= mean_depth
                    tgt_depth_var /= mean_depth
                    for ref_idx in range(len(ref_poses_var)):
                        ref_poses[ref_idx][:, :, :3, 3] /= mean_depth
                        ref_poses_var[ref_idx][:, :, :3, 3] /= mean_depth

                # compute output
                pose = torch.cat(ref_poses_var,1)
                start = time.time()
                output_depth = dpsnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
                elps = time.time() - start
                tgt_disp = args.mindepth*args.nlabel/tgt_depth
                output_disp = args.mindepth*args.nlabel/output_depth

                mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)

                output_disp_ = torch.squeeze(output_disp.data.cpu(),1)
                output_depth_ = torch.squeeze(output_depth.data.cpu(),1)

                errors[0,:,i] = compute_errors_test(tgt_depth[mask], output_depth_[mask], mean_depth)
                errors[1,:,i] = compute_errors_test(tgt_disp[mask], output_disp_[mask], mean_depth)
                left_filenames.append(str(left_filename))

                print('Elapsed Time {} Abs Error {:.4f}'.format(elps, errors[0,0,i]))

                if args.output_print:
                    # output_disp_n = (output_disp_).numpy()[0]
                    # np.save(output_dir/'{:04d}{}'.format(i,'.npy'), output_disp_n)
                    # disp = (255*tensor2array(torch.from_numpy(output_disp_n), max_value=args.nlabel, colormap='bone')).astype(np.uint8)
                    # # imsave(output_dir/'{:04d}_disp{}'.format(i,'.png'), disp)
                    # pil_disp = Image.fromarray(disp)
                    # pil_disp.save(output_dir/'{:04d}_disp{}'.format(i,'.png'))

                    # Save depthmaps.
                    tgt_idepth = torch.squeeze(tgt_depth.data).cpu().numpy()
                    tgt_idepth[tgt_idepth > 0] = 1.0 / tgt_idepth[tgt_idepth > 0]

                    output_idepth_ = np.squeeze(np.copy(output_depth_))
                    output_idepth_[output_idepth_ > 0] = 1.0 / output_idepth_[output_idepth_ > 0]

                    left_filename = val_loader.dataset.samples[ii]["tgt"]
                    left_dir, file_and_ext = os.path.split(left_filename)
                    left_dir = left_dir.replace(args.data, "") # Strip dataset prefix.

                    if left_dir[0] == "/":
                        left_output_dir = os.path.join(output_dir, left_dir[1:])
                    else:
                        left_output_dir = os.path.join(output_dir, left_dir)

                    image_num = os.path.splitext(file_and_ext)[0]
                    if not os.path.exists(left_output_dir):
                        os.makedirs(left_output_dir)
                    assert(os.path.exists(left_output_dir))
                    write_images(left_output_dir, image_num,
                                 output_idepth_, tgt_idepth)


    mean_errors = errors.mean(2)
    error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3']
    print("{}".format(args.output_dir))
    print("Depth Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    print("Disparity Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))

    np.savetxt(output_dir/'errors.csv', mean_errors, fmt='%1.4f', delimiter=',')
    np.savetxt(output_dir/"raw_depth_errors.csv", errors[0, :, :].T)

    with open(output_dir/"left_filenames.txt", "w") as ff:
        for idx in range(len(left_filenames)):
            ff.write("{}\n".format(left_filenames[idx]))

    mvs_errors = open(output_dir/"raw_depth_errors_mvs.csv", "w")
    rgbd_errors = open(output_dir/"raw_depth_errors_rgbd.csv", "w")
    sun3d_errors = open(output_dir/"raw_depth_errors_sun3d.csv", "w")
    scenes11_errors = open(output_dir/"raw_depth_errors_scenes11.csv", "w")
    for idx in range(len(left_filenames)):
        if "mvs" in left_filenames[idx]:
            mvs_errors.write("{},{},{},{},{},{},{},{}\n".format(
                errors[0, 0, idx], errors[0, 1, idx], errors[0, 2, idx], errors[0, 3, idx],
                errors[0, 4, idx], errors[0, 5, idx], errors[0, 6, idx], errors[0, 7, idx]))
        elif "rgbd" in left_filenames[idx]:
            rgbd_errors.write("{},{},{},{},{},{},{},{}\n".format(
                errors[0, 0, idx], errors[0, 1, idx], errors[0, 2, idx], errors[0, 3, idx],
                errors[0, 4, idx], errors[0, 5, idx], errors[0, 6, idx], errors[0, 7, idx]))
        elif "sun3d" in left_filenames[idx]:
            sun3d_errors.write("{},{},{},{},{},{},{},{}\n".format(
                errors[0, 0, idx], errors[0, 1, idx], errors[0, 2, idx], errors[0, 3, idx],
                errors[0, 4, idx], errors[0, 5, idx], errors[0, 6, idx], errors[0, 7, idx]))
        elif "scenes11" in left_filenames[idx]:
            scenes11_errors.write("{},{},{},{},{},{},{},{}\n".format(
                errors[0, 0, idx], errors[0, 1, idx], errors[0, 2, idx], errors[0, 3, idx],
                errors[0, 4, idx], errors[0, 5, idx], errors[0, 6, idx], errors[0, 7, idx]))
        else:
            print("Bad image: {}".format(left_filenames[idx]))
            assert(False)

if __name__ == '__main__':
    main()
