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
from stereo_dataset.depthmap_utils import depthmap_to_disparity
from stereo_dataset.gta_sfm_dataset import GTASfMStereoDataset
from stereo_dataset.stereo_sequence_folder import StereoSequenceFolder
from stereo_dataset import metrics

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

def write_images(output_dir, image_idx, depthmap_est, depthmap_true,
                 disparity_est, disparity_true):
    """Save colormapped depthmap images for debugging.
    """
    cmap = plt.get_cmap("magma")

    vmin = 0.0
    vmax = np.max(disparity_est)

    disparity = np.copy(disparity_est)
    debug = np.squeeze(cmap((disparity - vmin) / (vmax - vmin)))
    debug_image = Image.fromarray(np.uint8(debug[:, :, :3] * 255))
    debug_image.save(os.path.join(output_dir, "disparity_{}_est.jpg".format(image_idx)))

    disparity = np.copy(disparity_true)
    debug = np.squeeze(cmap((disparity - vmin) / (vmax - vmin)))
    debug_image = Image.fromarray(np.uint8(debug[:, :, :3] * 255))
    debug_image.save(os.path.join(output_dir, "disparity_{}_true.jpg".format(image_idx)))

    # idepthmaps.
    idepth_scale_factor = 5.0

    idepth = np.copy(depthmap_est)
    idepth[idepth > 0] = 1.0 / idepth[idepth > 0]
    debug = np.squeeze(cmap(idepth_scale_factor * idepth))
    debug_image = Image.fromarray(np.uint8(debug[:, :, :3] * 255))
    debug_image.save(os.path.join(output_dir, "idepthmap_{}_est.jpg".format(image_idx)))

    idepth = np.copy(depthmap_true)
    idepth[idepth > 0] = 1.0 / idepth[idepth > 0]
    debug = np.squeeze(cmap(idepth_scale_factor * idepth))
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

    errors = np.zeros((2, 8, int(len(val_loader)/args.print_freq)+1), np.float32)
    with torch.no_grad():
        for ii, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth) in enumerate(val_loader):
            if ii % args.print_freq == 0:
                i = int(ii / args.print_freq)
                tgt_img_var = Variable(tgt_img.cuda())
                ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
                ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses]
                intrinsics_var = Variable(intrinsics.cuda())
                intrinsics_inv_var = Variable(intrinsics_inv.cuda())
                tgt_depth_var = Variable(tgt_depth.cuda())

                if args.stereo_dataset:
                    # Load StereoDataset sample to get actual disparity.
                    sample = val_loader.dataset.stereo_dataset[ii]

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
                if args.stereo_dataset:
                    # Use mask defined from depthest.
                    mask_np = (sample["left_disparity_true"] >= 0) & (sample["left_disparity_true"] < 192)
                    mask = np.expand_dims(mask_np, 0)
                    mask = torch.from_numpy(mask)

                output_disp_ = torch.squeeze(output_disp.data.cpu(),1)
                output_depth_ = torch.squeeze(output_depth.data.cpu(),1)

                errors[0,:,i] = compute_errors_test(tgt_depth[mask], output_depth_[mask], mean_depth)
                errors[1,:,i] = compute_errors_test(tgt_disp[mask], output_disp_[mask], mean_depth)

                if args.stereo_dataset:
                    # Compute independent depth/disparity metrics.
                    left_depthmap_est = (output_depth_ * mean_depth).squeeze(0).numpy()
                    left_disparity_est = depthmap_to_disparity(sample["K"][:3, :3], sample["T_right_in_left"], left_depthmap_est)

                    # Compute depth metrics and write to file.
                    depth_metrics_idx = metrics.get_depth_prediction_metrics(
                        sample["left_depthmap_true"][mask_np], left_depthmap_est[mask_np])
                    depth_metrics_file = os.path.join(output_dir, "depth_metrics.txt")
                    if not os.path.exists(depth_metrics_file):
                        metrics.write_metrics_header(depth_metrics_file, depth_metrics_idx)
                    metrics.write_metrics(depth_metrics_file, sample["left_filename"], depth_metrics_idx)

                    # Compute disparity metrics and write to file.
                    disparity_metrics_idx = metrics.get_disparity_metrics(
                        sample["left_disparity_true"], left_disparity_est, mask_np)
                    disparity_metrics_file = os.path.join(output_dir, "disparity_metrics.txt")
                    if not os.path.exists(disparity_metrics_file):
                        metrics.write_metrics_header(disparity_metrics_file, disparity_metrics_idx)
                    metrics.write_metrics(disparity_metrics_file, sample["left_filename"], disparity_metrics_idx)

                    # Save runtime metrics.
                    runtime_metrics_file = os.path.join(output_dir, "runtime_metrics.txt")
                    if not os.path.exists(runtime_metrics_file):
                        with open(runtime_metrics_file, "w") as stream:
                            stream.write("file runtime_ms\n")
                    with open(runtime_metrics_file, "a") as stream:
                        stream.write("{} {}\n".format(sample["left_filename"], elps * 1000))

                print('Elapsed Time({}/{}): {} Abs Error {:.4f}'.format(ii, len(val_loader), elps, errors[0,0,i]))

                if args.output_print:
                    # output_disp_n = (output_disp_).numpy()[0]
                    # np.save(output_dir/'{:04d}{}'.format(i,'.npy'), output_disp_n)
                    # disp = (255*tensor2array(torch.from_numpy(output_disp_n), max_value=args.nlabel, colormap='bone')).astype(np.uint8)
                    # # imsave(output_dir/'{:04d}_disp{}'.format(i,'.png'), disp)
                    # pil_disp = Image.fromarray(disp)
                    # pil_disp.save(output_dir/'{:04d}_disp{}'.format(i,'.png'))

                    # Save depthmaps.
                    left_dir, file_and_ext = os.path.split(sample["left_filename"])
                    left_dir = left_dir.replace(val_loader.dataset.stereo_dataset.data_dir, "") # Strip dataset prefix.
                    left_output_dir = os.path.join(output_dir, left_dir[1:])
                    image_num = os.path.splitext(file_and_ext)[0]
                    if not os.path.exists(left_output_dir):
                        os.makedirs(left_output_dir)
                    assert(os.path.exists(left_output_dir))
                    write_images(left_output_dir, image_num,
                                 left_depthmap_est, sample["left_depthmap_true"],
                                 left_disparity_est, sample["left_disparity_true"])

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

    if args.stereo_dataset:
        # Compute metrics averaged across entire test set.
        avg_depth_metrics = metrics.compute_avg_metrics(os.path.join(output_dir, "depth_metrics.txt"))
        with open(os.path.join(output_dir, "avg_depth_metrics.txt"), "w") as ff:
            for key, value in avg_depth_metrics.items():
                ff.write("{}: {}\n".format(key, value))

        avg_disparity_metrics = metrics.compute_avg_metrics(os.path.join(output_dir, "disparity_metrics.txt"))
        with open(os.path.join(output_dir, "avg_disparity_metrics.txt"), "w") as ff:
            for key, value in avg_disparity_metrics.items():
                ff.write("{}: {}\n".format(key, value))

        runtimes = np.loadtxt(os.path.join(output_dir, "runtime_metrics.txt"),
                              skiprows=1, usecols=1)
        mean_runtime = np.mean(runtimes)
        with open(os.path.join(output_dir, "avg_runtime_metrics.txt"), "w") as ff:
            ff.write("runtime_ms: {}\n".format(mean_runtime))
            ff.write("num_samples: {}\n".format(len(runtimes)))

if __name__ == '__main__':
    main()
