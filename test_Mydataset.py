import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
import random
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.iner_track import InerTrack
import lib.test.parameter.iner_track as parameters
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time


save_flag = 1

def pre(path):

    if os.path.exists(os.path.join(path, 'flag.txt')):
        flag = 'common'
        # dicts = ['visible', 'infrared']
        with open(os.path.join(path, 'flag.txt')) as f:
            a = f.readlines()
            f.close()
        print('%s'%(a[0]))
        if a[0] == 'rgb':
            if os.path.exists(os.path.join(path, 'rgb_ref')):
                dicts = ['color', 'rgb_ref']
            else:
                dicts = ['color', 'infrared']
        elif a[0] == 'tir':
            if os.path.exists(os.path.join(path, 'tir_ref')):
                dicts = ['tir_ref', 'infrared']
            else:
                dicts = ['color', 'infrared']
            # dicts = ['tir_ref', 'infrared']
        else:
            return
    else:
        flag = 'attacked'
        dicts = ['visible', 'infrared']

    return dicts, flag


def genConfig(seq_path, set_type, name):

    if 'Mydataset' in set_type:
        dicts, flag = pre(seq_path)
        if flag == 'common':
            gt_rgb = name + '_1.txt'
            gt_tir = name + '_1.txt'
        else:
            gt_rgb = 'visible.txt'
            gt_tir = 'infrared.txt'

        RGB_img_list = sorted([seq_path + '/' + dicts[0] + '/' + p for p in os.listdir(seq_path + '/' + dicts[0]) if p.endswith(".jpg") or p.endswith(".png")])
        T_img_list = sorted([seq_path + '/' + dicts[1] + '/' + p for p in os.listdir(seq_path + '/' + dicts[1]) if p.endswith(".jpg") or p.endswith(".png")])

        RGB_gt = np.loadtxt(seq_path + '/' + gt_rgb, delimiter=',')
        T_gt = np.loadtxt(seq_path + '/' + gt_tir, delimiter=',')

    return RGB_img_list, T_img_list, RGB_gt, T_gt

def list2txt(bb):
    return str(bb[0]) + ',' + str(bb[1]) + ',' + str(bb[2]) + ',' + str(bb[3]) + '\n'

def save2file(res_root, name, bbox, toc):
    with open(res_root[0], 'a') as f:
        for bb in bbox:
            f.writelines(list2txt(bb))
        f.close()
    with open(res_root[1], 'a') as f:
        s = "Total time costed in video {} is {}, Fps is {}\n".format(name, str(toc / (len(bbox) - 1)), str((len(bbox)-1) / toc))
        f.writelines(s)
        f.close()

def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, verson='Base_t1', epoch=300, debug=0, script_name='prompt'):

    seq_txt = seq_name
    save_name = '{}_{}_{}'.format('Iner', verson, str(epoch))
    save_path = f'./results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    save_folder = f'./results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        # print(save_folder)
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return

    if script_name == 'tracker':
        params = parameters.parameters(yaml_name, epoch)
        mmtrack = MPLTTrack(params, 'Mydataset')  # "GTOT" # dataset_name
        tracker = RGBT(tracker=mmtrack)
    elif script_name == 'Iner':
        params = parameters.parameters(yaml_name, verson, epoch)
        mmtrack = InerTrack(params, dataset_name)  # "GTOT" # dataset_name
        tracker = RGBT(tracker=mmtrack)
    seq_path = seq_home + '/' + seq_name
    # print('——————————Process sequence: ' + seq_name + '——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name, seq_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    # result[0] = np.copy(RGB_gt[0])
    toc = 0
    pred = []
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = time.time()
        if frame_idx == 0:
            # initialization
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            tracker.initialize(image, RGB_gt[0].tolist(), seq_name)  # xywh
            pred.append(RGB_gt[0].tolist())
        elif frame_idx > 0:
            # track
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            region, confidence = tracker.track(image)  # xywh
            toc += time.time() - tic
            # result[frame_idx] = np.array(region)
            pred.append(region)
            print('——————————Process sequence: ' + seq_name + ', ————Frame: ' + str(frame_idx) + '——————————')
    if save_flag:
        save_time_path = f'./results/{dataset_name}/' + save_name + '/' + seq_txt + '_time.txt'
        save2file([save_path, save_time_path], seq_name, pred, toc)
        # toc += time.time() - tic
    # toc /= cv2.getTickFrequency()
    # if not debug:
    #     np.savetxt(save_path, result)
    # s = '{} , fps:{}'.format(seq_name, frame_idx / toc)
    # save_time_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    # np.savetxt(save_time_path, s)
    # print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class RGBT(object):
    def __init__(self, tracker):
        # print('——————————a——————————————')
        self.tracker = tracker

    def initialize(self, image, region, name):
        # print('——————————b——————————————')
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)

        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize([image, image], init_info, name)

    def track(self, img):
        '''TRACK'''
        # print('—————————c—————————————')
        outputs = self.tracker.track([img, img])
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score
def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
    print("Random seeds are initialized.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, required=False, default='Iner',
                        help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, required=False,
                        default='iner', 
                        help='Name of tracking method.')  
    parser.add_argument('--dataset_name', type=str, default='GTOT',
                        help='Name of dataset (MVRGBT).')
    parser.add_argument('--threads', default=4, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--verson', default='Iner_t100-hmft', type=str, help='Number of gpus')
    parser.add_argument('--epoch', default=25, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='sequential', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    args = parser.parse_args()

    init_seeds(10)

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    # path initialization
    seq_list = None
    if dataset_name == 'MVRGBT':
        seq_home = 'path to mvrgbt'
        with open(join(seq_home, 'list.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    else:
        raise ValueError("Error dataset!")
    # print('——————————z——————————————')
    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [
            (s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.verson, args.epoch, args.debug, args.script_name) for s
            in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [
            (s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.verson, args.epoch, args.debug, args.script_name) for s
            in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time() - start} seconds!")
