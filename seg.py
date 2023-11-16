import argparse
import os

import numpy as np
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import cv2
from mmcv.image import tensor2imgs

import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
    
def draw(model,
                    data_loader,
                    show=False,
                    out_dir='.\\final\\oppofind2x',
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, ft_maps, seg_maps = model(return_loss=False, **data)
        if True:
            out_dir='/mnt/ECE445/linjietong/SeMask-FPN/final/test'
        #if show or out_dir:
            #if i % 10 == 0:
            if True:
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for img, img_meta in zip(imgs, img_metas):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w , ori_h ))
                    #print(out_dir)
                    #print(img_meta['ori_filename'])
                    if out_dir:
                        #print(111)
                        out_file = osp.join(out_dir, str(i) + '_' + img_meta['ori_filename'])
                    else:
                        out_file = None
                    
                    ###print seg
                    seg = result[0]
                    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                    l1=img_show.copy()
                    l1[seg != 1] =0
                    color_seg[seg == 1] = 255
                    print('\n')
                    print(seg.max())
                    print('\n')
                    import cv2
                    if not os.path.exists(osp.join(out_dir, 'l1')):
                      os.makedirs(osp.join(out_dir, 'l1'))
                    if not os.path.exists(osp.join(out_dir, 'l2')):
                      os.makedirs(osp.join(out_dir, 'l2'))
                    #cv2.imwrite(osp.join(out_dir, 'l1',img_meta['ori_filename'][:-4]+'.png'),l1)
                    cv2.imwrite(osp.join(out_dir, 'l1',str(i)+'.png'),l1)
                    l2=img_show.copy()
                    l2[seg != 2] =0
                    #cv2.imwrite(osp.join(out_dir, 'l2',img_meta['ori_filename'][:-4]+'.png'),l2)
                    cv2.imwrite(osp.join(out_dir, 'l2',str(i)+'.png'),l2)

                    
        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def main():
    
    args = parse_args()
    
    #sample the video
    store='/mnt/ECE445/linjietong/SeMask-FPN/frame/test/'
    if not os.path.exists(store):
        os.makedirs(store)
    number=20
    VideoName='/mnt/ECE445/linjietong/SeMask-FPN/video/test.mp4'
    cap = cv2.VideoCapture(VideoName) 
    #get frame number
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    freq=frame_count/number
    print(freq)
    f=[]
    for i in range(number):
        f.append(int((i+1)*freq)-1)
    print(f)
    counter=0
    while True:
        res,img=cap.read()
        if(res==False):
            break
        if counter in f:
            print(counter)
            #w=os.path.join(store,VideoName[:-4]+'_'+str(counter)+'.png')
            w=os.path.join(store,str(counter)+'.png')
            print(w)
            cv2.imwrite(w,img)
        counter+=1
        
    #print(frame_count)
    #print(img)
#res,img=cap.read()
#print(res)
#print(counter)
    cap.release()
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)
    
    net_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params: {} M".format(net_params/1e6))

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        draw(model, data_loader, args.show, args.show_dir, efficient_test)
        #model.eval()
        #img=cv2.imread("/mnt/ECE445/linjietong/SeMask-FPN/data/Dental3/test/images/C01007541210-3-10_6.png")
        #img=np.expand_dims(img,0)
        #img=np.expand_dims(img,0)
        #img=torch.from_numpy(img)
        #img = img.cuda()
        #output=model(img)
        print('1')
        #outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  #efficient_test)
    #else:
    #    model = MMDistributedDataParallel(
    #        model.cuda(),
    #       device_ids=[torch.cuda.current_device()],
    #       broadcast_buffers=False)
    #    outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                # args.gpu_collect, efficient_test)
    '''
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)
    '''

if __name__ == '__main__':
    main()