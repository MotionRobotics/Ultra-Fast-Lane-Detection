import torch, os, cv2
from PIL import Image
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor


if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('starting video...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    # net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    # use_aux=False).cuda() # we dont need auxiliary segmentation in testing
    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False) # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    cap = cv2.VideoCapture("/home/joe/Projects/lanenet_annotation/Ski_Dataset/joe/data_video0.avi") 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_w, img_h = 1280,720
    row_anchor = tusimple_row_anchor

    frame_succes, frame = cap.read()

    while frame_succes:
        img = frame
        img2 = Image.fromarray(img)
        x = img_transforms(img2)
        x = x.unsqueeze(0)+1
        out = net(x) # out shape is (1, griding_num + 1, cls_num_per_lane, num_lanes)
                     # i.e. (1, column_grid_resolution, row_anchors, num_lanes)
        
        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()  # (griding_num + 1, cls_num_per_lane, num_lanes)
        out_j = out_j[:, ::-1, :]  # (griding_num + 1, cls_num_per_lane.reversed(), num_lanes)
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)  # convert to probablilities (griding_num, cls_num_per_lane.reversed(), num_lanes)
        idx = np.arange(cfg.griding_num) + 1  # [1, 2, ... , 99, 100]
        idx = idx.reshape(-1, 1, 1)  # idx.shape = (100, 1, 1)
        loc = np.sum(prob * idx, axis=0)  # loc.shape = (56, 4), then loc[i, j] is the expected griding_num of row_anchor i of lane j,
                                          # explained here https://github.com/cfzd/Ultra-Fast-Lane-Detection/issues/99
        out_j = np.argmax(out_j, axis=0)  # Maximum likelihood griding_num for each row_anchor and lane
        loc[out_j == cfg.griding_num] = 0  # If the most likely griding_num is background, set location to zero, zero values are ignored when drawing
        out_j = loc
        print(out_j)

        for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            cv2.circle(img,ppp,5,(0,255,0),-1)
        cv2.imshow("Net out", img)
        cv2.waitKey(1)

        frame_succes, frame = cap.read()

    cap.release()