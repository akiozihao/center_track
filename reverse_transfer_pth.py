from collections import OrderedDict
import sys
import torch

def transfer_pth(source_pth):
    dst_pth_info = dict()
    dst_state_dict = OrderedDict()
    source = torch.load(source_pth)
    source_state_dict = source['state_dict']
    # fulltrain_state_dict = torch.load('../mot17_fulltrain.pth')['state_dict']
    for k, v in source_state_dict.items():
        type = k.split('.')[0]
        if type == 'backbone':
            nk, nv = trans_base(k, v)
        elif type == 'neck':
            nk, nv = trans_neck(k, v)
        elif type == 'bbox_head':
            nk, nv = trans_head(k, v)
        dst_state_dict[nk] = nv
    # assert str(fulltrain_state_dict.keys()) == str(dst_state_dict.keys())
    for k, v in source.items():
        if k == 'state_dict':
            continue
        else:
            dst_pth_info[k] = v
    dst_pth_info['state_dict'] = dst_state_dict
    return dst_pth_info


def trans_base(k, v):
    l_k = k.split('.')
    l_k[0] = 'base'
    if l_k[1] == 'base_layer' or l_k[1] == 'pre_img_layer' or l_k[1] == 'pre_hm_layer':
        if l_k[2] == 'conv':
            l_k[2] = '0'
        elif l_k[2] == 'bn':
            l_k[2] = '1'
    if l_k[1] == 'level0' or l_k[1] == 'level1':
        if l_k[2] == '0' and l_k[-2] == 'conv':
            l_k.pop(-2)
        elif l_k[2] == '0' and l_k[-2] == 'bn':
            l_k[2] = '1'
            l_k.pop(-2)
    if l_k[2] == 'root':
        if l_k[3] == 'bn1':
            l_k[3] = 'bn'
    if l_k[3] == 'root':
        if l_k[4] == 'bn1':
            l_k[4] = 'bn'
    return '.'.join(str(i) for i in l_k), v


def trans_neck(k, v):
    l_k = k.split('.')[1:]
    if l_k[0] == 'dla_up':
        l_k = trans_dla(k)
    elif l_k[0] == 'ida_up':
        l_k = trans_ida(k)
    return '.'.join(str(i) for i in l_k), v


def trans_ida(k):
    l_k = k.split('.')[1:]
    if l_k[2] == 'bn':
        l_k[2] = 'actf.0'
    if  len(l_k) > 3 and l_k[3] == 'conv_offset':
        l_k[3] = 'conv_offset_mask'
    return l_k


def trans_dla(k):
    l_k = k.split('.')[1:]
    if len(l_k) > 3 and l_k[3] == 'bn':
        l_k[3] = 'actf.0'
    if len(l_k) > 4 and l_k[4] == 'conv_offset':
        l_k[4] = 'conv_offset_mask'
    return l_k


def trans_head(k, v):
    l_k = k.split('.')[1:]
    if l_k[0] == 'heatmap_head':
        l_k[0] = 'hm'
    elif l_k[0] == 'offset_head':
        l_k[0] = 'reg'
    elif l_k[0] == 'wh_head':
        l_k[0] = 'wh'
    elif l_k[0] == 'tracking_head':
        l_k[0] = 'tracking'
    elif l_k[0] == 'ltrb_amodal_head':
        l_k[0] = 'ltrb_amodal'
    return '.'.join(str(i) for i in l_k), v

if __name__ == '__main__':
    new_pth_info = transfer_pth(sys.argv[1])
    torch.save(new_pth_info, sys.argv[2])
