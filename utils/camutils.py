import pdb
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os

def cam_to_label(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None):
    b, c, h, w = cam.shape  # (2, 20, 28, 28)
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])  # (2, 20, 28, 28)
    valid_cam = cls_label_rep * cam    # (2, 20, 28, 28)
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)  # (2, 28, 28), (2, 28, 28)
    _pseudo_label += 1  # (2, 20, 28, 28)
    _pseudo_label[cam_value<=bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=high_thre] = ignore_index
        _pseudo_label[cam_value<=low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return valid_cam, pseudo_label

# 函数cam_to_roi_mask2用于根据输入的类别激活图(cam)、类别标签(cls_label)以及高阈值(hig_thre)和低阈值(low_thre),
# 生成感兴趣区域(ROI)掩码(roi_mask)。
def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    b, c, h, w = cam.shape  # (b, 20, 448, 448)
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])  # (b, 20, 448, 448)
    valid_cam = cls_label_rep * cam  # (2, 20, 448, 448)
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)  # 通过.max()返回(b, 448, 448)，沿着维度为1的通道取最大值，就是获取每个像素所对应的标签
    # _pseudo_label += 1 创建一个与类别激活图形状相同的初始ROI掩码（roi_mask），初始值为全1。掩码中的像素值表示对应位置的ROI状态
    roi_mask = torch.ones_like(cam_value, dtype=torch.int16)
    roi_mask[cam_value <= low_thre] = 0  # 将低于低阈值（low_thre）的类别激活值所对应的位置在ROI掩码中置为0，表示该位置不属于感兴趣区域
    roi_mask[cam_value >= hig_thre] = 2  # 将高于高阈值（hig_thre）的类别激活值所对应的位置在ROI掩码中置为2，表示该位置属于感兴趣区域

    return roi_mask

def get_valid_cam(cam, cls_label):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam

    return valid_cam

def ignore_img_box(label, img_box, ignore_index):

    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label
# 从给定的图像中根据ROI掩码提取裁剪图像，并返回裁剪后的图像列表以及相应的标志
def crop_from_roi_neg(images, roi_mask=None, crop_num=8, crop_size=96):

    crops = []
    
    b, c, h, w = images.shape   # (2, 3, 448, 448)

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)  # (2, 8, 3, 96, 96)
    flags = torch.ones(size=(b, crop_num + 2)).to(images.device)  # (2, 10)
    margin = crop_size//2

    for i1 in range(b):
        roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] <= 1).nonzero()  # 根据ROI掩码（roi_mask）获取非正样本的ROI索引（roi_index）。
        if roi_index.shape[0] < crop_num:   # 如果非正样本ROI的数量小于裁剪数量，则从所有ROI中随机选择（rand_index）
            roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] >= 0).nonzero() ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])  # 用于生成一个从0到n-1的随机列表
        crop_index = roi_index[rand_index[:crop_num], :]  # (8, 2)
        
        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0) 裁剪中心
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0+crop_size), w0:(w0+crop_size)]
            temp_mask = roi_mask[i1, h0:(h0+crop_size), w0:(w0+crop_size)]  # (96, 96)
            if temp_mask.sum() / (crop_size*crop_size) <= 0.2:  # 计算裁剪图像中不确定区域的比例
                # if ratio of uncertain regions < 0.2 then negative 如果比例小于等于0.2，则将相应的标志设置为0，表示该裁剪图像是负样本
                flags[i1, i2 + 2] = 0
    
    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1,)   # tuple(8)元组，(b, 1, 3, 96, 96)该函数将输入张量按照指定维度进行分块，并返回一个包含分块结果的元组
    crops = [c[:, 0] for c in _crops]   # list(8) -> (b, 3, 96, 96)

    return crops, flags  # list(8) -> (b, 3, 96, 96)  (2, 10)

def multi_scale_cam2(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    # b, 3, 448, 448
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)    # (2b, 3, 448, 448)

        # 辅助CAM，CAM  (b, 20, 28, 28) (b, 20, 28, 28)
        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:    # (多尺度)
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)   # (2b, 197, 168)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)    # (b, 20, 448,448)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))   # (b, 20, 448,448)
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5    # (b, 20, 448,448)

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)    # (b, 20, 448,448)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))    # (b, 20, 448,448)
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5    # (b, 20, 448,448)

    return cam, cam_aux

def label_to_aff_mask(cam_label, ignore_index=255):
    
    b, h, w = cam_label.shape   # (b, 28, 28)  # 获取输入CAM标签张量的维度信息

    _cam_label = cam_label.reshape(b, 1, -1)    # (2, 1, 784)   # (b, 1, h * w)将其视为一维张量,这样做是为了后续计算相似性掩码时使用
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])   # (2, 784, 784)  扩展操作，得到形状为 (b, h * w, h * w),_cam_label_rep[i]表示第i个样本的相似性掩码。
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1)  # (2, 784, 784) 将1，2维度交换位置
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)   # (2, 784, 784) 判断两者是否相等得到的是0,1矩阵
    
    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index
    aff_label[:, range(h*w), range(h*w)] = ignore_index
    return aff_label


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None, ignore_index=False,  img_box=None, down_scale=2):

    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()
    
    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)

    for idx, coord in enumerate(img_box):

        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))
        
        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1], coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1], coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label


def compute_seg_label_3(ori_img, cam_label, norm_cam, name, iter, saliency, cls_pred, save_heatmap=False,
                        cut_threshold=0.9):
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    cam_label = cam_label.astype(np.uint8)

    cam_dict = {}
    cam_np = np.zeros_like(norm_cam)
    for i in range(20):
        if cam_label[i] > 1e-5:
            cam_dict[i] = norm_cam[i]
            cam_np[i] = norm_cam[i]

    # # save heatmap
    # if save_heatmap:
    #     img = ori_img
    #     keys = list(cam_dict.keys())
    #     for target_class in keys:
    #         mask = cam_dict[target_class]
    #         heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #         img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    #         cam_output = heatmap * 0.5 + img * 0.5
    #         cv2.imwrite(os.path.join('/home/users/u5876230/ete_project/ete_output/vis/',
    #                                  name + '_{}_heatmap_orig.jpg'.format(classes[target_class])), cam_output)

    _, h, w = norm_cam.shape

    bg_score = np.power(1 - np.max(cam_np, 0), 32)
    bg_score = np.expand_dims(bg_score, axis=0)
    cam_all = np.concatenate((bg_score, cam_np))

    bkg_high_conf_area = np.zeros([h, w], dtype=bool)

    crf_label = np.argmax(cam_all, 0)

    crf_label[crf_label == 0] = 255
    # crf_label[saliency == 0] = 0

    for class_i in range(20):
        if cam_label[class_i] > 1e-5:
            cam_class = norm_cam[class_i, :, :]
            cam_class_order = cam_class[cam_class > 0]
            cam_class_order = np.sort(cam_class_order)
            confidence_pos = int(cam_class_order.shape[0] * cut_threshold)
            if confidence_pos > 0:
                confidence_value = cam_class_order[confidence_pos]
                bkg_high_conf_cls = np.logical_and((cam_class > confidence_value), (crf_label == 0))
                crf_label[bkg_high_conf_cls] = class_i + 1
                # saliency[bkg_high_conf_cls] = 255
                bkg_high_conf_conflict = np.logical_and(bkg_high_conf_cls, bkg_high_conf_area)
                crf_label[bkg_high_conf_conflict] = 255

                bkg_high_conf_area[bkg_high_conf_cls] = 1

    # remove background noise
    frg = ((crf_label != 0) * 255).astype('uint8')
    frg_dilate = cv2.morphologyEx(frg, cv2.MORPH_OPEN, kernel=np.ones((10, 10), np.uint8))
    crf_label[frg_dilate != 255] = 0

    # cv2.imwrite('/data/u5876230/ete_wsss/pseudo/{}.png'.format(name), crf_label)

    # rgb_pseudo_label = decode_segmap(crf_label, dataset="pascal")
    # cv2.imwrite('/home/users/u5876230/ete_project/ete_output/pseudo/{}_color.png'.format(name),
    # (rgb_pseudo_label * 255).astype('uint8') * 0.7 + ori_img * 0.3)
    # cv2.imwrite('/home/users/u5876230/ete_project/ete_output/vis/{}_orig.png'.format(name),ori_img)

    return crf_label