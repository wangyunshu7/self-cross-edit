import numpy as np
import torch
from utils3.gaussian_smoothing import GaussianSmoothing
from torch.nn import functional as F


def fn_get_topk(attention_map, K=1):
    H, W = attention_map.size()
    attention_map_detach = attention_map.view(H * W)  # .detach()
    topk_value, topk_index = attention_map_detach.topk(K, dim=0, largest=True, sorted=True)
    topk_coord_list = []
    topk_value_list = []
    for i in range(len(topk_index)):
        index = topk_index[i].cpu().numpy()
        coord = index // W, index % W
        topk_coord_list.append(coord)
        topk_value_list.append(topk_value[i])
    return topk_coord_list, topk_value_list


def fn_get_topk_plus(attention_map, K=1, threshold=0.8):
    H, W = attention_map.size()
    attention_map_detach = attention_map.view(H * W)  # .detach()
    topk_value, topk_index = attention_map_detach.topk(H * W, dim=0, largest=True, sorted=True)
    topk_coord_list = []
    topk_value_list = []
    threshold_coord_list = []
    threshold_value_list = []
    for i in range(len(topk_index)):
        index = topk_index[i].cpu().numpy()
        coord = index // W, index % W
        if i < K:  # topk_value[i]> threshold and
            topk_coord_list.append(coord)
            topk_value_list.append(topk_value[i])
            threshold_coord_list.append(coord)
            threshold_value_list.append(topk_value[i])
        elif i < 4 * K or topk_value[i] > threshold:
            threshold_coord_list.append(coord)
            threshold_value_list.append(topk_value[i])
        if i > 4 * K and topk_value[i] < threshold: break
    """if len(topk_value_list)==0:
            index = topk_index[0].cpu().numpy()
            coord = index // W, index % W
            topk_coord_list.append(coord)
            topk_value_list.append(topk_value[0])
            threshold_coord_list.append(coord)
            threshold_value_list.append(topk_value[0])"""

    return topk_coord_list, topk_value_list, threshold_coord_list, threshold_value_list


def fn_smoothing_func(attention_map):
    smoothing = GaussianSmoothing().to(attention_map.device)
    attention_map = F.pad(attention_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    attention_map = smoothing(attention_map).squeeze(0).squeeze(0)
    return attention_map


def fn_show_attention(
        cross_attention_maps,
        self_attention_maps,
        indices,
        K=1,
        attention_res=16,
        smooth_attentions=True,
):
    cross_attention_map_list, self_attention_map_list = [], []

    # cross attention map preprocessing
    cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps.float(), dim=-1)

    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]

    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_per_token = fn_smoothing_func(cross_attention_map_per_token)
        cross_attention_map_list.append(cross_attention_map_per_token)

    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        topk_coord_list, topk_value = fn_get_topk(cross_attention_map_per_token, K=K)

        self_attention_map_per_token_list = []
        for coord_x, coord_y in topk_coord_list:
            self_attention_map_per_token = self_attention_maps[coord_x, coord_y]
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()
            self_attention_map_per_token_list.append(self_attention_map_per_token)

        if len(self_attention_map_per_token_list) > 0:
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / len(
                self_attention_map_per_token_list)
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
                                            (
                                                        self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)

        self_attention_map_list.append(norm_self_attention_map_per_token)

    # tensor to numpy
    cross_attention_map_numpy = torch.cat(cross_attention_map_list, dim=0).cpu().detach().numpy()
    self_attention_map_numpy = torch.cat(self_attention_map_list, dim=0).cpu().detach().numpy()

    return cross_attention_map_numpy, self_attention_map_numpy


def fn_show_attention_plus(
        cross_attention_maps,
        self_attention_maps,
        indices,
        K=1,
        attention_res=16,
        smooth_attentions=True,
):
    # cross attention map preprocessing
    cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps.float(), dim=-1)

    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]
    cross_attention_map_list = []
    cross_attention_map_list_for_show = []
    self_attention_map_top_list = []
    otsu_masks = []
    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_per_token = fn_smoothing_func(cross_attention_map_per_token)
        topk_coord_list, topk_value_list = fn_get_topk(cross_attention_map_per_token, K=K)
        self_attn_map_top1 = self_attention_maps[topk_coord_list[0][0], topk_coord_list[0][1]].view(attention_res, attention_res).contiguous()
        # print(self_attn_map_top1.shape, self_attention_maps.shape)
        cross_attention_map_list_for_show.append(cross_attention_map_per_token.to(torch.float16).cpu().detach().numpy())
        self_attention_map_top_list.append(self_attn_map_top1.to(torch.float16).cpu().detach().numpy())
        cross_attention_map_list.append(cross_attention_map_per_token)

        # -----------------------------------
        # clean cross_attention_map_cur_token
        # -----------------------------------
        clean_cross_attention_map_per_token_mask = fn_get_otsu_mask(cross_attention_map_per_token)
        # clean_cross_attention_map_per_token_mask = fn_clean_mask(clean_cross_attention_map_per_token_mask, topk_coord_list[0][0], topk_coord_list[0][1])
        otsu_masks.append(clean_cross_attention_map_per_token_mask)

    self_attention_map_list = []
    self_attention_map_list_list=[]
    for i in range(len(cross_attention_map_list)):
        cross_attn_map_cur_token = cross_attention_map_list[i]
        self_attn_map_cur_token = torch.zeros_like(cross_attn_map_cur_token)
        mask_cur_token = otsu_masks[i]
        cross_attn_value_cur_token_sum = 0
        self_attention_map_per_token_list = []
        self_atten_map_list=[]
        for j in range(attention_res):
            for k in range(attention_res):
                if mask_cur_token[j, k] == 0: continue
                cross_attn_value_cur_token = cross_attn_map_cur_token[j, k]
                cross_attn_value_cur_token_sum = cross_attn_value_cur_token_sum + cross_attn_value_cur_token
                self_attn_map_cur_position = self_attention_maps[j, k].view(attention_res, attention_res).contiguous()
                self_attn_map_cur_token = self_attn_map_cur_token + cross_attn_value_cur_token * self_attn_map_cur_position
                self_atten_map_list.append(self_attn_map_cur_position)
        self_attention_map_per_token_list.append(self_attn_map_cur_token)
        self_attention_map_list_list.append(self_atten_map_list)
        if len(self_attention_map_per_token_list) > 0:
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / cross_attn_value_cur_token_sum
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
                                            (
                                                    self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
        # print(norm_self_attention_map_per_token.shape)
        self_attention_map_list.append(norm_self_attention_map_per_token.to(torch.float16).cpu().detach().numpy())

    # tensor to numpy
    # cross_attention_map_numpy = torch.cat(cross_attention_map_list, dim=0).cpu().detach().numpy()
    # self_attention_map_numpy = torch.cat(self_attention_map_list, dim=0).cpu().detach().numpy()

    return cross_attention_map_list_for_show, self_attention_map_list, self_attention_map_top_list  


import cv2


def fn_get_otsu_mask(x):
    x_numpy = x.to(torch.float16)
    x_numpy = x_numpy.cpu().detach().numpy()
    x_numpy = x_numpy * 255
    x_numpy = x_numpy.astype(np.uint16)

    opencv_threshold, _ = cv2.threshold(x_numpy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opencv_threshold = opencv_threshold * 1. / 255.

    otsu_mask = torch.where(
        x < opencv_threshold,
        torch.tensor(0, dtype=x.dtype, device=x.device),
        torch.tensor(1, dtype=x.dtype, device=x.device))

    return otsu_mask


def fn_clean_mask(otsu_mask, x, y):
    H, W = otsu_mask.size()
    direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def dfs(cur_x, cur_y):
        if cur_x >= 0 and cur_x < H and cur_y >= 0 and cur_y < W and otsu_mask[cur_x, cur_y] == 1:
            otsu_mask[cur_x, cur_y] = 2
            for delta_x, delta_y in direction:
                dfs(cur_x + delta_x, cur_y + delta_y)

    dfs(x, y)
    ret_otsu_mask = torch.where(
        otsu_mask < 2,
        torch.tensor(0, dtype=otsu_mask.dtype, device=otsu_mask.device),
        torch.tensor(1, dtype=otsu_mask.dtype, device=otsu_mask.device))

    return ret_otsu_mask


def fn_show_attention_plus_2(
        cross_attention_maps,
        self_attention_maps,
        indices,
        K=1,
        attention_res=16,
        smooth_attentions=True,
):
    # -----------------------------
    # 1. 预处理 (恢复你原来的处理方式)
    # -----------------------------
    cross_attention_maps = cross_attention_maps[:, :, 0:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps.float(), dim=-1)

    # -----------------------------
    # 2. 解析嵌套的 indices 并分类 Token
    # -----------------------------
    all_shifted_indices = []
    entity_indices = []
    attr_indices = []
    for group in indices:
        if not isinstance(group, list):
            group = [group]

        # 所有索引减 1 (偏移)
        shifted_group = []
        for item in group:
            if isinstance(item, str) and ":" in item:
                start, end = map(int, item.split(":"))
                shifted_group.append(tuple(idx for idx in range(start, end)))
            else:
                shifted_group.append(int(item))


        if len(shifted_group) > 0:

            ent = shifted_group[-1]
            entity_indices.append(ent)

            for attr in shifted_group[:-1]:
                attr_indices.append(attr)


    # 去重并保持顺序

    entity_indices = list(dict.fromkeys(entity_indices))
    attr_indices = list(dict.fromkeys(attr_indices))


    entity_cross_list = []
    attr_cross_list = []
    entity_self_list = []
    entity_otsu_masks = {}
    entity_cross_attn_maps = {}
    for i in entity_indices:
        if isinstance(i, tuple):
            attn_map = cross_attention_maps[:, :, list(i)].max(dim=-1)[0]
        else:
            attn_map = cross_attention_maps[:, :, i]
        if smooth_attentions:
            attn_map = fn_smoothing_func(attn_map)

        entity_cross_list.append(attn_map.to(torch.float16).cpu().detach().numpy())
        entity_cross_attn_maps[i] = attn_map  # 【修复2】保存起来供第4步使用

        topk_coords, _ = fn_get_topk(attn_map, K=K)
        clean_mask = fn_get_otsu_mask(attn_map)
        entity_otsu_masks[i] = clean_mask



    # 3.2 收集剩余属性词的 Cross Attention
    for i in attr_indices:
        if isinstance(i, tuple):
            cross_map = cross_attention_maps[:, :, list(i)].max(dim=-1)[0]
        else:
            cross_map = cross_attention_maps[:, :, i]

        if smooth_attentions:
            cross_map = fn_smoothing_func(cross_map)
        attr_cross_list.append(cross_map.to(torch.float16).cpu().detach().numpy())

    # -----------------------------
    # 4. 向量化计算实体 Token 的 Self Attention 提纯
    # -----------------------------
    # 展平为 [H*W, H*W] 的注意力矩阵
    self_attention_map_list = []

    for i in entity_indices:
        cross_attn_map_cur_token = entity_cross_attn_maps[i]
        self_attn_map_cur_token = torch.zeros_like(cross_attn_map_cur_token)
        mask_cur_token = entity_otsu_masks[i]
        cross_attn_value_cur_token_sum = 0
        self_attention_map_per_token_list = []
        self_atten_map_list = []
        for j in range(attention_res):
            for k in range(attention_res):
                if mask_cur_token[j, k] == 0: continue
                cross_attn_value_cur_token = cross_attn_map_cur_token[j, k]
                cross_attn_value_cur_token_sum += cross_attn_value_cur_token
                self_attn_map_cur_position = self_attention_maps[j, k].view(attention_res, attention_res).contiguous()
                self_attn_map_cur_token += cross_attn_value_cur_token * self_attn_map_cur_position
                self_atten_map_list.append(self_attn_map_cur_position)
        self_attention_map_per_token_list.append(self_attn_map_cur_token)

        if len(self_attention_map_per_token_list) > 0:
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / cross_attn_value_cur_token_sum
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
                                            (
                                                    self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
        entity_self_list.append(norm_self_attention_map_per_token.to(torch.float16).cpu().detach().numpy())

    # 返回三个列表
    return entity_cross_list, attr_cross_list, entity_self_list




