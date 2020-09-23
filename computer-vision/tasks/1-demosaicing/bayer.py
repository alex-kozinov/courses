import numpy as np
import math


def get_bayer_masks(n_rows, n_cols):
    colors_cell = np.array(
        [
            ['G', 'R'],
            ['B', 'G']
        ]
    )
    
    filter_masks = []
    for color in ['R', 'G', 'B']:
        color_mask = np.tile(colors_cell == color, (n_rows, n_cols))[:n_rows, :n_cols]
        filter_masks.append(color_mask)
    
    return np.dstack(filter_masks)


def get_colored_img(raw_img):
    raw_img = np.array(raw_img)
    raw_img_3d = np.dstack([raw_img, raw_img, raw_img])
        
    masks = get_bayer_masks(*raw_img.shape)
    return raw_img_3d * masks


def bilinear_interpolation(colored_img):
    colored_img = colored_img.astype(np.float64)
    masks = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])

    result = colored_img.copy()
    for i in range(colored_img.shape[0]):
        for j in range(colored_img.shape[1]):
            for k in range(colored_img.shape[2]):
                if masks[i][j][k] == 1:
                    continue

                if i == 0 or i == colored_img.shape[0] - 1:
                    continue
                if j == 0 or j == colored_img.shape[1] - 1:
                    continue

                res = 0
                cnt = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if masks[i + dx][j + dy][k] == 0:
                            continue
                        res += colored_img[i + dx][j + dy][k]
                        cnt += 1
                result[i, j, k] = res / cnt

    return result.astype(np.uint8)


def improved_interpolation(raw_img):
    raw_img = get_colored_img(raw_img)
    raw_img = raw_img.astype(np.float64)
    
    masks = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    
    result = raw_img
    for i in range(raw_img.shape[0]):
            for j in range(raw_img.shape[1]):
                for k in range(raw_img.shape[2]):
                    if i <= 1 or i >= raw_img.shape[0] - 2:
                        continue
                    if j <= 1 or j >= raw_img.shape[1] - 2:
                        continue
                        
                    if masks[i][j][k] == 1:
                        result[i][j][k] = raw_img[i][j][k]
                        continue
                        
                    if k == 0:
                        if masks[i][j][1] and masks[i][j - 1][0]:
                            result[i][j][0] = raw_img[i][j][1] * 5
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][0] += raw_img[i + dx][j + dy][1] * (-1)
                            
                            for dy in [-2, 2]:
                                result[i][j][0] += raw_img[i][j + dy][1] * (-1)
                            
                            for dx in [-2, 2]:
                                result[i][j][0] += raw_img[i + dx][j][1] * 0.5
                            
                            for dy in [-1, 1]:
                                result[i][j][0] += raw_img[i][j + dy][0] * 4
                            
                            result[i][j][0] /= 8
                    
                        if masks[i][j][1] and masks[i - 1][j][0]:
                            result[i][j][0] = raw_img[i][j][1] * 5
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][0] += raw_img[i + dx][j + dy][1] * (-1)
                            
                            for dy in [-2, 2]:
                                result[i][j][0] += raw_img[i][j + dy][1] * 0.5
                            
                            for dx in [-2, 2]:
                                result[i][j][0] += raw_img[i + dx][j][1] * (-1)
                            
                            for dx in [-1, 1]:
                                result[i][j][0] += raw_img[i + dx][j][0] * 4
                            
                            result[i][j][0] /= 8
                            
                        if masks[i][j][2]:
                            result[i][j][0] = raw_img[i][j][2] * 6
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][0] += raw_img[i + dx][j + dy][0] * 2
                            
                            for d in [-2, 2]:
                                result[i][j][0] += raw_img[i + d][j][2] * (-1.5)
                                result[i][j][0] += raw_img[i][j + d][2] * (-1.5)
                                
                            result[i][j][0] /= 8
                            
                    if k == 1:
                        if masks[i][j][0]:
                            result[i][j][1] = raw_img[i][j][0] * 4
                            
                            for d in [-1, 1]:
                                result[i][j][1] += raw_img[i + d][j][1] * 2
                                result[i][j][1] += raw_img[i][j + d][1] * 2
                                
                            for d in [-2, 2]:
                                result[i][j][1] += raw_img[i + d][j][0] * (-1)
                                result[i][j][1] += raw_img[i][j + d][0] * (-1)
                        
                            result[i][j][1] /= 8 
                        
                        if masks[i][j][2]:
                            result[i][j][1] = raw_img[i][j][2] * 4
                            
                            for d in [-1, 1]:
                                result[i][j][1] += raw_img[i + d][j][1] * 2
                                result[i][j][1] += raw_img[i][j + d][1] * 2
                                
                            for d in [-2, 2]:
                                result[i][j][1] += raw_img[i + d][j][2] * (-1)
                                result[i][j][1] += raw_img[i][j + d][2] * (-1)
                        
                            result[i][j][1] /= 8 
                   
                    if k == 2:
                        if masks[i][j][1] and masks[i][j - 1][2]:
                            result[i][j][2] = raw_img[i][j][1] * 5
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][2] += raw_img[i + dx][j + dy][1] * (-1)
                            
                            for dy in [-2, 2]:
                                result[i][j][2] += raw_img[i][j + dy][1] * (-1)
                            
                            for dx in [-2, 2]:
                                result[i][j][2] += raw_img[i + dx][j][1] * 0.5
                            
                            for dy in [-1, 1]:
                                result[i][j][2] += raw_img[i][j + dy][2] * 4
                            
                            result[i][j][2] /= 8
                    
                        if masks[i][j][1] and masks[i - 1][j][2]:
                            result[i][j][2] = raw_img[i][j][1] * 5
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][2] += raw_img[i + dx][j + dy][1] * (-1)
                            
                            for dy in [-2, 2]:
                                result[i][j][2] += raw_img[i][j + dy][1] * 0.5
                            
                            for dx in [-2, 2]:
                                result[i][j][2] += raw_img[i + dx][j][1] * (-1)
                            
                            for dx in [-1, 1]:
                                result[i][j][2] += raw_img[i + dx][j][2] * 4
                            
                            result[i][j][2] /= 8
                            
                        if masks[i][j][0]:
                            result[i][j][2] = raw_img[i][j][0] * 6
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][2] += raw_img[i + dx][j + dy][2] * 2
                            
                            for d in [-2, 2]:
                                result[i][j][2] += raw_img[i + d][j][0] * (-1.5)
                                result[i][j][2] += raw_img[i][j + d][0] * (-1.5)
                                
                            result[i][j][2] /= 8
    return np.clip(result, 0, 255).astype(np.uint8)     


def compute_mse(img_pred, img_gt):
    return ((img_pred - img_gt) ** 2).mean()


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)
    mse = compute_mse(img_pred, img_gt)
    
    if mse == 0:
        raise ValueError
    
    return round(10 * math.log((img_gt.max() ** 2) / mse, 10), 14)