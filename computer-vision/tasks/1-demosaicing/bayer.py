import numpy as np

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


