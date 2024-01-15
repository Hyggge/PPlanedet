import cv2
import os
import os.path as osp
import numpy as np

def imshow_lanes(img, lanes, show=False, out_file=None):
    img = cv2.resize(img, (1280, 720))
    mask = np.zeros_like(img)

    for lane in lanes:
        for i, (x, y) in enumerate(lane):
            x, y = x / 1640 * 1280, y / 590 * 720
            if i == 0:
                last_point = (int(x), int(y))
                continue
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            # cv2.circle(img, (x, y), 4, (255, 0, 0), 2)
            cv2.line(img, last_point, (x, y), (255, 0, 0), 8)
            cv2.line(mask, last_point, (x, y), (255, 255, 255), 8)
            last_point = (x, y)

    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)
        prefix, ext = osp.splitext(out_file)
        cv2.imwrite(prefix + "_mask" + ext, mask)