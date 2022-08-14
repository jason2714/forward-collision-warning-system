import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from pathlib import Path

last_left_line = []
last_right_line = []


def get_mask_ratio(mask):
    pixel_count = np.prod(mask.shape)
    zero_count = np.count_nonzero(mask == 0)
    return zero_count / pixel_count


def get_lane_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_dict_HSV = {'black': [(0, 0, 0), (180, 255, 70)],
                      'white': [(0, 0, 231), (180, 18, 255)],
                      'yellow': [(25, 50, 70), (35, 255, 255)],
                      # 'gray': [(0, 0, 40), (180, 25, 230)],
                      'gray': [(0, 0, 40), (180, 40, 230)],
                      'brown': [(20, 0, 50), (180, 125, 200)]}
    mask_yellow = cv2.inRange(hsv, *color_dict_HSV['yellow'])
    mask_gray = cv2.inRange(hsv, *color_dict_HSV['gray'])
    # mask_white = cv2.inRange(hsv, *color_dict_HSV['white'])
    # mask_gray = cv2.bitwise_or(mask_white, mask_gray)
    # mask_brown = cv2.inRange(hsv, *color_dict_HSV['brown'])
    # mask_gray = cv2.bitwise_or(mask_brown, mask_gray)
    mask_lane = cv2.bitwise_or(mask_gray, mask_yellow)
    # remove noise
    mask_lane = cv2.morphologyEx(mask_lane, cv2.MORPH_OPEN, kernel=np.ones((5, 5), dtype=np.uint8), iterations=6)
    # close mask
    mask_lane = cv2.morphologyEx(mask_lane, cv2.MORPH_CLOSE, kernel=np.ones((20, 20), dtype=np.uint8))
    #
    # improve mask by drawing the convexhull
    contours, hierarchy = cv2.findContours(mask_lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(mask_lane, [hull], 0, (255), -1)
    # erode mask a bit to migitate mask bleed of convexhull
    mask_lane = cv2.morphologyEx(mask_lane, cv2.MORPH_ERODE, kernel=np.ones((5, 5), dtype=np.uint8))

    # img = cv2.bitwise_and(img, img, mask=mask_lane)
    return mask_lane


def process_image(img):
    mask_lane = get_lane_mask(img.copy())
    mask_ratio = get_mask_ratio(mask_lane)
    # print(mask_ratio)
    if mask_ratio >= 0.85:
        img_lane = img.copy()
    else:
        img_lane = cv2.bitwise_and(img, img, mask=mask_lane)
    #
    # cv2.imshow('test', img_lane)
    # cv2.waitKey(0)

    gray_image = grayscale(img_lane)
    # increase gaussian kernel size
    gaus_blur = gaussian_blur(gray_image, 7)
    # adjust threshold
    edges = canny(gaus_blur, 50, 100)
    imshape = img.shape

    vertices = np.array([[(0, imshape[0] * 5 / 6),  # 左下
                          (imshape[1] / 2, imshape[0] / 5),  # 中間
                          (imshape[1] / 2, imshape[0] / 5),  # 中間
                          (imshape[1], imshape[0] * 5 / 6)]],  # 右下
                        dtype=np.int32)
    masked = region_of_interest(edges, vertices)

    # print(imshape)
    # print(vertices)
    # cv2.imshow('test', masked)
    # cv2.waitKey(0)

    rho = 1  # 半徑的分辨率
    theta = np.pi / 180  # 角度分辨率
    threshold = 20  # 判斷直線點數的臨界值
    min_line_len = 10  # 線段長度臨界值
    max_line_gap = imshape[0] / 3  # 線段上最近兩點之間的臨界值
    line_image = hough_lines(masked, rho, theta, threshold,
                             min_line_len, max_line_gap)

    # if (check):
    #     img = draw_text(img)

    result = weighted_img(line_image, img)
    # result = cv2.bitwise_and(result, result, mask=masked)
    return result


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # red_color = (255, 255, 255) # BGR
    # cv2.fillPoly(img, vertices, red_color)

    # cv2.imshow('Result', img)
    # cv2.waitKey(0)

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    global lastLineL, lastLineR
    lines = cv2.HoughLinesP(img,
                            rho,
                            theta,
                            threshold,
                            np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # self made
    min_slope_thr = 0.3
    max_slope_thr = 3
    if lines is None:
        # cv2.imshow('Result', img)
        # cv2.waitKey(0)
        lines = []
    else:
        lines = choose_lines(lines.reshape(-1, 4), min_slope_thr, max_slope_thr)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img = draw_lines(line_img, lines)

    return line_img


def get_average_line(lines, default_line):
    lines = np.array(lines)
    if not lines.any():
        return default_line, [default_line]

    n = 3
    normal_indices = []
    lines = np.concatenate([lines, [default_line] * n], axis=0) if np.any(default_line) else lines
    lines_mean = np.mean(lines, axis=0)
    lines_std = np.std(lines, axis=0)
    for m in range(1, 3):
        normal_indices = np.all(abs(lines - lines_mean) <= m * lines_std, axis=1)
        if np.any(normal_indices):
            break
    # outlier = not np.all(normal_indices)
    lines = lines[normal_indices]
    average_line = np.mean(lines, axis=0, dtype=np.int) if np.any(lines) else default_line

    return average_line, lines


def draw_lines(img, lines, thickness=3):
    global last_left_line, last_right_line
    img_height, img_width, _ = img.shape
    # for line in lines:
    #     cv2.line(img, line[:2], line[2:], [0, 255, 255], thickness)
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    slopes = np.array(list(map(lambda x: get_slope(*x), lines)))
    left_lines = lines[slopes < 0] if np.any(slopes) else []
    right_lines = lines[slopes > 0] if np.any(slopes) else []
    right_average_line, right_lines = get_average_line(right_lines, last_right_line)
    last_right_line = right_average_line
    left_average_line, left_lines = get_average_line(left_lines, last_left_line)
    last_left_line = left_average_line
    cv2.line(img, right_average_line[:2], right_average_line[2:], [0, 255, 0], thickness)
    cv2.line(img, left_average_line[:2], left_average_line[2:], [0, 255, 0], thickness)

    for line in right_lines:
        cv2.line(img, line[:2], line[2:], [0, 255, 255], thickness)
    for line in left_lines:
        cv2.line(img, line[:2], line[2:], [0, 255, 255], thickness)
    # cv2.line(img_tmp, right_average_line[:2], right_average_line[2:], [0, 255, 0], thickness)
    # cv2.line(img_tmp, left_average_line[:2], left_average_line[2:], [0, 255, 0], thickness)
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('Result tmp', img_tmp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


def get_slope(x1, y1, x2, y2):
    if x2 == x1:
        return float('inf')
    else:
        return (y2 - y1) / (x2 - x1)


def get_cross_point(x1, y1, x2, y2):
    m = get_slope(x1, y1, x2, y2)
    return (720 - y1) / m + x1


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)


def choose_lines(lines, min_slope_thr, max_slope_thr):  # 過濾斜率幾乎為平的線
    abs_slopes = np.abs([get_slope(*line) for line in lines])  # 獲得斜率數組
    chosen_indices = np.logical_and(min_slope_thr <= abs_slopes, abs_slopes <= max_slope_thr)
    lines = lines[chosen_indices]
    return lines

if __name__ == "__main__":
    input_dir = Path('input')
    output_dir = Path('output')
    # filename = input("請輸入欲辨識的影片檔名: ")
    file_name = 'test_video1.mp4'
    input_path = input_dir / file_name
    output_path = output_dir / file_name

    clip = VideoFileClip(str(input_path))
    out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(str(output_path), audio=False)
