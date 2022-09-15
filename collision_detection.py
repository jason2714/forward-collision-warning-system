from pathlib import Path
from argparse import ArgumentParser
from detect import car_detection
from utils.general import print_args
import cv2 as cv
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from lane_crop import process_image, line_intersection
from thread_with_return_value import ThreadWithReturnValue
import timeit
from tqdm import trange


def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--input_path', type=Path, default='./data/videos/test_video1.mp4', help='path of input video')
    parser.add_argument('--output_dir', type=Path, default='./output/', help='directory of output video')
    parser.add_argument('--warning_ratio', type=float, default=1 / 25,
                        help='warning ratio of the car size in the frame')

    # options for yolo
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='car_detection', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    opt, unknown = parser.parse_known_args()
    output_path = opt.output_dir / opt.input_path.name

    parser.set_defaults(output_path=output_path)
    parser.set_defaults(source=opt.input_path)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.device = select_device(opt.device)
    print_args(vars(opt))
    return opt


def point_line_xdiff(point, line):
    horizon_line = [point[0] - 10, point[1], *point]
    horizon_cross_point = line_intersection(horizon_line, line)
    assert horizon_cross_point[1] == point[1], "y of horizon_cross_point must be the same as y of point"
    return point[0] - horizon_cross_point[0]


def is_in_lane(det, left_lane, right_lane):
    left_bottom_point = [det[0], det[3]]
    right_bottom_point = det[2:]
    left_xdiff = point_line_xdiff(left_bottom_point, left_lane)
    right_xdiff = point_line_xdiff(right_bottom_point, right_lane)
    if left_xdiff >= 0 and right_xdiff <= 0:
        return True
    return False


def cal_car_frame_ratio(det, frame_shape):
    if len(frame_shape) == 3:
        frame_shape = frame_shape[:2]
    det_size = (det[2] - det[0]) * (det[3] - det[1])
    frame_size = frame_shape[0] * frame_shape[1]
    return det_size / frame_size


def main():
    opt = parse_opt()
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    model = DetectMultiBackend(opt.weights, device=opt.device, dnn=opt.dnn, data=opt.data, fp16=opt.half)
    cap = cv.VideoCapture(str(opt.input_path))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    output = cv.VideoWriter(str(opt.output_path), fourcc, fps, (frame_width, frame_height))
    num_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(opt.warning_ratio)
    start = timeit.default_timer()

    last_detections = []
    last_count = 5
    for idx in trange(num_frame, desc='Car Distance Detecting... '):
        if not cap.isOpened():
            break
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame or stream end. Exiting ...")
            break
        car_det_thread = ThreadWithReturnValue(target=car_detection,
                                               args=(opt, model, frame.copy()))
        lane_det_thread = ThreadWithReturnValue(target=process_image, args=(frame,))
        car_det_thread.start()
        lane_det_thread.start()
        detections = car_det_thread.join()
        if detections is None:
            detections = []
        result, left_lane, right_lane = lane_det_thread.join()
        detections_in_lane = []
        for det in detections:
            det, conf, cls = det[:4].copy(), det[4], det[5]
            det = list(map(int, det))
            if is_in_lane(det, left_lane, right_lane):
                detections_in_lane.append(det)
        if not detections_in_lane:
            last_count -= 1
            if last_count > 0:
                detections_in_lane = last_detections
        else:
            last_count = 5
            last_detections = detections_in_lane
        for det in detections_in_lane:
            car_frame_ratio = cal_car_frame_ratio(det, frame.shape)
            if car_frame_ratio >= opt.warning_ratio:
                # print(idx, 'warning', conf)
                bbox_color = (0, 0, 255)
                cv.putText(result, 'WARNING', (det[0], det[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                           1.5, bbox_color, 5, cv.LINE_AA)
            else:
                bbox_color = (255, 0, 0)
                # cv.putText(result, f'{car_frame_ratio:.4f}', (det[0], det[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                #            0.75, bbox_color, 2, cv.LINE_AA)
            cv.rectangle(result, det[:2], det[2:], bbox_color, 3)
        output.write(result)
        # cv.imshow('frame', result)
        # if cv.waitKey(1) == ord('q'):
        #     break

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    cap.release()
    output.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
