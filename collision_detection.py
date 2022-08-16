from pathlib import Path
from argparse import ArgumentParser
from detect import car_detection
from utils.general import print_args
import cv2 as cv
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from lane_crop import process_image
from thread_with_return_value import ThreadWithReturnValue
import timeit
from tqdm import trange

def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--input_path', type=Path, default='./data/videos/test_video1.mp4', help='path of input video')
    parser.add_argument('--output_dir', type=Path, default='./output/', help='directory of output video')

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


def main():
    opt = parse_opt()
    model = DetectMultiBackend(opt.weights, device=opt.device, dnn=opt.dnn, data=opt.data, fp16=opt.half)
    cap = cv.VideoCapture(str(opt.input_path))
    num_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    start = timeit.default_timer()

    for idx in trange(num_frame, desc='Car Distance Detecting... '):
        if not cap.isOpened():
            break
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame or stream end. Exiting ...")
            break
        car_det_thread = ThreadWithReturnValue(target=car_detection,
                                               args=(opt, model, frame))
        lane_det_thread = ThreadWithReturnValue(target=process_image, args=(frame,))
        car_det_thread.start()
        lane_det_thread.start()
        det = car_det_thread.join()
        lane_det_thread.join()
        # cv.imshow('frame', frame)
        # if cv.waitKey(1) == ord('q'):
        #     break

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
