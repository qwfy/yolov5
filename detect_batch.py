import argparse
import csv
import os.path
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, is_ascii, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


def parse_opt(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model path(s)')
    parser.add_argument('--source', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--fps', type=int, default=5, help='read the source in FPS if it is a video file')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='show results')
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=False, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=False, action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default=False, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize features')
    parser.add_argument('--output-dir', default='run', help='where to save results')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', default=False, action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--bs', type=int, default=160, help='batch size')
    opt = parser.parse_args(argv)
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

@torch.no_grad()
def run(weights,
        source,
        fps,
        img_size,
        conf_thres,
        iou_thres,
        max_det,
        device,
        view_img,
        save_txt,
        save_conf,
        save_crop,
        nosave,
        classes,
        agnostic_nms,
        augment,
        visualize,
        output_dir,
        line_thickness,
        hide_labels,
        hide_conf,
        half,
        bs,
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_img:
        (save_dir / 'images').mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    else:
        assert False, 'Only pytorch is supported'
    img_size = check_img_size(img_size, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    dataset = LoadImages(source, img_size=img_size, stride=stride, auto=pt, fps=fps)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.parameters())))  # run once

    total_batches, rem = divmod(len(dataset), bs)
    if rem > 0:
        total_batches += 1
    pbar = tqdm(total=(len(dataset)))
    dataset_iter = iter(dataset)
    for i_batch in range(total_batches):
        i_batch += 1
        # accumulate one batch
        batch_paths = []
        batch_imgs = []
        batch_im0ss = []
        batch_vid_caps = []
        batch_frames = []
        num_samples = bs if i_batch < total_batches else (rem if rem != 0 else bs)
        for _ in range(num_samples):
            path, img, im0s, vid_cap, frame = next(dataset_iter)
            batch_paths.append(path)
            batch_imgs.append(img)
            batch_im0ss.append(im0s)
            batch_vid_caps.append(vid_cap)
            batch_frames.append(frame)
            del path, img, im0s, vid_cap, frame

        batch_imgs = np.array(batch_imgs)
        assert batch_imgs.shape[:2] == (num_samples, 3)
        batch_imgs = torch.from_numpy(batch_imgs).to(device)
        batch_imgs = batch_imgs.half() if half else batch_imgs.float()  # uint8 to fp16/32
        batch_imgs = batch_imgs / 255.0  # 0 - 255 to 0.0 - 1.0
        visualize = increment_path(save_dir / Path(batch_paths[0]).stem, mkdir=True) if visualize else False
        batch_preds = model(batch_imgs, augment=augment, visualize=visualize)[0]

        for i in range(len(batch_paths)):
            pbar.update(1)

            path = batch_paths[i]
            img = batch_imgs[i]
            im0s = batch_im0ss[i]
            vid_cap = batch_vid_caps[i]
            pred = batch_preds[i]
            pred = torch.unsqueeze(pred, 0)
            frame = batch_frames[i]

            img = torch.unsqueeze(img, 0)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                p, s, im0 = path, '', im0s.copy()

                p = Path(p)  # to Path
                save_path = str(save_dir / 'images' / p.name)  # img.jpg
                txt_path = str(save_dir / 'label.csv')
                if dataset.mode == 'image':
                    txt_line_lead = p
                else:
                    txt_line_lead = str(p) + f'_{frame}'
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    label_rows = []
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = [*xywh, conf] if save_conf else [*xywh]  # label format
                            label_rows.append([str(txt_line_lead), int(cls)] + [float(x) for x in line])

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    if save_txt:
                        with open(txt_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerows(label_rows)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)


def main(argv):
    opt = parse_opt(argv)
    run(**vars(opt))


if __name__ == '__main__':
    main(sys.argv[1:])
