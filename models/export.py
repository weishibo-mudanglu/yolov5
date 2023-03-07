"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python3 models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import os
import torch
import sys
sys.path.append('/data/liutianchi/code/yolov5-3.0/utils')
from utils.google_utils import attempt_download
from utils.general import set_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./runs/vehicle/exp3/weights/best.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()

    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt.batch_size)
    print(opt.img_size)
    set_logging()

    # # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size)).to(torch.device('cuda'))  # image size(1,3,320,192) iDetection


    # Load PyTorch model
    attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=torch.device('cuda'))['model'].float()
    model.eval()
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # TorchScript export
    # try:
    #     print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    #     f = my_weights.replace('.pt', '.torchscript.pt')  # filename
    #     ts = torch.jit.trace(model, img)
    #     ts.save(f)
    #     print('TorchScript export success, saved as %s' % f)
    # except Exception as e:
    #     print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.fuse()  # only for ONNX
        # torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
        #                   output_names=['classes', 'boxes'] if y is None else ['output'])# 原版
        torch.onnx.export(model, img, f, verbose=False, opset_version=10, input_names=['input'],
                          output_names=['yolo1', 'yolo2', 'yolo3'])
        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # CoreML export
    # try:
    #     import coremltools as ct
    #
    #     print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
    #     # convert model from torchscript and apply pixel scaling as per detect.py
    #     model = ct.convert(ts, inputs=[ct.ImageType(name='images', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
    #     f = my_weights.replace('.pt', '.mlmodel')  # filename
    #     model.save(f)
    #     print('CoreML export success, saved as %s' % f)
    # except Exception as e:
    #     print('CoreML export failure: %s' % e)

    # try:
    #     import tensorrt as trt
    #     TRT_LOGGER = trt.Logger()
    #     explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    #     onnx_file_path = my_weights.replace('.pt', '.onnx')
    #     engine_file_path = my_weights.replace('.pt', '.trt')
    #
    #     """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    #     with trt.Builder(TRT_LOGGER) as builder,builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    #         builder.max_workspace_size = 1 << 30  # 256MiB
    #         builder.max_batch_size = 1
    #         # Parse model file
    #         if not os.path.exists(onnx_file_path):
    #             print(
    #                 'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
    #             exit(0)
    #         print('Loading ONNX file from path {}...'.format(onnx_file_path))
    #         with open(onnx_file_path, 'rb') as model:
    #             print('Beginning ONNX file parsing')
    #             parser.parse(model.read())
    #         last_layer = network.get_layer(network.num_layers - 1)
    #         network.mark_output(last_layer.get_output(0))
    #         network.get_input(0).shape = [1, 3, 640, 640]
    #         print('Completed parsing of ONNX file')
    #         print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
    #
    #         print('network layers ', network.num_layers)
    #
    #         engine = builder.build_cuda_engine(network)
    #         print("Completed creating Engine")
    #         with open(engine_file_path, "wb") as f:
    #             f.write(engine.serialize())
    # except Exception as e:
    #     print('trt export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
