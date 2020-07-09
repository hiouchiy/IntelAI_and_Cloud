#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from openvino.inference_engine import IECore
import cv2 as cv
import numpy as np
import os
from argparse import ArgumentParser, SUPPRESS
import logging as log
import sys


def build_arg():
    parser = ArgumentParser(add_help=False)
    in_args = parser.add_argument_group('Options')
    in_args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Help with the script.')
    in_args.add_argument("-m", "--model", help="Required. Path to .xml file with pre-trained model.",
                         required=True, type=str)
    in_args.add_argument("--coeffs", help="Required. Path to .npy file with color coefficients.",
                         required=True, type=str)
    in_args.add_argument("-d", "--device",
                         help="Optional. Specify target device for infer: CPU, GPU, FPGA, HDDL or MYRIAD. "
                              "Default: CPU",
                         default="CPU", type=str)
    in_args.add_argument('-i', "--input",
                         help='Required. Input to process.',
                         required=True, type=str, metavar='"<path>"')
    in_args.add_argument("-v", "--verbose", help="Optional. Enable display of processing logs on screen.",
                         action='store_true', default=False)
    return parser


if __name__ == '__main__':
    args = build_arg().parse_args()
    coeffs = args.coeffs
    
    # mean is stored in the source caffe model and passed to IR
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug("Load network")
    ie = IECore()
    load_net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")
    load_net.batch_size = 1
    exec_net = ie.load_network(network=load_net, device_name="CPU")

    assert len(load_net.inputs) == 1, "Expected number of inputs is equal 1"
    input_blob = next(iter(load_net.inputs))
    input_shape = load_net.inputs[input_blob].shape
    assert input_shape[1] == 1, "Expected model input shape with 1 channel"

    assert len(load_net.outputs) == 1, "Expected number of outputs is equal 1"
    output_blob = next(iter(load_net.outputs))
    output_shape = load_net.outputs[output_blob].shape
    assert output_shape == [1, 313, 56, 56], "Shape of outputs does not match network shape outputs"

    _, _, h_in, w_in = input_shape

    try:
        input_source = int(args.input)
    except ValueError:
        input_source = args.input

    color_coeff = np.load(coeffs).astype(np.float32)
    assert color_coeff.shape == (313, 2), "Current shape of color coefficients does not match required shape"

    log.debug("#############################")

    original_frame = cv.imread(input_source)
    (h_orig, w_orig) = original_frame.shape[:2]

    log.debug("Preprocessing frame")
    if original_frame.shape[2] > 1:
        frame = cv.cvtColor(cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
    else:
        frame = cv.cvtColor(original_frame, cv.COLOR_GRAY2RGB)

    img_rgb = frame.astype(np.float32) / 255
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    img_l_rs = cv.resize(img_lab.copy(), (w_in, h_in))[:, :, 0]

    log.debug("Network inference")
    res = exec_net.infer(inputs={input_blob: [img_l_rs]})

    update_res = (res[output_blob] * color_coeff.transpose()[:, :, np.newaxis, np.newaxis]).sum(1)

    log.debug("Get results")
    out = update_res.transpose((1, 2, 0))
    out = cv.resize(out, (w_orig, h_orig))
    img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)
    
    imshowSize = (w_orig, h_orig)
    colorize_image = (cv.resize(img_bgr_out, imshowSize) * 255).astype(np.uint8)
    
    cv.imwrite('colorized_'+input_source, colorize_image)
