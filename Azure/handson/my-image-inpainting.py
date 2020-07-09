import sys

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

# ----------------------------------------------------------------------------

g_mouseX=-1
g_mouseY=-1
g_mouseBtn = -1     # 0=left, 1=right, -1=none

g_UIState = 0       # 0: normal UI, 1: wait for a click
g_clickedFlag = False
g_inpaintFlag = False

g_penSize = 8
g_canvas = []
g_mask   = []

def clearMask():
    global g_canvas
    global g_mask
    g_mask = np.full(g_canvas.shape, [0,0,0], np.uint8)     # The size of the mask is the same as the canvas

# ----------------------------------------------------------------------------

def main():
    _H=0
    _W=1
    _C=2

    global g_canvas, g_mask
    global g_threshold
    global g_UIState
    global g_inpaintFlag
    global g_clickedFlag

    if len(sys.argv)<2:
        print('Please specify an input file', file=sys.stderr)
        return -1
    g_canvas = cv2.imread(sys.argv[1])

    ie = IECore()

    model='gmcnn-places2-tf'
    model = './public/'+model+'/FP16/'+model
    net = ie.read_network(model+'.xml', model+'.bin')
    input_blob1 = 'Placeholder'
    input_blob2 = 'Placeholder_1'
    out_blob    = 'Minimum'
    in_shape1   = net.inputs[input_blob1].shape   # 1,3,512,680
    in_shape2   = net.inputs[input_blob2].shape
    out_shape  = net.outputs[out_blob].shape      # 1,3,512,680
    exec_net = ie.load_network(net, 'CPU')

    clearMask()
	
	img = g_canvas | g_mask
	img = cv2.resize(img, (in_shape1[3], in_shape1[2]))
	img = img.transpose((_C, _H, _W))
	img = img.reshape(in_shape1)

	msk = cv2.resize(g_mask, (in_shape2[3], in_shape2[2]))
	msk = msk.transpose((_C, _H, _W))
	msk = msk[0,:,:]
	msk = np.where(msk>0., 1., 0.).astype(np.float32)
	msk = msk.reshape(in_shape2)

	res = exec_net.infer(inputs={input_blob1: img, input_blob2: msk})

	out = np.transpose(res[out_blob], (0, 2, 3, 1)).astype(np.uint8)
	out = cv2.cvtColor(out[0], cv2.COLOR_RGB2BGR)

	cv2.imwrite('after_'+sys.argv[1], out)

    return 0

if __name__ == '__main__':
    sys.exit(main())

