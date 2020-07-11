import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2
import time
import glob
import random
import pandas as pd
from PIL import Image
import PIL
import io
import argparse
import sys
from openvino.inference_engine import IECore

class Model(object):

    def __init__(self):
        self.labels = []
        labels_filename = "labels.txt"

        # Create a list of labels.
#        with open(labels_filename, 'rt') as lf:
#            for l in lf:
#                self.labels.append(l.strip())

    def predict(self, imageFile):
        raise NotImplementedError
    
    def convert_to_opencv(self, image):
        # RGB -> BGR conversion is performed as well.
        image = image.convert('RGB')
        r,g,b = np.array(image).T
        opencv_image = np.array([b,g,r]).transpose()
        return opencv_image

    def crop_center(self, img,cropx,cropy):
        h, w = img.shape[:2]
        startx = w//2-(cropx//2)
        starty = h//2-(cropy//2)
        return img[starty:starty+cropy, startx:startx+cropx]

    def resize_down_to_1600_max_dim(self, image):
        h, w = image.shape[:2]
        if (h < 1600 and w < 1600):
            return image

        new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
        return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

    def resize_to_256_square(self, image):
        h, w = image.shape[:2]
        return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

    def update_orientation(self, image):
        exif_orientation_tag = 0x0112
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if (exif != None and exif_orientation_tag in exif):
                orientation = exif.get(exif_orientation_tag, 1)
                # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
                orientation -= 1
                if orientation >= 4:
                    image = image.transpose(Image.TRANSPOSE)
                if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

class TFModel(Model):

    def __init__(self, modelFilePath):
        super(TFModel, self).__init__()
        
        graph_def = tf.compat.v1.GraphDef()

        # These are set to the default names from exported models, update as needed.
        #filename = "api/models/resnet50/resnet50_fp32_pretrained_model.pb"
        filename = modelFilePath

        # Import the TF graph
        with tf.io.gfile.GFile(filename, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session()
                
    def predict(self, imageFile):
        start1 = time.time() #ここ追加

        # Load from a file
        image = Image.open(imageFile)

        # Update orientation based on EXIF tags, if the file has orientation info.
        image = super().update_orientation(image)

        # Convert to OpenCV format
        image = super().convert_to_opencv(image)

        # If the image has either w or h greater than 1600 we resize it down respecting
        # aspect ratio such that the largest dimension is 1600
        image = super().resize_down_to_1600_max_dim(image)

        # We next get the largest center square
        h, w = image.shape[:2]
        min_dim = min(w,h)
        max_square_image = super().crop_center(image, min_dim, min_dim)

        # Resize that square down to 256x256
        augmented_image = super().resize_to_256_square(max_square_image)

        # Get the input size of the model
        input_tensor_shape = self.sess.graph.get_tensor_by_name('input_1_1:0').shape.as_list()
        network_input_size = input_tensor_shape[1]

        # Crop the center for the specified network_input_Size
        augmented_image = super().crop_center(augmented_image, network_input_size, network_input_size)
        frame = augmented_image

        # These names are part of the model and cannot be changed.
        output_layer = 'dense_1_1/Softmax:0'
        input_node = 'input_1_1:0'

        try:
            prob_tensor = self.sess.graph.get_tensor_by_name(output_layer)
            start2 = time.time() #ここ追加
            predictions, = self.sess.run(prob_tensor, {input_node: [augmented_image] })
            infer_time = time.time() - start2
        except KeyError:
            print ("Couldn't find classification output layer: " + output_layer + ".")
            print ("Verify this a model exported from an Object Detection project.")
            exit(-1)

        # Print the highest probability label
        highest_probability_index = np.argmax(predictions)
        total_time = time.time() - start1
        
        #return total_time, infer_time, self.labels[highest_probability_index], frame  #ここ追加
        return total_time, infer_time, "", frame  #ここ追加

class OpenVINOModel(Model):

    def __init__(self, target_device, modelFilePath):
        super(OpenVINOModel, self).__init__()

        # These are set to the default names from exported models, update as needed.
        model_xml = modelFilePath
        model_bin = modelFilePath.replace('.xml', '.bin')

        # Plugin initialization for specified device and load extensions library if specified
        # Set the desired device name as 'device' parameter. This sample support these 3 names: CPU, GPU, MYRIAD
        ie = IECore()

        # Read IR
        self.net = ie.read_network(model=model_xml, weights=model_bin)

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1

        # Loading model to the plugin
        self.exec_net = ie.load_network(network=self.net, device_name='CPU', num_requests=1)

    def predict(self, imageFile):
        start1 = time.time() #ここ追加

        # Load from a file
        image = Image.open(imageFile)

        # Update orientation based on EXIF tags, if the file has orientation info.
        image = super().update_orientation(image)

        # Convert to OpenCV format
        image = super().convert_to_opencv(image)

        # If the image has either w or h greater than 1600 we resize it down respecting
        # aspect ratio such that the largest dimension is 1600
        image = super().resize_down_to_1600_max_dim(image)

        # We next get the largest center square
        h, w = image.shape[:2]
        min_dim = min(w,h)
        max_square_image = super().crop_center(image, min_dim, min_dim)

        # Resize that square down to 256x256
        augmented_image = super().resize_to_256_square(max_square_image)

        # Get the input size of the model
        n, c, h, w = self.net.inputs[self.input_blob].shape

        # Crop the center for the specified network_input_Size
        augmented_image = super().crop_center(augmented_image, w, h)
        frame = augmented_image

        #
        augmented_image = augmented_image.transpose((2, 0, 1))

        images = np.ndarray(shape=(n, c, h, w))
        images[0] = augmented_image

        start2 = time.time() #ここ追加
        predictions = self.exec_net.infer(inputs={self.input_blob: images})
        infer_time = time.time() - start2

        # Print the highest probability label
#        predictions = predictions[self.out_blob]
#        highest_probability_index = predictions[0].argsort()[-1:][::-1]

        total_time = time.time() - start1

        return total_time, infer_time, "", frame  #ここ追加


def run_inference(modelFile, model_type="tf", target_device='CPU', dataset_dir=".", total=500):
    if model_type == 'tf':
        model = TFModel(modelFile)
    elif model_type == 'tf_int8':
        model = TFModel(modelFile)
    else:
        if target_device == 'GPU':
            model = OpenVINOModel('GPU', modelFile)
        elif target_device == 'MYRIAD':
            model = OpenVINOModel('MYRIAD', modelFile)
        else:
            model = OpenVINOModel('CPU', modelFile)

    total_infer_spent_time = 0
    total_spent_time = 0
    list_df = pd.DataFrame( columns=['正解ラベル','予測ラベル','全処理時間(msec)','推論時間(msec)'] )

    file_list = glob.glob(os.path.join(dataset_dir, "*"))
    for i in range(total):
        img_path = random.choice(file_list)
        img_cat = os.path.split(os.path.dirname(img_path))[1]
        total_time, infer_time, pred_label, frame = model.predict(img_path)

        if i > 1:
            total_infer_spent_time += infer_time
            total_spent_time += total_time

        print(img_path, str(int(total_time*1000.0)) + 'msec', str(int(infer_time*1000.0)) + 'msec', pred_label) #ここ追加

        tmp_se = pd.Series( [img_cat, pred_label, str(int(total_time * 1000)), str(int(infer_time * 1000)) ], index=list_df.columns )
        list_df = list_df.append( tmp_se, ignore_index=True ) 

    print()
    print('全' + str(total) + '枚 完了！')
    print()
    print("平均処理時間: " + str(int((total_spent_time / (total-1))*1000.0)) + " ms/枚")
    print("平均推論時間: " + str(int((total_infer_spent_time / (total-1))*1000.0)) + " ms/枚")

    return int((total_spent_time / (total-1))*1000.0), int((total_infer_spent_time / (total-1))*1000.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_graph", default=None,
                        help="graph/model to be executed")
    parser.add_argument("--openvino", action='store_true',
                        help="Run on OpenVINO IE")
    parser.add_argument("--num_images",
                        help="Number of images to be infered",type=int, default=50)
    parser.add_argument("--dataset_dir",
                        help="Dataset dir", default=None)
    args = parser.parse_args()
    
    if args.input_graph:
        model_file = args.input_graph
    else:
        sys.exit("Please provide a graph file.")

    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        sys.exit("Please provide a dataset dir.")
    
    model_type = 'tf'
    if args.openvino:
        model_type = 'openvino'

    print('Starting inference...')
    tf_total_time, tf_infer_time = run_inference(model_file, model_type=model_type, dataset_dir=dataset_dir, total=args.num_images)