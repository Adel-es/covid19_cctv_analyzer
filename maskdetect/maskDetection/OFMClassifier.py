import tensorflow as tf
import numpy as np
#import wget
import os, sys 
directory = os.path.abspath(os.path.dirname(__file__))
pDirectory = os.path.dirname(directory)
sys.path.append(pDirectory)
import libs.fpsCalculator as FPST

class OFMClassifier:
    """
    Perform image classification with the given model. The model is a .h5 file
    which if the classifier can not find it at the path it will download it
    from neuralet repository automatically.
    :param config: Is a Config instance which provides necessary parameters.
    """

    def __init__(self):
        #tf.debugging.set_log_device_placement(True)
        self.gpu_number = 0; 
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[self.gpu_number], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[self.gpu_number], True)
            except RuntimeError as e:
                print(e)
        self.model_data = directory + "/OFMClassifier.h5" 
        print("model is at >> " + self.model_data)
        self.classifier_model = tf.keras.models.load_model(self.model_data)
        self.timer = FPST.FPSCalc() 
        self.fps = 0
        self.width = 45
        self.height = 45


    def inference(self, resized_rgb_image) -> list :
        '''
        Inference function sets input tensor to input image and gets the output.
        The interpreter instance provides corresponding class id output which is used for creating result
        Args:
            resized_rgb_image: Array of images with shape (no_images, img_height, img_width, channels)
        Returns:
            result: List of class id for each input image. ex: [0, 0, 1, 1, 0]
            scores: The classification confidence for each class. ex: [.99, .75, .80, 1.0]
        '''
        
        if np.shape(resized_rgb_image)[0] == 0:
            return [], []
        self.timer.start()
        output_dict = self.classifier_model.predict(resized_rgb_image)
        self.fps = self.timer.end() 
        
        result = list(np.argmax(output_dict, axis=1))  # returns class id
        scores = []
        for i, itm in enumerate(output_dict):
            scores.append(itm[result[i]])

        return result, scores
