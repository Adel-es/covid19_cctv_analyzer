from . import OFMClassifier

class Classifier:
    """
    Classifier class is a high-level class for classifying images using x86 devices.
    When an instance of the Classifier is created you can call inference method and feed your
    input image in order to get the classifier results.
    :param config: Is a Config instance which provides necessary parameters.
    """
    def __init__(self):
        self.net = OFMClassifier.OFMClassifier() 
        self.width = self.net.width 
        self.height = self.net.height 
        
    def inference(self, resized_rgb_image):
        self.fps = self.net.fps
        output, scores = self.net.inference(resized_rgb_image)
        return output, scores
