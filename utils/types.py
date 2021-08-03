from enum import Enum 

class TrackToken:
    def __init__(self, bbox, tid):
        self.bbox = bbox
        self.tid = tid

class MaskToken(Enum) : 
    Masked = 1 
    NotMasked = 2 
    FaceNotFound = 3
    NotNear = 4
    UnKnown = 5 