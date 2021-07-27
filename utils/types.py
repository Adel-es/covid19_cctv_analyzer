from enum import Enum 

class TrackToken:
    def __init__(self, bbox, tid):
        self.bbox = bbox
        self.tid = tid

class MaskToken(Enum) : 
    Masked = 1 
    NotMasked = 2 
    NotNear = 3