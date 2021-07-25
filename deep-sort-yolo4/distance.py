import math

DISTANCE_BBOX_RATIO_CRITERIA = 5

def getCentroid(bbox, return_int=False):
    '''
    Returns the coordinate of the center point of the lower side of the bbox.
    If return_int is true, the type of return value is (int, int), otherwise, (float, float)
    '''
    start_x, start_y, end_x, end_y = bbox
    centroid_x = (start_x + end_x) / 2
    if return_int:
        centroid_x = int(centroid_x)
        end_y = int(end_y)
    centroid = (centroid_x, end_y)
    return centroid

def getBoundingBoxWidth(bbox):
    start_x, start_y, end_x, end_y = bbox
    width = end_x - start_x
    return width

def checkDistance(trackingRslt, reidRslt, distanceRslt):
    for aFrameTracking, aFrameReid in zip(trackingRslt, reidRslt):       
        aFrameDistance = []
        
        # If there is the confirmed case in this frame
        if aFrameReid != -1:
            confirmedIdx = aFrameReid
            confirmed_case = aFrameTracking[confirmedIdx]
        
            c_width = getBoundingBoxWidth(confirmed_case.bbox)
            c_centroid = getCentroid(confirmed_case.bbox)
            for person in aFrameTracking:
                # Find the average bounding box width of two people
                width = getBoundingBoxWidth(person.bbox)
                bbox_average_width = (c_width + width) / 2
                # Find the distance between centroids for two people
                centroid = getCentroid(person.bbox)
                distance = math.sqrt(math.pow((c_centroid[0] - centroid[0]), 2) + math.pow((c_centroid[1] - centroid[1]), 2)) 
                # Compare the bbox_average_width and distance to determine if the two people are close
                is_close = distance <= DISTANCE_BBOX_RATIO_CRITERIA * bbox_average_width
                if is_close:
                    aFrameDistance.append(True)
                else:
                    aFrameDistance.append(False)        
        distanceRslt.append(aFrameDistance)