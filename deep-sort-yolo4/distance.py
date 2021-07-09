import math

def getCentroid(bbox, return_int=False):
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
    
def checkDistance(people, confirmed_case):
    '''
    Param
        people: list of (bbox, id) of everyone in the frame. 
        confirmed_case: (bbox, id) of confirmed case
    Return
        close_people: list of (bbox, id) close to the confirmed case including the confirmed case
    '''
    c_bbox, c_id = confirmed_case
    c_width = getBoundingBoxWidth(c_bbox)
    c_centroid = getCentroid(c_bbox)
    close_people = []
    for person in people:
        # Find the average bounding box width of two people
        bbox, id = person
        width = getBoundingBoxWidth(bbox)
        bbox_average_width = (c_width + width) / 2
        # Find the distance between centroids for two people
        centroid = getCentroid(bbox)
        distance = math.sqrt(math.pow((c_centroid[0] - centroid[0]), 2) + math.pow((c_centroid[1] - centroid[1]), 2)) 
        # Compare the bbox_average_width and distance to determine whether to put person in close_people
        is_close = distance <= 5 * bbox_average_width
        if is_close:
            close_people.append(person)
    return close_people
