from __future__ import print_function
import cv2
import numpy as np
import sys
import time
import pyzbar.pyzbar as pyzbar
from PIL import Image, ImageFilter
import pandas as pd
import random
from shapely.geometry import Polygon, Point


class Furniture():
    def __init__(self, conf, length, width, support, polygon, name):
        self.conf = Point(conf)
        self.width = width
        self.length = length
        self.support = support
        self.name = name
        self.polygon = Polygon([ [self.conf.x-self.length/2*np.cos(self.conf.z)-self.width/2*np.sin(self.conf.z),self.conf.y-self.length/2*np.sin(self.conf.z)+self.width/2*np.cos(self.conf.z)],
                                    [self.conf.x-self.length/2*np.cos(self.conf.z)+self.width/2*np.sin(self.conf.z),self.conf.y-self.length/2*np.sin(self.conf.z)-self.width/2*np.cos(self.conf.z)],
                                    [self.conf.x+self.length/2*np.cos(self.conf.z)+self.width/2*np.sin(self.conf.z),self.conf.y+self.length/2*np.sin(self.conf.z)-self.width/2*np.cos(self.conf.z)],
                                    [self.conf.x+self.length/2*np.cos(self.conf.z)-self.width/2*np.sin(self.conf.z),self.conf.y+self.length/2*np.sin(self.conf.z)+self.width/2*np.cos(self.conf.z)] ]) # If all objects are rectangles
        # self.polygon = Polygon(np.squeeze(polygon)) # If not all objects are rectangles

class Light():
    def __init__(self, point, intensity):
        self.point = Point(point)
        self.intensity = intensity

class Room():
    def __init__(self, polygon, surfaceRisk, name):
        self.polygon = Polygon(polygon)
        self.surfaceRisk = surfaceRisk
        self.name = name
        print(self.name)
        print(self.polygon)

def find_meter_from_pixel(pixel):
    unit_size_px = float(50)
    unit_size_m = 0.10
    x = (pixel* unit_size_m)/unit_size_px
    return x

def decode(im) :
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im)

  return decodedObjects

# Display barcode and QR code location
def display(im, decodedObjects):

  # Loop over all decoded objects
  for decodedObject in decodedObjects:
    points = decodedObject.polygon

    # If the points do not form a quad, find convex hull
    if len(points) > 4 :
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
      hull = list(map(tuple, np.squeeze(hull)))
    else :
      hull = points;

    # Number of points in the convex hull
    n = len(hull)

    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (100,100,0), 5)

  # Display results
  im_s = cv2.resize(im, (1500, 1000))
  cv2.imshow("Results", im_s);
  cv2.waitKey(0);

def find_rooms(img, noise_removal_threshold=10000, corners_threshold=0.1,
               room_closing_max_length=20000, gap_in_wall_threshold=20000):
    """

    :param img: grey scale image of rooms, already eroded and doors removed etc.
    :param noise_removal_threshold: Minimal area of blobs to be kept.
    :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    :param room_closing_max_length: Maximum line length to add to close off open doors.
    :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    :return: rooms: list of numpy arrays containing boolean masks for each detected room
             colored_house: A colored version of the input image, where each room has a random color.
    """
    assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal
    img[img < 80] = 0
    img[img > 80] = 255

    # Mark the outside of the house as black
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [biggest_contour], 255)
    img[mask == 0] = 0


    # Find the connected components in the house
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    indexes_group = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1]
    stats_old = stats.copy()
    stats = stats[indexes_group]
    r_area = []
    rooms = []
    for component_id, stat in zip(indexes_group, stats):
	    # print(component_id)
	    # print(stat)
	    component = (labels == component_id)
	    if img[component].sum() == 0 or np.count_nonzero(component) < gap_in_wall_threshold:
	           color = 0
	    else:
	           r_area.append(component_id)
	           color = np.random.randint(0, 255, size=3)
	    img[component] = color
    # print(r_area)
    main_room = r_area[0]
    bathroom = r_area[1]
    # print(stats_old[main_room])
    # print(stats_old[bathroom])
    main_room_center = (int(stats_old[main_room][0]+stats_old[main_room][2]/3),int(stats_old[main_room][1]+stats_old[main_room][3]/2))
    bathroom_center = (int(stats_old[bathroom][0]+stats_old[bathroom][2]/3),int(stats_old[bathroom][1]+stats_old[bathroom][3]/2))
    # cv2.putText(img,'main_room', main_room_center, cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 3, cv2.LINE_AA)
    # cv2.putText(img,'bathroom', bathroom_center, cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 3, cv2.LINE_AA)
    contour1 = np.array([[stats_old[main_room][0],stats_old[main_room][1]],[stats_old[main_room][0]+stats_old[main_room][2],stats_old[main_room][1]],[stats_old[main_room][0]+stats_old[main_room][2],stats_old[main_room][1]+stats_old[main_room][3]],[stats_old[main_room][0],stats_old[main_room][1]+stats_old[main_room][3]]], dtype=np.int32)
    # contour2 = np.array([[stats_old[bathroom][0],stats_old[bathroom][1]],[stats_old[bathroom][0]+stats_old[bathroom][2],stats_old[bathroom][1]],[stats_old[bathroom][0]+stats_old[bathroom][2],stats_old[bathroom][1]+stats_old[bathroom][3]],[stats_old[bathroom][0],stats_old[bathroom][1]+stats_old[bathroom][3]]], dtype=np.int32)
    # contour1 = np.array([[4.202000000000001, 0.764], [7.322, 0.764], [7.322, 1.264], [7.622, 3.288], [7, 3.388], [7, 4.088], [6.4, 4.088], [6.4, 4.988], [4.702000000000001, 4.988], [4.702000000000001, 3.988], [4.202000000000001, 1.264]], dtype=np.float32) # for A-K-Inboard-Headwall
    # contour2 = np.array([[4.702000000000001, 4.988], [3.402000000000001, 4.988], [3.402000000000001, 4.088], [4.02000000000001, 4.088], [4.002000000000001, 3.388], [4.502000000000001, 3.288], [4.702000000000001, 3.988]], dtype=np.float32) # for A-K-Inboard-Headwall
    # contour2 = np.array([[3.46, 2.424], [4.97, 2.424], [4.97, 3.2], [4.5, 3.838], [3.46, 3.838], [3.46, 2.424]], dtype=np.float32) # for A-K-Outboard-Footwall & Room-1
    contour2 = np.array([[3.464, 1.968], [5.098, 1.968], [5.098, 2.832], [4.298, 3.932], [3.464, 3.932], [3.464, 1.968]], dtype=np.float32) # for P22-Inboard-Footwall & Room-2
    # contour2 = np.array([[4.746, 1.968], [6.244000000000001, 1.968], [6.244000000000001, 3.862], [5.346, 3.862], [4.746, 2.862], [4.746, 1.968]], dtype=np.float32) # for P22-Outboard-Footwall
    # contour1 = np.array([[3.366, 2.88], [4.968, 2.88], [4.968, 3.6], [6.37, 5.2], [6.37, 8], [6, 8.3], [6, 8.54], [4.8, 8.54], [4.8, 7.8], [3.366, 7.8], [3.366, 2.88]], dtype=np.float32) # for S-B Outboard-Footwall
    # contour2 = np.array([[4.968, 2.88], [6.378, 2.88], [6.378, 4.558], [5.7, 5], [4.968, 3.6], [4.968, 2.88]], dtype=np.float32) # for S-B Outboard-Footwall
    # contour1 = np.array([[3.142, 2.376], [5.4, 2.376], [5.7, 2.776], [6, 2.556], [6.37, 3.276], [6.37, 4.076], [5.6, 4.4], [6, 5], [6.37, 5], [6.37, 6.8], [4.8, 7.524], [3.142, 7.524], [3.142, 2.376]], dtype=np.float32) # for J-M Outboard-Footwall
    # contour2 = np.array([[4.662, 2.376], [5.4, 2.376], [5.7, 2.776], [6, 2.556], [6.37, 3.276], [6.37, 4.076], [5.6, 4.4], [4.662, 3.2], [4.662, 2.376]], dtype=np.float32) # for J-M Outboard-Footwall
    # contour1 = np.array([[4.154, 0.764], [7.4, 0.764], [7.4, 1.2], [7.8, 1.2], [8.2, 3.3] , [7.6, 3.66], [7.4, 3.2], [6.8, 4.2], [6.8, 5.88], [4.8, 5.88], [4.8, 4], [4.154, 0.764]], dtype=np.float32) # for J-C Inboard-Headwall
    # contour2 = np.array([[3.176, 4.4], [3.5, 3.5], [4.2, 4.2], [4.7, 4.2], [4.8, 4.4], [4.8, 5.88], [3.6, 5.88], [3.6, 5.3], [3.176, 5.3], [3.176, 4.4]], dtype=np.float32) # for J-C Inboard-Headwall
    # contour1 = np.array([[4.154, 0.764], [7.8, 0.764], [8.2, 3.3] , [7.6, 3.66], [7.4, 3.2], [6.8, 4.2], [6.8, 5.88], [4.8, 5.88], [4.8, 4], [4.154, 0.764]], dtype=np.float32) # for Room-3 Inboard-Headwall
    # contour2 = np.array([[3.176, 4.4], [3.5, 3.5], [4.2, 4.2], [4.7, 4.2], [4.8, 4.4], [4.8, 5.88], [3.6, 5.88], [3.6, 5.3], [3.176, 5.3], [3.176, 4.4]], dtype=np.float32) # for Room-3 Inboard-Headwall
    # contour1 = np.array([[4.146, 0.764], [7.384, 0.764], [7.384, 4.2], [6.8, 4.2], [6.8, 5.504], [5.4, 5.504], [5.4, 4.114], [4.146, 4.114], [4.146, 0.764]], dtype=np.float32) # for J-G Headwall-Adjacent
    # contour2 = np.array([[3.474, 4.114], [5.358, 4.114], [5.358, 5.3], [4.8, 5.3], [4.9, 5.664], [4.2, 5.664], [4.15, 5.3], [3.474, 5.3], [3.474, 4.114]], dtype=np.float32) # for J-G Headwall-Adjacent
    # contour1 = np.array([[4.146, 0.764], [7.384, 0.764], [7.384, 4.2], [4.146, 4.2], [4.146, 0.764]], dtype=np.float32) # for B-L Inboard-Adjacent
    # contour2 = np.array([[3.738, 4.178], [5.342, 4.178], [5.342, 5.402], [4, 5.5], [3.9, 4.8], [3.75, 4.8], [3.738, 4.178]], dtype=np.float32) # for B-L Inboard-Adjacent
    # contour1 = np.array([[3.412, 1.93], [6.59, 1.93], [6.59, 7.14], [3.412, 7.14], [3.412, 1.93]], dtype=np.float32) # for K-B Inboard-Corner
    # contour2 = np.array([[3.412, 1.968], [5.016, 1.968], [5.016, 4.034], [3.412, 4.034], [3.412, 1.968]], dtype=np.float32) # for K-B Inboard-Corner
    # contour1 = np.array([[3.8, 1.642], [5, 1.642], [5.2, 3.8], [6, 3.7], [6.434, 7.14], [3.6, 7.14], [3.2, 3.6], [3.8, 3.4], [3.8, 1.642]], dtype=np.float32) # for K-B Inboard-Corner-Canted
    # contour2 = np.array([[5.074, 1.672], [6.716, 1.672], [6.716, 3.736], [5.074, 3.736], [5.074, 1.672]], dtype=np.float32) # for K-B Inboard-Corner-Canted
    # contour1 = np.array([[4.136, 0.764], [7.384, 0.764], [7.384, 3.7], [7.2, 4.2], [6, 4.4], [6.4, 5.4], [4.748, 6], [4.136, 4], [4.136, 0.764]], dtype=np.float32) # for B-JH Inboard-Headwall
    # contour2 = np.array([[3.4, 4.6], [4.136, 4], [4.748, 6], [3.7, 6.404], [3.4, 4.6]], dtype=np.float32) # for for B-JH Inboard-Headwall

    # if you are using obtained borders, use these lines:
    width_t = stats_old[main_room][2]+stats_old[bathroom][2]
    length_t = stats_old[main_room][3]+stats_old[bathroom][3]

    # if you are using costom borders, use these lines:
    rooms.append(Room(find_meter_from_pixel(contour1), 1, 'main_room'))
    # rooms.append(Room(find_meter_from_pixel(contour2), 1.05, 'bathroom'))
    # rooms.append(Room(contour1, 1, 'main_room'))
    rooms.append(Room(contour2, 1.05, 'bathroom'))

    return rooms, width_t, length_t, img

def find_objects(img, decodedObjects, object_library, noise_removal_threshold=5000, corners_threshold=0.1,
               room_closing_max_length=1500, gap_in_wall_threshold=1500):
    """

    :param img: grey scale image of rooms, already eroded and doors removed etc.
    :param noise_removal_threshold: Minimal area of blobs to be kept.
    :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    :param room_closing_max_length: Maximum line length to add to close off open doors.
    :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    :return: rooms: list of numpy arrays containing boolean masks for each detected room
             colored_house: A colored version of the input image, where each room has a random color.
    """
    assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal
    img[img == 0] = 255
    img[img < 200] = 0
    img[img > 200] = 255

    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)

    objects = contours
    assigned_objects = []
    sample_zones = {}
    lights = []
    doors = []
    for QR_obj in decodedObjects:

        QR_center = find_QR_center(QR_obj)
        # print(QR_center)
        for obj in objects:
            if cv2.pointPolygonTest(obj, QR_center, False) == 1:
                rect = cv2.minAreaRect(obj) # ((x_c, y_c), (w, h), angle)
                object_code = ([int(s) for s in QR_obj.data.split() if s.isdigit()])
                obj_df = object_library.loc[object_library['Object Code'] == object_code[0]]
                if "Light" in QR_obj.data.decode("utf-8"):
                    new_light = Light(Point([find_meter_from_pixel(rect[0][0]),find_meter_from_pixel(rect[0][1])]), obj_df.iloc[0]['Light Intensity'])
                    lights.append(new_light)
                elif "Door" in QR_obj.data.decode("utf-8"):
                    new_obj = Furniture([find_meter_from_pixel(rect[0][0]), find_meter_from_pixel(rect[0][1]), rect[2]+0.01], find_meter_from_pixel(rect[1][0]), find_meter_from_pixel(rect[1][1]), obj_df.iloc[0]['Support Level'], find_meter_from_pixel(obj), QR_obj.data.decode("utf-8"))
                    doors.append(new_obj)
                    # print(new_obj.name)
                    # print(new_obj.conf)
                    if "11" in QR_obj.data.decode("utf-8") or "35" in QR_obj.data.decode("utf-8"):
                        sample_zones["Main Door"] = new_obj.polygon
                else:
                    new_obj = Furniture([find_meter_from_pixel(rect[0][0]), find_meter_from_pixel(rect[0][1]), rect[2]+0.01], find_meter_from_pixel(rect[1][0]), find_meter_from_pixel(rect[1][1]), obj_df.iloc[0]['Support Level'], find_meter_from_pixel(obj), QR_obj.data.decode("utf-8"))
                    assigned_objects.append(new_obj)
                    # print(new_obj.name)
                    # print(new_obj.conf)

                    # Numbers to be changed to detect sittable side of objects:
                    #Numbers for A-K-Inboard-Headwall : [4, (2,4), 3, 2, 2, 2, 4]
                    #Numbers for A-K-Outboard-Footwall: [1, (2,4), 1, 2, 2, 2, 2]
                    #Numbers for P22-Inboard-footwall : [2, (2,4), 2, 1, 2, 2, 1]
                    #Numbers for P22-Outboard-footwall: [2, (2,4), 2, 1, 2, 2, 3]
                    #Numbers for P22-Nested           : [1, (2,4), 2, 1, 2, 2, 2]
                    #Numbers for S-B-Outboard-Footwall: [3, (2,4), 3, 2, 2, 2, 2]
                    #Numbers for J-M-Outboard-Footwall: [1, (2,4), 3, 2, 2, 2, 4]
                    #Numbers for J-C-Outboard-Footwall: [2, (2,4), 3, 2, 2, 2, 4]
                    #Numbers for J-G-Outboard-Footwall: [3, (1,3), 1, 2, 2, 2, 1]
                    #Numbers for B-L-Outboard-Footwall: [1, (1,3), 1, 2, 2, 2, 4]
                    #Numbers for B-JH-Inboard-Headwall: [2, (1,3), 1, 1, 1, 2, 4]
                    #Numbers for K-B-Inboard-Corner   : [1, (2,4), 1, 1, 4, 2, 2]
                    #Numbers for K-B-Canted-Corner    : [1, (2,4), 1, 1, 4, 2, 2]
                    #Numbers for Room-1               : [1, (2,4), 3, 2, 2, 2, 2]
                    #Numbers for Room-2               : [2, (2,4), 3, 4, 2, 2, 1]
                    #Numbers for Room-3               : [3, (2,4), 2, 2, 2, 2, 1]
                    #Numbers for Room-4               : [4, (2,4), 1, 2, 2, 2, 3]

                    sides = []
                    zone = [find_meter_from_pixel(rect[0][0]), find_meter_from_pixel(rect[0][1]), rect[2]+0.01, find_meter_from_pixel(rect[1][1]), find_meter_from_pixel(rect[1][0])]
                    if "Toilet" in QR_obj.data.decode("utf-8"):
                        sides = [2]
                    if "Bed" in QR_obj.data.decode("utf-8"):
                        sides = [2,4]
                    if "Chair-Patient" in QR_obj.data.decode("utf-8"):
                        sides = [3]
                    if "Chair-Visitor" in QR_obj.data.decode("utf-8"):
                        sides = [4]
                    if "Sofa" in QR_obj.data.decode("utf-8"):
                        sides = [2]
                    if "Couch" in QR_obj.data.decode("utf-8"):
                        sides = [2]
                    if "Sink-Bath" in QR_obj.data.decode("utf-8"):
                        sides = [1]

                    l = 0.3
                    for side in sides:
                        zone = [find_meter_from_pixel(rect[0][0]), find_meter_from_pixel(rect[0][1]), rect[2]+0.01, find_meter_from_pixel(rect[1][1]), find_meter_from_pixel(rect[1][0])]
                        if side == 1:
                            zone[4] += l
                            zone[0] += float(l)/2*np.cos(zone[2])
                            zone[1] += float(l)/2*np.sin(zone[2])
                        if side == 2:
                            zone[3] += l
                            zone[0] -= float(l)/2*np.sin(zone[2])
                            zone[1] += float(l)/2*np.cos(zone[2])
                        if side == 3:
                            zone[4] += l
                            zone[0] -= float(l)/2*np.cos(zone[2])
                            zone[1] -= float(l)/2*np.sin(zone[2])
                        if side == 4:
                            zone[3] += l
                            zone[0] += float(l)/2*np.sin(zone[2])
                            zone[1] -= float(l)/2*np.cos(zone[2])
                        corners_sitting = Polygon([[zone[0]-float(zone[4])/2*np.cos(zone[2])-float(zone[3])/2*np.sin(zone[2]),zone[1]-float(zone[4])/2*np.sin(zone[2])+float(zone[3])/2*np.cos(zone[2])],
                                                    [zone[0]-float(zone[4])/2*np.cos(zone[2])+float(zone[3])/2*np.sin(zone[2]),zone[1]-float(zone[4])/2*np.sin(zone[2])-float(zone[3])/2*np.cos(zone[2])],
                                                    [zone[0]+float(zone[4])/2*np.cos(zone[2])+float(zone[3])/2*np.sin(zone[2]),zone[1]+float(zone[4])/2*np.sin(zone[2])-float(zone[3])/2*np.cos(zone[2])],
                                                    [zone[0]+float(zone[4])/2*np.cos(zone[2])-float(zone[3])/2*np.sin(zone[2]),zone[1]+float(zone[4])/2*np.sin(zone[2])+float(zone[3])/2*np.cos(zone[2])] ])

                        if "Toilet" in QR_obj.data.decode("utf-8"):
                            sample_zones["Toilet"] = corners_sitting
                        elif "Chair-Patient" in QR_obj.data.decode("utf-8"):
                            sample_zones["Chair-Patient"] = corners_sitting
                        elif "Chair-Visitor" in QR_obj.data.decode("utf-8"):
                            sample_zones["Chair-Visitor"] = corners_sitting
                        elif "Sofa" in QR_obj.data.decode("utf-8"):
                            sample_zones["Sofa"] = corners_sitting
                        elif "Couch" in QR_obj.data.decode("utf-8"):
                            sample_zones["Couch"] = corners_sitting
                        elif "Sink-Bath" in QR_obj.data.decode("utf-8"):
                            sample_zones["Sink-Bath"] = corners_sitting
                        elif "Bed" in QR_obj.data.decode("utf-8"):
                            if side == sides[0]:
                                sample_zones["Bed_L"] = corners_sitting
                            elif side == sides[1]:
                                sample_zones["Bed_R"] = corners_sitting

            # cv2.drawContours(mask,[obj],0,(200,200,0),2)
    # img = ~mask
    # colored_house_resized = cv2.resize(img, (1500, 1000))
    # cv2.imshow('Objects', colored_house_resized)
    # cv2.waitKey()

    return assigned_objects, sample_zones, doors, lights, img

def find_QR_center(obj):
	# print("here")
	# print(obj)
	center_x = float(sum([obj.polygon[i].x for i in range(4)]))/4
	center_y = float(sum([obj.polygon[i].y for i in range(4)]))/4
	return (center_x,center_y)

def detect_walls(img):
	kernel = np.ones((30,30), np.uint8)
	img_dilation = cv2.dilate(img, kernel, iterations=1)
	kernel = np.ones((200,200), np.uint8)
	img_dilation2 = cv2.dilate(img, kernel, iterations=1)
	edges = cv2.Canny(img_dilation,0,1000,apertureSize = 3)
	minLineLength=10

	lines_list = []
	lines = cv2.HoughLinesP(image=edges,rho=0.8,theta=0.0005, threshold=150,lines=np.array([]), minLineLength=minLineLength,maxLineGap=200)

	duplicates = []
	for i in range(len(lines)):
	       for j in range(i+1,len(lines)):
	              if (lines[i][0][0]-lines[j][0][0])**2 < 2500 and (lines[i][0][1]-lines[j][0][1])**2 < 2500 and  (lines[i][0][2]-lines[j][0][2])**2 < 2500 and  (lines[i][0][3]-lines[j][0][3])**2 < 2500:
	                     if i not in duplicates:
	                            duplicates.append(i)
	lines = np.delete(lines, duplicates, 0)
	for line in lines:
            lines_list.append(find_meter_from_pixel(line))
	# print("len(lines_list):")
	# print(len(lines_list))
	a,b,c = lines.shape
	for i in range(a):
	       cv2.line(img_dilation2, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (100, 200, 20), 3, cv2.LINE_AA)
	img_dilation2 = cv2.resize(img_dilation2, (1500, 1000))
	# cv2.imshow('img_dilation2', img_dilation2)
	# cv2.waitKey()
	# print(lines)

	#still need to merge lines?
	return lines_list


def read_blueprint(image_file_name, library_file_name):

	img = cv2.imread(image_file_name, 0)
	object_library = pd.read_csv(library_file_name)
	rooms, width_t, length_t, colored_house = find_rooms(img.copy())
	colored_house_resized = cv2.resize(colored_house, (1500, 1000))
	#cv2.imshow('Room Sections', colored_house_resized)
	#cv2.waitKey()

	decodedObjects = decode(img)

	objects, sample_zones, doors, lights, detected_objects = find_objects(img.copy(), decodedObjects, object_library)
	detected_objects_resized = cv2.resize(detected_objects, (1500, 1000))

	walls = detect_walls(img)
	return rooms, objects, sample_zones, walls, doors, lights
