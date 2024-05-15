import cv2 
import matplotlib

matplotlib.use('TkAgg')


def draw_circle(image, object, img_ratio, color=0):
    center_coordinates = (int((float(object["x1"]) + (float(object["x2"]) - float(object["x1"])) / 2)*img_ratio), int((float(object["y1"]) + (float(object["y2"]) - float(object["y1"])) / 2)*img_ratio))
    radius = 0
    if color == 0:
        color_circle = (0, 0, 255)
    else:
        color_circle = color
    thickness_circle = 10

    cv2.circle(image, center_coordinates, radius, color_circle, thickness_circle)


def draw_line(image, start_point, end_point):
    color, thickness = (255,255,255), 2
    cv2.arrowedLine(image, start_point, end_point, color, thickness)


def draw_text(image, track, text):
    object = track.to_tlbr()

    coordinates = (int(object[0]), int(object[1]-10))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color_text = (255, 255, 255)
    thickness = 2

    cv2.putText(image, text, coordinates, font, fontScale, color_text, thickness)
  

def draw_rectangle(image, track, color):
    object = track.to_tlbr()
    0
    start_point, end_point = (int(object[0]), int(object[1])), (int(object[2]), int(object[3]))
    color_rectangle = color
    thickness_rectangle = 2
    
    cv2.rectangle(image, start_point, end_point, color_rectangle, thickness_rectangle)

def draw_parallelogram(image, top_left, top_right, bottom_left, bottom_right):
    thickness = 2
    color = (57, 255, 20)
    # print(top_left)
    top_left = (int(top_left[0]), int(top_left[1]))
    top_right = (int(top_right[0]), int(top_right[1]))
    bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
    cv2.line(image, top_left, top_right, color, thickness)
    cv2.line(image, top_right, bottom_right, color, thickness)
    cv2.line(image, bottom_right, bottom_left, color, thickness)
    cv2.line(image, bottom_left, top_left, color, thickness)
    