import cv2
import numpy as np
import math


_TITLE = 'Monitor Gestures'
_TITLE_MASK = 'Monitor Gestures - Mask'
_AREA_COORDINATES = {
    'TOP_RIGHT': 600,
    'TOP_LEFT': 400,
    'BOTTOM_RIGHT': 300,
    'BOTTOM_LEFT': 100
}
_HSV_LOWER = np.array([0,20,70], dtype=np.uint8)
_HSV_UPPER = np.array([20,255,255], dtype=np.uint8)
_KARNEL = np.ones((3,3),np.uint8)
_CONTOUR_RATE = 0.0005
_FONT = cv2.FONT_HERSHEY_SIMPLEX

def _defineGestures(frame, mask, area):
    _text = 'Show hand in marked area!'
    _contour = _search_max_contour(mask)

    if _contour is not None:
        _approx, _hull, _defects, _area_not_covered, _contour_area = _convexHull(_contour)
        _number_finger = 0

        #code for finding number of defects due to fingers
        if _defects is not None and _defects.shape[0] is not None:
            for index in range(_defects.shape[0]):
                _defect_start, _defect_end, _defect_far, _defect_distance = _defects[index, 0]
                _start = tuple(_approx[_defect_start][0])
                _end = tuple(_approx[_defect_end][0])
                _far = tuple(_approx[_defect_far][0])
                
                _finger_area, _finger_distance, _finger_angle = _getFingerData(_start, _end, _far)

                if _finger_angle <= 90 and _finger_distance > 30:
                    _number_finger += 1
                    cv2.circle(area, _far, 3, [0, 0, 255], -1)
                
                cv2.line(area, _start, _end, [0, 255, 0], 2) # draw line around hand
        
            _number_finger += 1
            _text = _get_text(_number_finger, _area_not_covered, _contour_area)
    
    _write_text(frame, _text)
            
def _write_text(frame, text):
    cv2.putText(frame, text, (0,50), _FONT, 1, (0,0,255), 3, cv2.LINE_AA)

def _get_text(number_finger, area_not_covered, contour_area):
    _text = ''

    if number_finger == 1:
        if contour_area < 2000:
            _text = 'Show hand in marked area!'
        elif area_not_covered < 12:
            _text = '0'
        else:
            _text = '1'
    elif number_finger == 2:
        _text = '2'
    elif number_finger == 3:
        _text = '3'
    elif number_finger == 4:
        _text = '4'
    elif number_finger == 5:
        _text = '5'
    else:
        _text = 'Please, adjust your hand!'
    return _text

def _getSqrt(X, Y):
    return math.sqrt((X[0] - Y[0]) ** 2 + (X[1] -Y[1]) ** 2)

def _getSqrtArea(X, Y, Z, S):
    return math.sqrt(S * (S - X) * (S - Y) * (S - Z))

def _getFingerDistance(X, S):
    return (2 * S) / X

def _getFingerAngle(X, Y, Z):
    return math.cos((Y ** 2 + Z ** 2 - X ** 2) / (2 * Y * Z)) * 57

def _getFingerData(start, end, far):
    _triangle_point_x = _getSqrt(end, start)
    _triangle_point_y = _getSqrt(far, start)
    _triangle_point_z = _getSqrt(end, far)
    _trinagle_area = (_triangle_point_x + _triangle_point_y + _triangle_point_z) / 2
    
    _area = _getSqrtArea(_triangle_point_x, _triangle_point_y, _triangle_point_z, _trinagle_area)
    _distance = _getFingerDistance(_triangle_point_x, _area)
    _angle = _getFingerAngle(_triangle_point_x, _triangle_point_y, _triangle_point_z)

    return _area, _distance, _angle

def _convexHull(contour):
     # approx the contour a little
    _epsilon = _CONTOUR_RATE * cv2.arcLength(contour, True)

    _approx= cv2.approxPolyDP(contour, _epsilon, True)
    _hull = cv2.convexHull(contour) # make convex hull around hand
    
    # define area of hull and area of hand
    _hull_area = cv2.contourArea(_hull)
    _contour_area = cv2.contourArea(contour)

    # area not covered by hand
    _area_not_covered = ((_hull_area - _contour_area) / _contour_area) * 100

    # find the defects in convex hull with respect to hand
    _hull = cv2.convexHull(_approx, returnPoints=False)
    _defects = cv2.convexityDefects(_approx, _hull)

    return _approx, _hull, _defects, _area_not_covered, _contour_area

def _search_max_contour(frame):
    try:
        _contours, _hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return max(_contours, key = lambda x: cv2.contourArea(x)) # find contour of max area(hand)
    except:
        return None

def _apply_mask(hsv_frame):
    _mask = cv2.inRange(hsv_frame, _HSV_LOWER, _HSV_UPPER) # extract skin colur image
    _mask = cv2.dilate(_mask, _KARNEL, iterations=4) # fill the spots inside the hand
    _mask = cv2.GaussianBlur(_mask, (5,5), 100) 
    return _mask

def _isFinished(key):
    return key == 27 or cv2.getWindowProperty(_TITLE, 0) < 0 or cv2.getWindowProperty(_TITLE_MASK, 0) < 0

def _generateMonitoringArea(frame):
    _area = frame[_AREA_COORDINATES['BOTTOM_LEFT']:_AREA_COORDINATES['BOTTOM_RIGHT'], _AREA_COORDINATES['TOP_LEFT']:_AREA_COORDINATES['TOP_RIGHT']]
    cv2.rectangle(frame, (_AREA_COORDINATES['TOP_LEFT'], _AREA_COORDINATES['BOTTOM_LEFT']), (_AREA_COORDINATES['TOP_RIGHT'],_AREA_COORDINATES['BOTTOM_RIGHT']), (0, 255, 0), 0)
    return _area

def monitor(capture):
    try:
        while True:
            _, _frame = capture.read()
            _frame = cv2.flip(_frame, 1) # inverts the image
            _area = _generateMonitoringArea(_frame)
            
            _hsv = cv2.cvtColor(_area, cv2.COLOR_BGR2HSV) # convert RGB/BGR to HSV -> increases the saturation of the image to view the edges
            
            _mask = _apply_mask(_hsv)
            _defineGestures(_frame, _mask, _area)

            cv2.imshow(_TITLE, _frame)
            cv2.imshow(_TITLE_MASK, _mask)

            if _isFinished(cv2.waitKey(1)):
                capture.release()
                cv2.destroyAllWindow()
                break

    except Exception as ex:
        print(ex)
        print('The execution of the gesture monitor is finished!')

if __name__ == "__main__":
    monitor(cv2.VideoCapture(0))