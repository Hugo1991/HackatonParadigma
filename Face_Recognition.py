import cv2
import numpy as np
import colorsys
lower_blue = np.array([255,255, 255], dtype=np.uint8)
upper_blue = np.array([255,255,255], dtype=np.uint8)



def convert_to_black_white(image, min, max):
    return cv2.threshold(image, min, max, cv2.THRESH_BINARY)[1]

def invert_colors(image, min, max):
    return cv2.threshold(image, min, max, cv2.THRESH_BINARY_INV)[1]


def convert_to_gaussian(image):
    return cv2.adaptiveThreshold(image, 20, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_contrast(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def first_filter(fgbg, frame, kernelCl, kernelOp):
    '''

    :param fgbg:
    :param frame:
    :param kernelCl:
    :param kernelOp:
    :return:
    '''
    frame_transform = fgbg.apply((frame))
    frame_transform = convert_to_black_white(frame_transform, 127, 255)
    frame_transform = cv2.morphologyEx(frame_transform, cv2.MORPH_OPEN, kernelOp)
    frame_transform = cv2.morphologyEx(frame_transform, cv2.MORPH_CLOSE, kernelCl)
    contours = cv2.findContours(frame_transform, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return frame_transform, contours[1]


def diff(img, img1):  # returns just the difference of the two images
    return cv2.absdiff(img, img1)


def diff_remove_bg(img0, img, img1):  # removes the background but requires three images
    d1 = diff(img0, img)
    d2 = diff(img, img1)
    return cv2.bitwise_and(d1, d2)
upper_body = cv2.CascadeClassifier('/home/hfernandez/PycharmProjects/untitled/opencv-3.4.0/data/haarcascades/haarcascade_upperbody.xml')
face_cascade = cv2.CascadeClassifier('./opencv-3.4.0/data/haarcascades/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('./opencv-3.4.0/data/haarcascades/haarcascade_eye.xml')
#cap = cv2.VideoCapture("/home/hfernandez/PycharmProjects/untitled/videos/test1.mp4")
#cap = cv2.VideoCapture("/home/hfernandez/PycharmProjects/untitled/videos/test3_2_up_down.mp4")
cap = cv2.VideoCapture("/home/hfernandez/PycharmProjects/untitled/videos/test6_doble_abrazo.mp4")
#cap = cv2.VideoCapture("/home/hfernandez/PycharmProjects/untitled/videos/test10_bola.mp4")
w = cap.get(3)
h = cap.get(4)
frameArea = h*w
areaTH = frameArea/250

line_up = int(2*(h/5))
line_down = int(3*(h/5))

up_limit = int(1*(h/5))
down_limit = int(4*(h/5))

line_down_color = (255,0,0)
line_up_color = (0,0,255)

#img = cv2.imread('/home/hfernandez/Descargas/prueba1.jpeg')
kernelCl = np.ones((100, 1), np.uint8)
kernelOp = np.ones((3, 3), np.uint8)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)  # Create the background substractor
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if i==10:
        firs_frame = convert_to_gray(frame)
    i+=1
    try:

        frame_transform = convert_to_gray(frame)
        frame_transform = diff(firs_frame, frame_transform)
        #frame_transform = convert_to_black_white(frame_transform, 12, 60)

        frame_transform = convert_to_black_white(frame_transform, 60, 220)

        #
        #
        #frame_transform, contours = first_filter(fgbg, frame, kernelCl, kernelOp)
        frame_transform = cv2.medianBlur(frame_transform,5)
        #frame_transform=cv2.bilateralFilter(frame_transform, 9, 75, 75)
        #frame_transform = cv2.morphologyEx(frame_transform, cv2.MORPH_OPEN, kernelOp)
        frame_transform = cv2.morphologyEx(frame_transform, cv2.MORPH_CLOSE, kernelCl)
        contours = cv2.findContours(frame_transform, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #frame_transform = cv2.Canny(frame_transform, 180, 260)
        #frame_transform, contours = first_filter(fgbg, frame_transform, kernelCl, kernelOp)

        for countour in contours[1]:
            area = cv2.contourArea(countour)
            if area > areaTH:
                M = cv2.moments(countour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                x, y, w, h = cv2.boundingRect(countour)
                if area > 20000 and area <50000:
                    cv2.rectangle(frame_transform, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    pass
                elif area > 50000 and area<80000:
                    cv2.rectangle(frame_transform, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    pass
                elif area>650000:
                    cv2.rectangle(frame_transform, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    pass


        cv2.imshow('frame', frame_transform)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()




def detect_faces(gray):
    faces = face_cascade.detectMultiScale(gray, 1.01, 1)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
