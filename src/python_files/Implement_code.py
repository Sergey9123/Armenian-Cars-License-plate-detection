import cv2 as cv
import os
import numpy as np

def img_resize(im):
    desired_size = 800
    old_size = im.shape[:2]

    ratio = desired_size/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,value=color)
    return new_im


def clearing_img(img,from_dir):
    image = cv.imread(f'../../data/{from_dir}/'+img)
    image = img_resize(image)
    return image

def save_data(from_dir,to_dir,count):
    images = os.listdir(f'../../data/{from_dir}/')
    count = count
    for image in images:
        if (image[-4:] == ".jpg" or image[-4:] ==  ".png" or image[-4:] == '.JPG' or image[-4:] == '.PNG' or image[-5:] == '.jpeg'):
            cv.imwrite(f"../../data/{to_dir}/car_"+(5-len(str(count)))*"0"+str(count)+".jpg",clearing_img(image,from_dir))
        count += 1



import cv2
cap = cv2.VideoCapture("data_video.mp4")
i = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('car_'+str(i)+'.jpg', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()



















































