import cv2

path = 'video.mp4'
output = './frames/'
vid_obj = cv2.VideoCapture(path)

frame = 0
check = 1

while check:
    check, image = vid_obj.read()
    cv2.imwrite(output + "frame%d.jpg" %frame, image)

    frame+=1