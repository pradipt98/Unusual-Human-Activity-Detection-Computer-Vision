import cv2
import numpy as np

cap = cv2.VideoCapture('people.avi')  #this method for FC if 0 or --

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read()  # Capture frame-by-frame
ret, frame2 = cap.read()  # Capture frame-by-frame
print(frame1.shape)

while cap.isOpened():
    imagen = cv2.absdiff(frame1, frame2) #Absolute difference method
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)   #That Difference convert into -> grayscale image
    blur = cv2.GaussianBlur(gray, (5,5), 0)        # Gaussian Blur for smooth and blurred image , also pass ksize(ht,width) and sigmaX (Averaging, Gaussian, Median, Bilateral filter)
    ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)  # (Simple)Thresholding i.e Converting image into binary form and separating regions which are higher than the set threshold.
    dilated = cv2.dilate(thresh, None, iterations=3) # Image filtering using dilate method to fill the holes -> Better Contours
    contours,ret = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #contours<- py list of contours in the image.
                                        #contour retrieval mode # contour approximation method
    for contour in contours:          # Apply the foll contour on original contours
        (x, y, w, h) = cv2.boundingRect(contour) # This method fetches X-Y coordinates and Height and Width

        if cv2.contourArea(contour) < 900: # defining area of contour
            continue          # pt1     #pt2       #color   #size
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)  # This is for the green rectangle
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)   # This is the status for top left corner
                                                            #locn                             #fontscale
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280,720))
    out.write(image)
    cv2.imshow("Este tu resultado!", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27: # keyboard binding function
        break

cv2.destroyAllWindows()
cap.release()
out.release()
