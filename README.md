# Car Speed Estimation with Deep Learning

## Problem Statement : 
Estimate vehicle speed using Traffic cameras without calibrating pixel scale. 

## Libraries Used : 
- tensorflow
- tensornets
- opencv
- numpy 
- matplotlib

## Solution : 

### Assumptions :
- The first assumption is that the white markings separating various lanes is of a fixed length of 1.5 meters.__[source](http://www.ctp.gov.in/RoadMarkings.htm)__
- In addition to this the distance between 2 white markings is a fixed length of 3 meters.
- This video generates 3336 frames over 2 minutes and 13 seconds, that is __25fps__.
- This solution is only valid for cars driving one way, that is away from the camera. 


### Estimating Time :
1. __gen_frames.py__
- Although this script is not explicitly required, it was used to calculate the frame rate used(Total time/total frames)
- Additionally, any of the frames generated can be used with detect_road.py(but prefer a frame with empty road)

### Estimating Distance :
1. __detect_road.py__
- A single frame is passed to the script.
- Triangles are cut through this frame extracting only the road and its white markings. 
- The resultant image is converted to grayscale, then the noise is reduced by using gaussian blur, and the white markings are generated using Canny edge detection, houghlines transform . __(opencv)__
- Finally these lines are passed to main script.
- Please note that the lines selected may warry for different frames run, so try to use the same frame.

2. __main.py__
- This script relies on pretrained YoloV3 for vehicle detection.
- This yolo is run on a smaller resolution (416, 416) and detects 5 classes namely - __'bicycle', 'car', 'bike', 'bus' and 'truck'__.
- As the frames are captured, the image is resized, preprocessed and Yolo predicts the bounding boxes. 
- A vehicle class is maintained to keep track of vehicle as the frames progress. 
- A list of vehicle objects are maintained (and kept in memory for upto 10 frames ). 
- A vehicle present in the previous frame is identified by tracking the vehcicle type, and the (x,y) position of a single vertex of the bounding box which can only increase along the X Axis and decrease along Y Axis(origin at top left corner)__*This condition is validated only on the basis of the fourth assumption*__ 
- The vehicle is tracked as it moves along frames and identified if it passed the 4 fixed lines parellel to the X-axis.
- Since we know the distance between these 4 lines, we track the number of frames passed between the starting line and the ending line and hence calculate the velocity. 

## Resources:
- The YoloV3 (pretrained) used in main.py is inspired from __[this article from towardsdatascience](https://towardsdatascience.com/track-vehicles-and-people-using-yolov3-and-tensorflow-4f3d0e5b1b5f?gi=2094c76c1536)__
- The white markings detected using canny edge detection, houghlines transform is inspired from __[this medium article](https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0)__.