# LucasKanadeTracker
Lucas Kanade Tracker Implementation using OpenCV and Python3. 
The code has been evaluated on three video sequences from the [Visual Tracker benchmark database](http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html): featuring a car on the road,
a baby fighting a dragon, and Usain bolt’s race.

![Tracking a car in the video sequence](output/car_tracking.gif)
## Dependencies
Please install the following dependencies before running the code.
1. Python 3
2. OpenCV 3.4 or higher
3. Scipy
4. Numpy

## Build Instructions
To run the code, First clone the repository
```
git clone https://github.com/AmanVirmani/LucasKanadeTracker
```

To run the code on each of the test video sequence,
```
python3 lktracker_baby.py
python3 lktracker_bolt.py
python3 lktracker_car.py
```
