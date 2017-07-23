
#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/core/cuda.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<stdio.h>
#include<time.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;

String face_cascade_name = "C:\\haarcascades\\haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "C:\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";
int timePrev = 0;
int frameCount = 0;

void detectAndDisplay(Mat frame);
///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
	printf("cuda : %d\n", getCudaEnabledDeviceCount());
	cv::VideoCapture capWebcam(0);		// declare a VideoCapture object and associate to webcam, 0 => use 1st webcam
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); while (1); return 0; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return 0; };
	if (capWebcam.isOpened() == false) {				// check if VideoCapture object was associated to webcam successfully
		std::cout << "error: capWebcam not accessed successfully\n\n";	// if not, print error message to std out
		return(0);														// and exit program
	}

	Mat imgOriginal;		// input image

	char charCheckForEscKey = 0;

	while (charCheckForEscKey != 27 && capWebcam.isOpened()) {		// until the Esc key is pressed or webcam connection is lost
		frameCount++;
		if (time(NULL) - timePrev >= 1) {
			printf("FPS : %d\n",frameCount);
			frameCount = 0;
			timePrev = time(NULL);
		}
		bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);		// get next frame

		if (!blnFrameReadSuccessfully || imgOriginal.empty()) {		// if frame not read successfully
			std::cout << "error: frame not read from webcam\n";		// print error message to std out
			break;													// and jump out of while loop
		}		

			// declare windows
		//cv::namedWindow("imgOriginal", CV_WINDOW_AUTOSIZE);	// note: you can use CV_WINDOW_NORMAL which allows resizing the window
		//cv::imshow("imgOriginal", imgOriginal);			// show windows
		detectAndDisplay(imgOriginal);
		charCheckForEscKey = cv::waitKey(1);			// delay (in ms) and get key press, if any
	}	// end while

	return(0);
}

void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	cv::Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point face_center(faces[i].x, faces[i].y);
		Point face_size(faces[i].x+faces[i].width, faces[i].y+faces[i].height);
		rectangle(frame, face_center, face_size, Scalar(255, 0, 0), 2, 8);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y);
			Point eye_size(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y + eyes[j].height);
			rectangle(frame, eye_center, eye_size, Scalar(255, 0, 0), 2, 8);
		}
	}
	//-- Show what you got
	imshow(window_name, frame);
}