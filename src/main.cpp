#include <iostream>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[]) {
	cv::VideoCapture vid(0);
	if(!vid.isOpened()) {
		return -1;
	}
	cv::Mat flow, frame;
	cv::UMat gray, prevgray, uflow;
	cv::namedWindow("optflow");
	for(;;) {
		vid >> frame;
		cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		if(!prevgray.empty()) {
			calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
			uflow.copyTo(flow);
			cv::imshow("optflow", frame);
		}
		cv::Point2f total;
		const int step = 16;
		for(unsigned int y = 0; y < flow.rows; y += step) {
			for(unsigned int x = 0; x < flow.cols; x += step) {
				total += flow.at<cv::Point2f>(y, x);
			}
		}
		if(sqrt(total.x * total.x + total.y * total.y) > 500) {
			std::cout << total << std::endl;
		}
		if(cv::waitKey(30) >= 0) {
			break;
		} else {
			std::swap(prevgray, gray);
		}
	}
	return 0;
}
