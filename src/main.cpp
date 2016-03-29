#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#define CONFIG_NUM_WORKERS 2

cv::UMat gray, prevgray;

void camera_process(boost::shared_ptr<cv::Mat> frame) {
	cv::Mat flow;
	cv::UMat uflow;
	cvtColor(*frame, gray, cv::COLOR_BGR2GRAY);
	if(!prevgray.empty()) {
		calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
		uflow.copyTo(flow);
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
	std::swap(prevgray, gray);
}

void camera_loop(boost::shared_ptr<boost::asio::io_service> io_service, cv::VideoCapture vid) {
	auto frame = boost::make_shared<cv::Mat>();
	vid >> *frame;
	io_service->post(std::bind(camera_process, frame));
	cv::imshow("optflow", *frame);
	if(cv::waitKey(30) < 0) {
		io_service->post(std::bind(camera_loop, io_service, vid));
	}
}

void camera_main(boost::shared_ptr<boost::asio::io_service> io_service) {
	cv::VideoCapture vid(0);
	if(!vid.isOpened()) { throw std::runtime_error("failed to open video capture device"); }
	cv::namedWindow("optflow");
	io_service->post(std::bind(camera_loop, io_service, vid));
}

int slow_main(int argc, char *argv[]) {
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

void worker_main(boost::shared_ptr<boost::asio::io_service> io_service) {
	io_service->run();
}

int main(int argc, char *argv[]) {
	//slow_main(argc, argv);
	auto io_service = boost::make_shared<boost::asio::io_service>();
	auto work       = boost::make_shared<boost::asio::io_service::work>(*io_service);
	auto strand     = boost::make_shared<boost::asio::io_service::strand>(*io_service);
	boost::thread_group workers;
	for(unsigned int w = 0; w < CONFIG_NUM_WORKERS; ++w) {
		workers.create_thread(boost::bind(worker_main, io_service));
	}
	io_service->post(boost::bind(camera_main, io_service));
	work.reset();
	workers.join_all();
	return 0;
}
