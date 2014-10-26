#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <MMsystem.h>
#include <sstream>
#include <string>
#include <math.h>
#include <fstream>
#include <vector>
#include <iostream>
#include "myImage.hpp"
#include "roi.hpp"
#include "handGesture.hpp"
#include "main.hpp"

using namespace cv;
using namespace std;

/* Global Variables  */
int fontFace = FONT_HERSHEY_PLAIN;
int square_len;
int avgColor[NSAMPLES][3];
int c_lower[NSAMPLES][3];
int c_upper[NSAMPLES][3];
int avgBGR[3];
int nrOfDefects;
int iSinceKFInit;
struct dim{ int w; int h; }boundingDim;
VideoWriter out;
Mat edges;
My_ROI roi1, roi2, roi3, roi4, roi5, roi6;
vector <My_ROI> roi;
vector <KalmanFilter> kf;
vector <Mat_<float> > measurement;

//int gameFlag;

/* end global variables */

void init(MyImage *m){
	square_len = 20;
	iSinceKFInit = 0;
}

// change a color from one space to another
void col2origCol(int hsv[3], int bgr[3], Mat src){
	Mat avgBGRMat = src.clone();
	for (int i = 0; i<3; i++){
		avgBGRMat.data[i] = hsv[i];
	}
	cvtColor(avgBGRMat, avgBGRMat, COL2ORIGCOL);
	for (int i = 0; i<3; i++){
		bgr[i] = avgBGRMat.data[i];
	}
}

void printText(Mat src, string text){
	int fontFace = FONT_HERSHEY_PLAIN;
	putText(src, text, Point(src.cols / 2, src.rows / 10), fontFace, 1.2f, Scalar(200, 0, 0), 2);
}

void waitForPalmCover(MyImage* m){
	m->cap >> m->src;
	flip(m->src, m->src, 1);
	roi.push_back(My_ROI(Point(m->src.cols / 3, m->src.rows / 6), Point(m->src.cols / 3 + square_len, m->src.rows / 6 + square_len), m->src));
	roi.push_back(My_ROI(Point(m->src.cols / 4, m->src.rows / 2), Point(m->src.cols / 4 + square_len, m->src.rows / 2 + square_len), m->src));
	roi.push_back(My_ROI(Point(m->src.cols / 3, m->src.rows / 1.5), Point(m->src.cols / 3 + square_len, m->src.rows / 1.5 + square_len), m->src));
	roi.push_back(My_ROI(Point(m->src.cols / 2, m->src.rows / 2), Point(m->src.cols / 2 + square_len, m->src.rows / 2 + square_len), m->src));
	roi.push_back(My_ROI(Point(m->src.cols / 2.5, m->src.rows / 2.5), Point(m->src.cols / 2.5 + square_len, m->src.rows / 2.5 + square_len), m->src));
	roi.push_back(My_ROI(Point(m->src.cols / 2, m->src.rows / 1.5), Point(m->src.cols / 2 + square_len, m->src.rows / 1.5 + square_len), m->src));
	roi.push_back(My_ROI(Point(m->src.cols / 2.5, m->src.rows / 1.8), Point(m->src.cols / 2.5 + square_len, m->src.rows / 1.8 + square_len), m->src));


	for (int i = 0; i<50; i++){
		m->cap >> m->src;
		flip(m->src, m->src, 1);
		for (int j = 0; j<NSAMPLES; j++){
			roi[j].draw_rectangle(m->src);
		}
		string imgText = string("Cover rectangles with palm");
		printText(m->src, imgText);

		if (i == 30){
			//	imwrite("./images/waitforpalm1.jpg",m->src);
		}

		imshow("img1", m->src);
		out << m->src;
		if (cv::waitKey(30) >= 0) break;
	}
}

int getMedian(vector<int> val){
	int median;
	size_t size = val.size();
	sort(val.begin(), val.end());
	if (size % 2 == 0)  {
		median = val[size / 2 - 1];
	}
	else{
		median = val[size / 2];
	}
	return median;
}


void getAvgColor(MyImage *m, My_ROI roi, int avg[3]){
	Mat r;
	roi.roi_ptr.copyTo(r);
	vector<int>hm;
	vector<int>sm;
	vector<int>lm;
	// generate vectors
	for (int i = 2; i<r.rows - 2; i++){
		for (int j = 2; j<r.cols - 2; j++){
			hm.push_back(r.data[r.channels()*(r.cols*i + j) + 0]);
			sm.push_back(r.data[r.channels()*(r.cols*i + j) + 1]);
			lm.push_back(r.data[r.channels()*(r.cols*i + j) + 2]);
		}
	}
	avg[0] = getMedian(hm);
	avg[1] = getMedian(sm);
	avg[2] = getMedian(lm);
}

void average(MyImage *m){
	m->cap >> m->src;
	flip(m->src, m->src, 1);
	for (int i = 0; i<30; i++){
		m->cap >> m->src;
		flip(m->src, m->src, 1);
		cvtColor(m->src, m->src, ORIGCOL2COL);
		for (int j = 0; j<NSAMPLES; j++){
			getAvgColor(m, roi[j], avgColor[j]);
			roi[j].draw_rectangle(m->src);
		}
		cvtColor(m->src, m->src, COL2ORIGCOL);
		string imgText = string("Finding average color of hand");
		printText(m->src, imgText);
		imshow("img1", m->src);
		if (cv::waitKey(30) >= 0) break;
	}
}

void initTrackbars(){

	for (int i = 0; i<NSAMPLES; i++){
		c_lower[i][0] = 12;
		c_upper[i][0] = 7;
		c_lower[i][1] = 30;
		c_upper[i][1] = 40;
		c_lower[i][2] = 80;
		c_upper[i][2] = 80;
	}
	createTrackbar("lower1", "trackbars", &c_lower[0][0], 255);
	createTrackbar("lower2", "trackbars", &c_lower[0][1], 255);
	createTrackbar("lower3", "trackbars", &c_lower[0][2], 255);
	createTrackbar("upper1", "trackbars", &c_upper[0][0], 255);
	createTrackbar("upper2", "trackbars", &c_upper[0][1], 255);
	createTrackbar("upper3", "trackbars", &c_upper[0][2], 255);

}


void normalizeColors(MyImage * myImage){
	// copy all boundries read from trackbar
	// to all of the different boundries
	for (int i = 1; i<NSAMPLES; i++){
		for (int j = 0; j<3; j++){
			c_lower[i][j] = c_lower[0][j];
			c_upper[i][j] = c_upper[0][j];
		}
	}
	// normalize all boundries so that 
	// threshold is whithin 0-255
	for (int i = 0; i<NSAMPLES; i++){
		if ((avgColor[i][0] - c_lower[i][0]) <0){
			c_lower[i][0] = avgColor[i][0];
		}if ((avgColor[i][1] - c_lower[i][1]) <0){
			c_lower[i][1] = avgColor[i][1];
		}if ((avgColor[i][2] - c_lower[i][2]) <0){
			c_lower[i][2] = avgColor[i][2];
		}if ((avgColor[i][0] + c_upper[i][0]) >255){
			c_upper[i][0] = 255 - avgColor[i][0];
		}if ((avgColor[i][1] + c_upper[i][1]) >255){
			c_upper[i][1] = 255 - avgColor[i][1];
		}if ((avgColor[i][2] + c_upper[i][2]) >255){
			c_upper[i][2] = 255 - avgColor[i][2];
		}
	}
}

void produceBinaries(MyImage *m){
	Scalar lowerBound;
	Scalar upperBound;
	Mat foo;
	for (int i = 0; i<NSAMPLES; i++){
		normalizeColors(m);
		lowerBound = Scalar(avgColor[i][0] - c_lower[i][0], avgColor[i][1] - c_lower[i][1], avgColor[i][2] - c_lower[i][2]);
		upperBound = Scalar(avgColor[i][0] + c_upper[i][0], avgColor[i][1] + c_upper[i][1], avgColor[i][2] + c_upper[i][2]);
		m->bwList.push_back(Mat(m->srcLR.rows, m->srcLR.cols, CV_8U));
		inRange(m->srcLR, lowerBound, upperBound, m->bwList[i]);
	}
	m->bwList[0].copyTo(m->bw);
	for (int i = 1; i<NSAMPLES; i++){
		m->bw += m->bwList[i];
	}
	medianBlur(m->bw, m->bw, 7);
}

void initWindows(MyImage m){
	//namedWindow("trackbars", CV_WINDOW_KEEPRATIO);
	namedWindow("img1", CV_WINDOW_KEEPRATIO);
}

void showWindows(MyImage m){
	pyrDown(m.bw, m.bw);
	pyrDown(m.bw, m.bw);
	Rect roi(Point(3 * m.src.cols / 4, 0), m.bw.size());
	vector<Mat> channels;
	Mat result;
	for (int i = 0; i<3; i++)
		channels.push_back(m.bw);
	merge(channels, result);
	result.copyTo(m.src(roi));
	imshow("img1", m.src);

	// ready to play
	//gameFlag = 0;
}

int findBiggestContour(vector<vector<Point> > contours){
	int indexOfBiggestContour = -1;
	int sizeOfBiggestContour = 0;
	for (int i = 0; i < contours.size(); i++){
		if (contours[i].size() > sizeOfBiggestContour){
			sizeOfBiggestContour = contours[i].size();
			indexOfBiggestContour = i;
		}
	}
	return indexOfBiggestContour;
}

void myDrawContours(MyImage *m, HandGesture *hg){
	drawContours(m->src, hg->hullP, hg->cIdx, cv::Scalar(200, 0, 0), 2, 8, vector<Vec4i>(), 0, Point());




	rectangle(m->src, hg->bRect.tl(), hg->bRect.br(), Scalar(0, 0, 200));
	vector<Vec4i>::iterator d = hg->defects[hg->cIdx].begin();
	int fontFace = FONT_HERSHEY_PLAIN;


	vector<Mat> channels;
	Mat result;
	for (int i = 0; i<3; i++)
		channels.push_back(m->bw);
	merge(channels, result);
	//	drawContours(result,hg->contours,hg->cIdx,cv::Scalar(0,200,0),6, 8, vector<Vec4i>(), 0, Point());
	drawContours(result, hg->hullP, hg->cIdx, cv::Scalar(0, 0, 250), 10, 8, vector<Vec4i>(), 0, Point());


	while (d != hg->defects[hg->cIdx].end()) {
		Vec4i& v = (*d);
		int startidx = v[0]; Point ptStart(hg->contours[hg->cIdx][startidx]);
		int endidx = v[1]; Point ptEnd(hg->contours[hg->cIdx][endidx]);
		int faridx = v[2]; Point ptFar(hg->contours[hg->cIdx][faridx]);
		float depth = v[3] / 256;
		/*
		line( m->src, ptStart, ptFar, Scalar(0,255,0), 1 );
		line( m->src, ptEnd, ptFar, Scalar(0,255,0), 1 );
		circle( m->src, ptFar,   4, Scalar(0,255,0), 2 );
		circle( m->src, ptEnd,   4, Scalar(0,0,255), 2 );
		circle( m->src, ptStart,   4, Scalar(255,0,0), 2 );
		*/
		circle(result, ptFar, 9, Scalar(0, 205, 0), 5);


		d++;

	}
	//	imwrite("./images/contour_defects_before_eliminate.jpg",result);

}

void makeContours(MyImage *m, HandGesture* hg){
	Mat aBw;
	pyrUp(m->bw, m->bw);
	m->bw.copyTo(aBw);
	findContours(aBw, hg->contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	hg->initVectors();
	hg->cIdx = findBiggestContour(hg->contours);
	if (hg->cIdx != -1){
		//		approxPolyDP( Mat(hg->contours[hg->cIdx]), hg->contours[hg->cIdx], 11, true );
		hg->bRect = boundingRect(Mat(hg->contours[hg->cIdx]));
		convexHull(Mat(hg->contours[hg->cIdx]), hg->hullP[hg->cIdx], false, true);
		convexHull(Mat(hg->contours[hg->cIdx]), hg->hullI[hg->cIdx], false, false);
		approxPolyDP(Mat(hg->hullP[hg->cIdx]), hg->hullP[hg->cIdx], 18, true);
		if (hg->contours[hg->cIdx].size()>3){
			convexityDefects(hg->contours[hg->cIdx], hg->hullI[hg->cIdx], hg->defects[hg->cIdx]);
			hg->eleminateDefects(m);
		}
		bool isHand = hg->detectIfHand();
		hg->printGestureInfo(m->src);
		if (isHand){
			hg->getFingerTips(m);
			hg->drawFingerTips(m);
			myDrawContours(m, hg);
		}
	}
}

void overlayImage(const cv::Mat &background, const cv::Mat &foreground,
	cv::Mat &output, cv::Point2i location)
{
	background.copyTo(output);


	// start at the row indicated by location, or at row 0 if location.y is negative.
	for (int y = max(location.y, 0); y < background.rows; ++y)
	{
		int fY = y - location.y; // because of the translation

		// we are done of we have processed all rows of the foreground image.
		if (fY >= foreground.rows)
			break;

		// start at the column indicated by location, 

		// or at column 0 if location.x is negative.
		for (int x = max(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x; // because of the translation.

			// we are done with this row if the column is outside of the foreground image.
			if (fX >= foreground.cols)
				break;

			// determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
			double opacity =
				((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

				/ 255.;


			// and now combine the background and foreground pixel, using the opacity, 

			// but only if opacity > 0.
			for (int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx =
					foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx =
					background.data[y * background.step + x * background.channels() + c];
				output.data[y*output.step + output.channels()*x + c] =
					backgroundPx * (1. - opacity) + foregroundPx * opacity;
			}
		}
	}
}

int main(){

	namedWindow("Rock-Paper-Scissor", WINDOW_AUTOSIZE);
	moveWindow("Rock-Paper-Scissor", 500, 80);
	Mat cover = imread("cover3.jpg");
	imshow("Rock-Paper-Scissor", cover);
	waitKey(1);
	PlaySound(TEXT("s_ini.wav"), NULL, SND_SYNC);


	MyImage m(0);
	HandGesture hg;
	init(&m);
	m.cap >> m.src;
	namedWindow("img1", CV_WINDOW_KEEPRATIO);
	moveWindow("img1", 50, 380);
	resizeWindow("img1", 400, 300);
	out.open("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, m.src.size(), true);
	waitForPalmCover(&m);
	average(&m);
	//destroyWindow("img1");
	


	Mat board = imread("board.jpg");
	Mat paper = imread("paper.jpg");
	Mat rock = imread("rock.jpg");
	Mat scissor = imread("scissor.jpg");
	Mat pp = imread("paper_paper.png");
	Mat pr = imread("paper_rock.png");
	Mat ps = imread("paper_scissor.png");
	Mat rp = imread("rock_paper.png");
	Mat rr = imread("rock_rock.png");
	Mat rs = imread("rock_scissor.png");
	Mat sp = imread("scissor_paper.png");
	Mat sr = imread("scissor_rock.png");
	Mat ss = imread("scissor_scissor.png");
	Mat result;

	int result_com = 0;
	int result_user = 0;
	int detect_result = 0;
	int temp_com = 0;
	int temp_user = 0;
	char Key;

	//initWindows(m);
	initTrackbars();
	int count = 0;
	int countCycle = 1;
	for (;;){	// cycle = 20 frame
		hg.frameNumber++;
		// count for number of frame
		count++;

		//cout << count << endl;
		if (count == 1){
			//cout << "start" << endl;
			PlaySound(TEXT("s_RockPaperScissor.wav"), NULL, SND_SYNC);
			//cout << "end" << endl;
		}

		



		m.cap >> m.src;


		flip(m.src, m.src, 1);
		pyrDown(m.src, m.srcLR);
		blur(m.srcLR, m.srcLR, Size(3, 3));
		cvtColor(m.srcLR, m.srcLR, ORIGCOL2COL);
		produceBinaries(&m);
		cvtColor(m.srcLR, m.srcLR, COL2ORIGCOL);
		makeContours(&m, &hg);
		hg.getFingerNumber(&m);
		
		//out << m.src;
		//imwrite("result.jpg", m.src);
		//if (cv::waitKey(30) == char('q')) break;


		if (count == 5){

			showWindows(m);

			// get the result
			result_com = rand() % 3;
			detect_result = hg.fin_num;
			cout << "detection result: " << hg.fin_num << endl;
			if (detect_result < 2)
				result_user = 1;
			else if (detect_result > 2)
				result_user = 0;
			else
				result_user = 2;

			//cout << "imshow" << endl;
			//cout << "computer: " << result_com << endl;
			//cout << "user: " << result_user << endl;
			temp_com = result_com;
			temp_user = result_user;


			if (result_com == 0){ //paper
				if (result_user == 0){ //paper
					//overlayImage(board, paper, result, cv::Point(80, 60));
					//overlayImage(result, paper, result, cv::Point(480, 60));
					overlayImage(board, pp, result, Point(0, 0));
					imshow("Rock-Paper-Scissor", result);
					waitKey(1);
					//PlaySound(TEXT("s_again.wav"), NULL, SND_SYNC);
				}
				else if (result_user == 1){ //rock
					//overlayImage(board, paper, result, cv::Point(80, 60));
					//overlayImage(result, rock, result, cv::Point(480, 50));
					overlayImage(board, pr, result, Point(0, 0));
					imshow("Rock-Paper-Scissor", result);
					waitKey(1);
					//PlaySound(TEXT("s_win.wav"), NULL, SND_SYNC);
				}
				else{ //scissor
					//overlayImage(board, paper, result, cv::Point(80, 60));
					//overlayImage(result, scissor, result, cv::Point(490, 50));
					overlayImage(board, ps, result, Point(0, 0));
					imshow("Rock-Paper-Scissor", result);
					waitKey(1);
					//PlaySound(TEXT("s_sad.wav"), NULL, SND_SYNC);
				}
			}
			else if (result_com == 1){ // rock
				if (result_user == 0){ // paper
					//overlayImage(board, rock, result, cv::Point(80, 50));
					//overlayImage(result, paper, result, cv::Point(480, 60));
					overlayImage(board, rp, result, Point(0, 0));
					imshow("Rock-Paper-Scissor", result);
					waitKey(1);
					//PlaySound(TEXT("s_sad.wav"), NULL, SND_SYNC);
				}
				else if (result_user == 1){ // rock
					//overlayImage(board, rock, result, cv::Point(80, 50));
					//overlayImage(result, rock, result, cv::Point(480, 50));
					overlayImage(board, rr, result, Point(0, 0));
					imshow("Rock-Paper-Scissor", result);
					waitKey(1);
					//PlaySound(TEXT("s_again.wav"), NULL, SND_SYNC);
				}
				else{ // scissor
					//overlayImage(board, rock, result, cv::Point(80, 50));
					//overlayImage(result, scissor, result, cv::Point(490, 50));
					overlayImage(board, rs, result, Point(0, 0));
					imshow("Rock-Paper-Scissor", result);
					waitKey(1);
					//PlaySound(TEXT("s_win.wav"), NULL, SND_SYNC);
				}
			}
			else { // scissor
				if (result_user == 0){ // paper
					//overlayImage(board, scissor, result, cv::Point(90, 50));
					//overlayImage(result, paper, result, cv::Point(480, 60));
					overlayImage(board, sp, result, Point(0, 0));
					imshow("Rock-Paper-Scissor", result);
					waitKey(1);
					//PlaySound(TEXT("s_win.wav"), NULL, SND_SYNC);
				}
				else if (result_user == 1){ // rock
					//overlayImage(board, scissor, result, cv::Point(90, 50));
					//overlayImage(result, rock, result, cv::Point(480, 50));
					overlayImage(board, sr, result, Point(0, 0));
					imshow("Rock-Paper-Scissor", result);
					waitKey(1);
					//PlaySound(TEXT("s_sad.wav"), NULL, SND_SYNC);
				}
				else{ // scissor
					//overlayImage(board, scissor, result, cv::Point(90, 50));
					//overlayImage(result, scissor, result, cv::Point(490, 50));
					overlayImage(board, ss, result, Point(0, 0));
					imshow("Rock-Paper-Scissor", result);
					waitKey(1);
					//PlaySound(TEXT("s_again.wav"), NULL, SND_SYNC);
				}
			}
		}
		// play the feedback
		else if (count == 6){

			//cout << "play" << endl;
			//cout << "computer: " << temp_com << endl;
			//cout << "user: " << temp_user << endl;

			if (temp_com == 0){ //paper
				if (temp_user == 0){ //paper
					PlaySound(TEXT("s_again.wav"), NULL, SND_SYNC);
				}
				else if (temp_user == 1){ //rock
					PlaySound(TEXT("s_win.wav"), NULL, SND_SYNC);
				}
				else{ //scissor
					PlaySound(TEXT("s_sad.wav"), NULL, SND_SYNC);
				}
			}
			else if (temp_com == 1){ // rock
				if (temp_user == 0){ // paper
					PlaySound(TEXT("s_sad.wav"), NULL, SND_SYNC);
				}
				else if (temp_user == 1){ // rock
					PlaySound(TEXT("s_again.wav"), NULL, SND_SYNC);
				}
				else{ // scissor
					PlaySound(TEXT("s_win.wav"), NULL, SND_SYNC);
				}
			}
			else { // scissor
				if (temp_user == 0){ // paper
					PlaySound(TEXT("s_win.wav"), NULL, SND_SYNC);
				}
				else if (temp_user == 1){ // rock
					PlaySound(TEXT("s_sad.wav"), NULL, SND_SYNC);
				}
				else{ // scissor
					PlaySound(TEXT("s_again.wav"), NULL, SND_SYNC);
				}
			}

		}
		else if (count == 8){

			if (countCycle == 5){
				//Key = waitKey();
				//if (Key == '1'){
					PlaySound(TEXT("s_end.wav"), NULL, SND_SYNC);
					break;
				//}
			}
			
			count = 0;
			countCycle++;

		}

	}
	destroyAllWindows();
	out.release();
	m.cap.release();
	return 0;
}
