#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>

using namespace cv;
using namespace std;
using namespace dlib;

// Network Definition
/////////////////////////////////////////////////////////////////////////////////////////////////////
template <long num_filters, typename SUBNET>
using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
/////////////////////////////////////////////////////////////////////////////////////////////////////

void detectFaceDlibMMOD(net_type mmodFaceDetector, Mat &frameDlibMmod)
{

    int frameHeight = frameDlibMmod.rows;
    int frameWidth = frameDlibMmod.cols;

    // Convert OpenCV image format to Dlib's image format
    cv_image<bgr_pixel> dlibIm(frameDlibMmod);
    matrix<rgb_pixel> dlibMatrix;
    assign_image(dlibMatrix, dlibIm);

    // Detect faces in the image
    std::vector<dlib::mmod_rect> faceRects = mmodFaceDetector(dlibMatrix);
    std::cout << "face number: " << faceRects.size() << std::endl;
   
}

int main(int argc, const char **argv)
{
    String mmodModelPath = "/home/femu/spdk/lib/nvmf/data/mmod_human_face_detector.dat";
    net_type mmodFaceDetector;
    deserialize(mmodModelPath) >> mmodFaceDetector;

    Mat frame;
    frame=imread("/mnt/ndp/test.bmp");

    double tt_dlibMmod = 0;
    double fpsDlibMmod = 0;

    double t = cv::getTickCount();
    detectFaceDlibMMOD(mmodFaceDetector, frame);
    tt_dlibMmod = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    fpsDlibMmod = 1 / tt_dlibMmod;
    return 0;
}