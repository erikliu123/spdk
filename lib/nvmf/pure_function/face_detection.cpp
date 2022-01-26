#include "dlib/image_processing/frontal_face_detector.h"
//#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
//#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/opencv.h"

#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>

//由于dlib和opencv中有相当一部分类同名，故不能同时对它们使用using namespace，否则会出现一些莫名其妙的问题
// using namespace dlib;
using namespace std;


// using namespace cv;
//  void func()
//  {
//      String mmodModelPath = "./mmod_human_face_detector.dat";
//      net_type mmodFaceDetector;
//      deserialize(mmodModelPath) >> mmodFaceDetector;

//     // Convert OpenCV image format to Dlib's image format
//     cv_image<bgr_pixel> dlibIm(frameDlibMmodSmall);
//     matrix<rgb_pixel> dlibMatrix;
//     assign_image(dlibMatrix, dlibIm);

//     // Detect faces in the image
//     std::vector<dlib::mmod_rect> faceRects = mmodFaceDetector(dlibMatrix);

//     for ( size_t i = 0; i < faceRects.size(); i++ )
//     {
//         int x1 = faceRects[i].rect.left();
//         int y1 = faceRects[i].rect.top();
//         int x2 = faceRects[i].rect.right();
//         int y2 = faceRects[i].rect.bottom();
//         cv::rectangle(frameDlibMmod, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
//     }
// }
#define SCALE 14
#define cR (int)(0.299 * (1 << SCALE) + 0.5)
#define cG (int)(0.587 * (1 << SCALE) + 0.5)
#define cB ((1 << SCALE) - cR - cG)

#define descale(x, n) (((x) + (1 << ((n)-1))) >> (n))

void icvCvt_BGR2Gray(const uchar *bgr, int bgr_step,
                     uchar *gray, int gray_step,
                     int height, int width, int _swap_rb)
{
    int i;
    for (; height--; gray += gray_step)
    {
        short cBGR0 = cB;
        short cBGR2 = cR;
        if (_swap_rb)
            std::swap(cBGR0, cBGR2);
        for (i = 0; i < width; i++, bgr += 3)
        {
            int t = descale(bgr[0] * cBGR0 + bgr[1] * cG + bgr[2] * cBGR2, SCALE);
            gray[i] = (uchar)t;
        }

        bgr += bgr_step - width * 3;
    }
}

void line_one_face_detections(cv::Mat img, std::vector<dlib::full_object_detection> fs)
{
    int i, j;
    for (j = 0; j < fs.size(); j++)
    {
        cv::Point p1, p2;
        for (i = 0; i < 67; i++)
        {
            // 下巴到脸颊 0 ~ 16
            //左边眉毛 17 ~ 21
            //右边眉毛 21 ~ 26
            //鼻梁     27 ~ 30
            //鼻孔        31 ~ 35
            //左眼        36 ~ 41
            //右眼        42 ~ 47
            //嘴唇外圈  48 ~ 59
            //嘴唇内圈  59 ~ 67
            switch (i)
            {
            case 16:
            case 21:
            case 26:
            case 30:
            case 35:
            case 41:
            case 47:
            case 59:
                i++;
                break;
            default:
                break;
            }

            p1.x = fs[j].part(i).x();
            p1.y = fs[j].part(i).y();
            p2.x = fs[j].part(i + 1).x();
            p2.y = fs[j].part(i + 1).y();
            cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 2, 4, 0);
        }
    }
}


// void cnn_process(uchar *read_ptr)
// {
//     for (int i = 0; i < 360; i++)
//     {
//         for (int j = 0; j < 1080; j++)
//         {
//             int src_index = 3 * (i * 1080 + j) + 0x8a;
//             int dst_index = 3 * ((719 - i) * 1080 + j) + 0x8a;
//             // int src_index = index + 0x8a;
//             for (int k = 0; k < 3; k++)
//                 std::swap(read_ptr[src_index + k], read_ptr[dst_index + k]);
//         }
//     }
//     // memcpy(temp, read_ptr + 0x8a, ndp_req->total_len - 0x8a);
//     cv::Mat frame(720, 1080, CV_8UC3, read_ptr + 0x8a); //+ 0x8a);//1080*3);//read_ptr + 0x8a); //简单粗暴处理先
//     // frame.setDefaultAllocator()
//     cv::Mat dst = frame.clone();
//     //提取灰度图
//     cv::cvtColor(frame, dst, CV_BGR2GRAY);
//     net_type net;
//     dlib::matrix<rgb_pixel> img;//读取数据
//     //调用assign_image()方法，不仅可以完成类型转化，而且，按照文档里说的，不用对传入的数据进行拷贝，所以虽然在进行数据类型转换，但是耗费的时间比较低。
//     //dlib::assign_image(dimg, dlib::cv_image<uchar>(dst));
//     dlib::assign_image(img, cv_image<uchar>(dst)); //cv_image<rgb_pixel>(dst)会报错
//     load_image(img, "/mnt/ndp/test.bmp");
//     /*TODO：malloc有问题，先不管了。。*/
//     dlib::deserialize("/home/femu/spdk/lib/nvmf/data/mmod_human_face_detector.dat") >> net; //存在问题 
//     std::vector<dlib::mmod_rect> dets_temp = net(img);
//     std::cout << "in CNN, Number of faces detected: " << dets_temp.size() << std::endl;
// }

void direct_process(uchar *read_ptr)
{

    for (int i = 0; i < 360; i++)
    {
        for (int j = 0; j < 1080; j++)
        {
            int src_index = 3 * (i * 1080 + j) + 0x8a;
            int dst_index = 3 * ((719 - i) * 1080 + j) + 0x8a;
            // int src_index = index + 0x8a;
            for (int k = 0; k < 3; k++)
                std::swap(read_ptr[src_index + k], read_ptr[dst_index + k]);
        }
    }
    // memcpy(temp, read_ptr + 0x8a, ndp_req->total_len - 0x8a);
    cv::Mat frame(720, 1080, CV_8UC3, read_ptr + 0x8a); //+ 0x8a);//1080*3);//read_ptr + 0x8a); //简单粗暴处理先
    // frame.setDefaultAllocator()
    cv::Mat dst = frame.clone();
    //提取灰度图
    cv::cvtColor(frame, dst, CV_BGR2GRAY);

    //加载dlib的人脸识别器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector(); //这个地方为什么报错了？？

    //加载人脸形状探测器, TODO:从spdk bdev加载
    dlib::shape_predictor sp;
    // dlib::deserialize("/mnt/ndp/shape_predictor_68_face_landmarks.dat") >> sp;
    dlib::deserialize("/home/femu/spdk/lib/nvmf/data/shape_predictor_68_face_landmarks.dat") >> sp;
    // Mat转化为dlib的matrix
    dlib::array2d<dlib::bgr_pixel> dimg;
    //调用assign_image()方法，不仅可以完成类型转化，而且，按照文档里说的，不用对传入的数据进行拷贝，所以虽然在进行数据类型转换，但是耗费的时间比较低。
    dlib::assign_image(dimg, dlib::cv_image<uchar>(dst));

    //获取一系列人脸所在区域
    std::vector<dlib::rectangle> dets = detector(dimg);
    int i = 0;
    std::cout << "Number of faces detected: " << dets.size() << std::endl;

    if (dets.size() == 0)
    {

        return;
    }

    for (i = 0; i < dets.size(); i++)
    {
        //画出人脸所在区域
        cv::Rect r;
        r.x = dets[i].left();
        r.y = dets[i].top();
        r.width = dets[i].width();
        r.height = dets[i].height();
        printf("%d %d %d %d\n", r.x, r.y, r.width, r.height);
    }
    fflush(stdout);

    //获取人脸特征点分布
    std::vector<dlib::full_object_detection> shapes;

    for (i = 0; i < dets.size(); i++) //为什么出现内存溢出
    {
        dlib::full_object_detection shape = sp(dimg, dets[i]); //获取指定一个区域的人脸形状
        shapes.push_back(shape);
    }

    //指出每个检测到的人脸的位置
    for (i = 0; i < dets.size(); i++)
    {
        //画出人脸所在区域
        cv::Rect r;
        r.x = dets[i].left();
        r.y = dets[i].top();
        r.width = dets[i].width();
        r.height = dets[i].height();
        cv::rectangle(frame, r, cv::Scalar(0, 0, 255), 1, 1, 0);
    }

    line_one_face_detections(frame, shapes);
    //把结果返回

    fflush(stdout);
    cv::imwrite("/mnt/ndp/FOOL_src1.jpg", frame);
}

uchar *g_buf; //[3 * 1024 * 1024];
int main(int argc, char *argv[])
{
#ifndef DEBUG
    if (argc != 2)
    {
        std::cout << "you should specified a picture!" << std::endl;
        return 0;
    }
#else
    int fd = open("/mnt/ndp/test.bmp", O_RDONLY);
    g_buf = (uchar *)malloc(3 * 1024 * 1024);
    assert(fd > 0 && g_buf);
    int cnt = read(fd, g_buf, 3 * 1024 * 1024);
    direct_process(g_buf);
    //cnn_process(g_buf);
    return 0;
#endif
    cv::Mat frame = cv::imread(argv[1]); // TODO:IO时延统计
    cv::Mat dst;
    // frame.u->data
    //读取文件数据
    //提取灰度图
    cv::cvtColor(frame, dst, CV_BGR2GRAY);

    //加载dlib的人脸识别器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    //加载人脸形状探测器
    dlib::shape_predictor sp;
    dlib::deserialize("../data/shape_predictor_68_face_landmarks.dat") >> sp;

    // Mat转化为dlib的matrix
    dlib::array2d<dlib::bgr_pixel> dimg;
    dlib::assign_image(dimg, dlib::cv_image<uchar>(dst));

    //获取一系列人脸所在区域
    std::vector<dlib::rectangle> dets = detector(dimg);
    std::cout << "Number of faces detected: " << dets.size() << std::endl;

    if (dets.size() == 0)
        return 0;

    //获取人脸特征点分布
    std::vector<dlib::full_object_detection> shapes;
    int i = 0;
    for (i = 0; i < dets.size(); i++)
    {
        dlib::full_object_detection shape = sp(dimg, dets[i]); //获取指定一个区域的人脸形状
        shapes.push_back(shape);
    }

    //指出每个检测到的人脸的位置
    for (i = 0; i < dets.size(); i++)
    {
        //画出人脸所在区域
        cv::Rect r;
        r.x = dets[i].left();
        r.y = dets[i].top();
        r.width = dets[i].width();
        r.height = dets[i].height();
        cv::rectangle(frame, r, cv::Scalar(0, 0, 255), 1, 1, 0);
    }

    line_one_face_detections(frame, shapes);

    // cv::imshow("frame", frame);
    cv::imwrite("src1.jpg", frame);
    // cv::waitKey(0);
    return 0;
}