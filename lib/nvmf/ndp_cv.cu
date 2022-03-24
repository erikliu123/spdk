#include "dlib/image_processing/frontal_face_detector.h"
//#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include "dlib/opencv.h"
#include "dlib/dnn.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <ctime>
#include <fcntl.h>

#include "spdk/env.h"
extern "C"
{
/*加入extern是为了去除警告 -- warning #1556-D: name linkage conflicts with previous declaration of variable "SPDK_LOG_ndp"*/
#include "nvmf_internal.h"
#include "spdk/log.h" 
#include "spdk/bdev.h"
#include "spdk/bdev_module.h"
};

#include "ndp.h"
#include "ndp_cv.h"
#include "ndp/helper_cuda.h"
#include "file_info.h"


// extern int produce_fsinfo(const char *path, int depth);
extern std::map<std::string, std::pair<int64_t, std::vector<file_extent>>> file_lbas_map; //文件名和对应的lba数组


void line_one_face_detections(cv::Mat img, std::vector<dlib::full_object_detection> fs);
// extern "C" int spdk_nvmf_request_complete(spdk_nvmf_request *req); //必须要显示定义成C调用

using namespace dlib;

template <long num_filters, typename SUBNET>
using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = relu<affine<con5<45, SUBNET>>>;

#define FACE_FEATURE
using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

void ndp_face_detection_complete(struct spdk_bdev_io *bdev_io, bool success,
                                        void *cb_arg)
{
    
    struct ndp_subrequest *sub_req = (struct ndp_subrequest *)cb_arg;
    struct ndp_request *ndp_req = sub_req->req;
    struct spdk_nvmf_request *req = ndp_req->nvmf_req;
    struct spdk_nvme_cpl *response = &req->rsp->nvme_cpl;//可能会因为超时被释放掉
    bool cnn_flag = false;
    int sc = 0, sct = 0;
    int ret = 0;
    uint32_t cdw0 = 0;
    bool face_feature_flag = ndp_req->task.face_detection.face_feature_flag;

    cnn_flag = ndp_req->accel; //是否使用加速场景, cdw13
    // spdk_log_set_flag("ndp");
    spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc); //假设中途没有数据块读取错误
    SPDK_INFOLOG(ndp, "cdw0=%d sct=%d, sc=%d\n", cdw0, sct, sc);
    // response->cdw0 = cdw0;
    // response->status.sc = sc;
    // response->status.sct = sct;
    //统计读取的数据是否完成，完成才完成请求

    sub_req->read_bdev_blocks++;
    spdk_bdev_free_io(bdev_io);
    SPDK_INFOLOG(ndp, "continuous blk: %d, total read blocks:%d\n", sub_req->read_bdev_blocks, sub_req->total_bdev_blocks);
    if (sub_req->read_bdev_blocks == sub_req->total_bdev_blocks)
    {
        //读取人脸
        sub_req->end_io_time = spdk_get_ticks();
        ndp_req->total_io_time += (sub_req->end_io_time - sub_req->start_io_time)/3500;
        ret = ndp_compute_face_detection(sub_req);
        spdk_ndp_request_complete(sub_req, ret);
    }
}

extern "C"
{
    int process_image(struct ndp_subrequest *sub_req, char *input_name)
    {
        //读取文件
        int ret = ndp_read_file(sub_req, input_name, ndp_face_detection_complete, sub_req);
        if (ret)
            return ret;

        if (!sub_req->req->spdk_read_flag)
        {
            ret = ndp_compute_face_detection(sub_req);
            return ret;
        }
        return 0;
    }
};

/*对接同步(read)和异步(SPDK)完成两种行为， 同步行为的话*/
//非SPDK读取，那么针对错误情况无需结束请求，只要释放之前分配的内存空间即可
int ndp_compute_face_detection(struct ndp_subrequest *sub_req)
{
     struct ndp_request *ndp_req = sub_req->req;
    unsigned char *tempbuf = sub_req->read_ptr;
    bool spdk_read_flag = ndp_req->spdk_read_flag;
    bool cnn_flag = ndp_req->accel;
    bool face_feature_flag = ndp_req->task.face_detection.face_feature_flag;

    for (int i = 0; i < 360; i++)
    {
        for (int j = 0; j < 1080; j++)
        {
            int src_index = 3 * (i * 1080 + j) + 0x8a;
            int dst_index = 3 * ((719 - i) * 1080 + j) + 0x8a;
            // int src_index = index + 0x8a;
            for (int k = 0; k < 3; k++)
                std::swap(tempbuf[src_index + k], tempbuf[dst_index + k]);
        }
    }

#if 0        
        //for(int i=0; i< 100 * 256; i += 256)
        for(int i = 0; i < 500; i ++)
        {
            if(i%16 == 0)
            {
                printf("\n%08x: ", i);
            }
            if(1 || sub_req->read_ptr[i] != 0)
            {
               
                printf("%02x ", sub_req->read_ptr[i]);
            }
        }
        putchar('\n');
#endif
    cv::Mat frame(720, 1080, CV_8UC3, sub_req->read_ptr + 0x8a);
    cv::Mat dst; //(720, 1080, CV_8UC1, temp);
                 // assert(dst.u->data != frame.u->data);
                 //提取灰度图
    cv::cvtColor(frame, dst, CV_BGR2GRAY); // munmap_chunk失败了(2022.1.25)
    std::vector<dlib::rectangle> dets;
    std::vector<dlib::mmod_rect> dets_cnn;
    dlib::array2d<dlib::bgr_pixel> dimg;
    if (!cnn_flag)
    {
        //加载dlib的人脸识别器
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector(); //这个地方为什么报错了？？

        /* Mat转化为dlib的matrix */
        //调用assign_image()方法，不仅可以完成类型转化，不用对传入的数据进行拷贝，所以虽然在进行数据类型转换，但是耗费的时间比较低。
        dlib::assign_image(dimg, dlib::cv_image<uchar>(dst));

        //获取一系列人脸所在区域
        dets = detector(dimg);
    }
    else
    { //用到GPU加速，所以有时执行很慢
        // std::vector<dlib::rectangle> dets;
        net_type net;
        dlib::matrix<rgb_pixel> img; //读取数据
        //调用assign_image()方法，不仅可以完成类型转化，而且，按照文档里说的，不用对传入的数据进行拷贝，所以虽然在进行数据类型转换，但是耗费的时间比较低。
        // dlib::assign_image(dimg, dlib::cv_image<uchar>(dst));
        dlib::assign_image(img, cv_image<uchar>(dst)); // cv_image<rgb_pixel>(dst)会报错
        /*TODO：malloc有问题，先不管了。。*/
        dlib::deserialize("/home/femu/spdk/lib/nvmf/data/mmod_human_face_detector.dat") >> net; //存在问题
        dets_cnn = net(img);
        // dlib::cnn_face_detection_model_v1  model = dlib::cnn_face_detection_model_v1("/home/femu/spdk/lib/nvmf/data/mmod_human_face_detector.dat");//
    }
    if (cnn_flag)
    {
        for (auto x : dets_cnn)
            dets.push_back(x.rect);
    }

    if (dets.size() == 0)
    {

        SPDK_ERRLOG("#####\t no face detected \t#####\n");
        // spdk_ndp_request_complete(sub_req ,ERR_NO_FACE);
        return -ERR_NO_FACE;
    }

    SPDK_INFOLOG(ndp, "### accel:%d ###\tNumber of faces detected: %Lu\n", ndp_req->accel, dets.size());

    //获取人脸特征点分布
    if (face_feature_flag)
    {
        int i = 0;
        //加载人脸形状探测器, TODO:从spdk bdev加载
        dlib::shape_predictor sp;
        // dlib::deserialize("/mnt/ndp/shape_predictor_68_face_landmarks.dat") >> sp;
        dlib::deserialize("/home/femu/spdk/lib/nvmf/data/shape_predictor_68_face_landmarks.dat") >> sp;
        std::vector<dlib::full_object_detection> shapes;
        for (i = 0; i < dets.size(); i++) // TODO 什么出现malloc失败？
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
    }
    sub_req->end_compute_time = spdk_get_ticks(); //计算结束时间
    //打印时间戳
    SPDK_INFOLOG(ndp, "file[%s], exclude malloc, total time: %lu, SPDK malloc time: %lu , SPDK IO consume: %lu, computation time: %lu\n", sub_req->read_file, (sub_req->end_compute_time - sub_req->start_io_time)/3500,  (sub_req->start_io_time - sub_req->malloc_time)/3500, (sub_req->end_io_time - sub_req->start_io_time)/3500, (sub_req->end_compute_time - sub_req->end_io_time)/3500);
    
    if (!spdk_read_flag)//
    {
        cv::imwrite("/mnt/ndp/normal_path_face_detect.jpg", frame);
    }
    else
    {
        cv::imwrite("/mnt/ndp/spdk_face_detect.jpg", frame);
    }
    // spdk_ndp_request_complete(sub_req ,0);
    return 0;
}

/*描绘人脸特征*/
void line_one_face_detections(cv::Mat img, std::vector<dlib::full_object_detection> fs)
{
    int i, j;
    for (j = 0; j < (int)fs.size(); j++)
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
