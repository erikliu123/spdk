#include "dlib/dlib/image_processing/frontal_face_detector.h"
#include "dlib/dlib/image_processing/render_face_detections.h"
#include "dlib/dlib/image_processing.h"
#include "dlib/dlib/gui_widgets.h"
#include "dlib/dlib/image_io.h"
#include "dlib/dlib/opencv.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <ctime>
#include <fcntl.h>

#include "spdk/env.h"
#include "spdk/log.h"

#include "ndp.h"
#include "ndp/helper_cuda.h"
#include "file_info.h"

extern int produce_fsinfo(const char *path, int depth);
#if 1
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
#endif


extern std::map<std::string, std::pair<int64_t, std::vector<file_extent>>> file_lbas_map; //文件名和对应的lba数组

extern "C"{
#include "nvmf_internal.h"
};

#define SPDK_READ
#ifdef SPDK_READ
#include "spdk/env.h"
#include "spdk/bdev.h"
#include "spdk/bdev_module.h"
//extern "C" int spdk_nvmf_request_complete(spdk_nvmf_request *req); //必须要显示定义成C调用


static void ndp_face_detection_complete(struct spdk_bdev_io *bdev_io, bool success,
                                        void *cb_arg)
{
    struct ndp_request *ndp_req = (struct ndp_request *)cb_arg;
    struct spdk_nvmf_request *req = ndp_req->nvmf_req;
    struct spdk_nvme_cpl *response = &req->rsp->nvme_cpl;
    int  sc = 0, sct = 0;
    uint32_t cdw0 = 0;

    spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc);//假设中途没有数据块读取错误
    SPDK_NOTICELOG("cdw0=%d sct=%d, sc=%d\n",cdw0, sct, sc);
    response->cdw0 = cdw0;
    response->status.sc = sc;
    response->status.sct = sct;
    //统计读取的数据是否完成，完成才完成请求

    ndp_req->read_bdev_blocks++;
    spdk_bdev_free_io(bdev_io);
    SPDK_NOTICELOG("continuous blk: %d, total read blocks:%d\n", ndp_req->read_bdev_blocks, ndp_req->total_bdev_blocks);
    if (ndp_req->read_bdev_blocks == ndp_req->total_bdev_blocks)
    {
        //读取人脸
        ndp_req->end_io_time = spdk_get_ticks();
        //拷贝一下
        uchar *temp = new uchar[3 * 1024 * 1024];
        // uchar *temp = (uchar *)malloc(3*1024*1024) ;//uchar[3*1024*1024/*ndp_req->total_len*/];//(char *)malloc
        //[1]);
        //spdk_dma_free(ndp_req->ptr.read_ptr);
        //ndp_req->ptr.read_ptr = temp;
        SPDK_NOTICELOG("IO consume: %d ticks, HEADER[%x]\n", ndp_req->end_io_time - ndp_req->start_io_time, ndp_req->ptr.read_ptr[0]
        );
#if 0        
        for(int i=0; i< 100 * 256; i += 256)//为什么读出来全都是0。。。
        {
            if(i%16 == 0)
            {
                printf("\n%08x: ", i);
            }
            if(1 || ndp_req->ptr.read_ptr[i] != 0)
            {
               
                printf("%02x ", ndp_req->ptr.read_ptr[i]);
            }
        }
        putchar('\n');
        fflush(stdout);
        // return ;
#endif  
        //必须要将图片进行翻转！    
        for(int i = 0; i < 360; i++)
        {
            for(int j = 0; j < 1080; j++)
            {
                int src_index = 3*(i*1080 + j) + 0x8a;
                int dst_index = 3*((719-i)*1080 + j) + 0x8a;
                //int src_index = index + 0x8a;
                for(int k=0; k<3; k++)
                std::swap(ndp_req->ptr.read_ptr[src_index + k], ndp_req->ptr.read_ptr[dst_index + k]);

            }
        }
        memcpy(temp, ndp_req->ptr.read_ptr + 0x8a, ndp_req->total_len);
        //memcpy(temp, ndp_req->ptr.read_ptr + 0x8a, ndp_req->total_len - 0x8a);
        cv::Mat frame(720, 1080, CV_8UC3, ndp_req->ptr.read_ptr + 0x8a); //1080*3);//ndp_req->ptr.read_ptr + 0x8a); //简单粗暴处理先
        //frame.setDefaultAllocator()
        cv::Mat dst(720, 1080, CV_8UC3, temp);
        //= frame.clone();
        //assert(dst.u->data != frame.u->data);
        //提取灰度图
        cv::cvtColor(frame, dst, CV_BGR2GRAY);

        //加载dlib的人脸识别器
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();//这个地方为什么报错了？？

        //加载人脸形状探测器, TODO:从spdk bdev加载
        dlib::shape_predictor sp;
        //dlib::deserialize("/mnt/ndp/shape_predictor_68_face_landmarks.dat") >> sp;
        dlib::deserialize("/home/femu/spdk/lib/nvmf/data/shape_predictor_68_face_landmarks.dat") >>sp;
        // Mat转化为dlib的matrix
        dlib::array2d<dlib::bgr_pixel> dimg;
        //调用assign_image()方法，不仅可以完成类型转化，而且，按照文档里说的，不用对传入的数据进行拷贝，所以虽然在进行数据类型转换，但是耗费的时间比较低。
        dlib::assign_image(dimg, dlib::cv_image<uchar>(dst));

        //获取一系列人脸所在区域
        std::vector<dlib::rectangle> dets = detector(dimg);
        int i = 0;
        std::cout << "Number of faces detected: " << dets.size() << std::endl;

        
        //dlib::load_image(dimg, "/mnt/ndp/test.bmp");// what():  bmp load error 6: header too small
        
        if (dets.size() == 0){
            SPDK_NOTICELOG("#######face detection call finished!#####\n");
            spdk_nvmf_request_complete(req);
            free(temp);
        
            free(ndp_req);
            return;
        }

        // for (i = 0; i < dets.size(); i++)
        // {
        //     //画出人脸所在区域
        //     cv::Rect r;
        //     r.x = dets[i].left();
        //     r.y = dets[i].top();
        //     r.width = dets[i].width();
        //     r.height = dets[i].height();
        //     printf("%d %d %d %d\n", r.x, r.y, r.width, r.height);
        // }
        // fflush(stdout);

        //获取人脸特征点分布
        std::vector<dlib::full_object_detection> shapes;
        for (i = 0; i < dets.size(); i++)//TODO 什么出现malloc失败？
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
        // cv::imshow("frame", frame);
        SPDK_NOTICELOG("#######face detection call finished!#####\n");
        fflush(stdout);
        spdk_nvmf_request_complete(req);
        //free(temp);
        //free(ndp_req->ptr.read_ptr);
        spdk_dma_free(ndp_req->ptr.read_ptr);
        free(ndp_req);
        cv::imwrite("/mnt/ndp/spdk_src1.jpg", frame);
    }
}

#endif

extern "C"
{
    int process_image(struct ndp_request *ndp_req, char *input_name)
    {
#ifdef SPDK_READ
        //在mnt/ndp下读取数据
        //得到IO时延，必须是direct IO
        //TODO:调用SPDK读取数据块, 获取IO时延
        /*读取数据，离散的更好*/
        // get LBAlist
        // extent_num = walk_fibmap(fd, &statbuf, BLKSIZE, BLKSIZE / SECTORSIZE, 0, lba);
        // Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);/
        std::pair<int64_t, std::vector<file_extent>> lba;
        if (file_lbas_map.find(input_name) != file_lbas_map.end()) //获取文件大小、元数据
            lba = file_lbas_map[input_name];
        else
        {
            return -ENOENT;
        }

        unsigned char *readbuf = (unsigned char *)spdk_dma_zmalloc(lba.first + BLK_SIZE, 0x200000, NULL);
       
        assert(readbuf != NULL);
        readbuf[0] = 0x30;
        //  char *readbuf = (char *)malloc(lba.first + BLKSIZE);
        ndp_req->read_bdev_blocks = 0;
        ndp_req->total_bdev_blocks = lba.second.size();
        ndp_req->ptr.read_ptr = readbuf;
        ndp_req->total_len = lba.first;
        // spdk_bdev_open_ext(BDEV_NAME, true, NULL, NULL,&desc);
        // SPDK_NOTICELOG("OPEN DEVICE SUUCEESFULLY");
        // io_ch = spdk_bdev_get_io_channel(desc);
        ndp_req->start_io_time = spdk_get_ticks();
        int offset = 0;
        for (int i = 0; i < lba.second.size(); i++)
        {
            // struct spdk_nvme_cmd nvme_cmd;
            // nvme_cmd.nsid = 0x1;
            // nvme_cmd.opc = 0x2;
            // nvme_cmd.cdw10 = 0x0;
            // nvme_cmd.cdw11 = 0x0;
            //spdk_bdev_nvme_io_passthru(ndp_req->desc, ndp_req->io_ch, &nvme_cmd, readbuf, 0x100, ndp_face_detection_complete, ndp_req);
            spdk_bdev_read(ndp_req->desc, ndp_req->io_ch, readbuf + offset, lba.second[i].first_block<< FILE_SYS_BIT , lba.second[i].block_count << BLK_SHIFT_BIT, ndp_face_detection_complete, ndp_req);
            // spdk_bdev_read(ndp_req->desc, ndp_req->io_ch, readbuf + offset, 0x400 , 512, ndp_face_detection_complete, ndp_req);
            offset += lba.second[i].block_count << BLK_SHIFT_BIT;
            SPDK_NOTICELOG("read lba[%d], len[%d]\n", lba.second[i].first_block, lba.second[i].block_count<<BLK_SHIFT_BIT);
        }
        return 0;
#endif
        /*完全可行的路，再把IO转成直接IO更好。。现在相当于文件预取，没有完整的说明盘内带宽的优势。。。*/
        cv::Mat frame = cv::imread(input_name);
        cv::Mat dst;
        //提取灰度图
        cv::cvtColor(frame, dst, CV_BGR2GRAY);

        //加载dlib的人脸识别器
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

        //加载人脸形状探测器
        dlib::shape_predictor sp;
        dlib::deserialize("/home/femu/spdk/lib/nvmf/data/shape_predictor_68_face_landmarks.dat") >> sp;

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
        //结束请求
         struct spdk_nvme_cpl *response = &ndp_req->nvmf_req->rsp->nvme_cpl;
        response->cdw0 = 0;
        response->status.sc = 0;
        response->status.sct = 0;
        spdk_nvmf_request_complete(ndp_req->nvmf_req);
        free(ndp_req);
        // cv::imshow("frame", frame);
        cv::imwrite("/mnt/ndp/src1.jpg", frame);
        // cv::waitKey(0);
        return 0;
    }
};