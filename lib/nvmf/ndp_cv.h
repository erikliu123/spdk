#ifndef __NDP_CV_H__
#define __NDP_CV_H__

int ndp_compute_face_detection(struct ndp_subrequest *ndp_req);//最终的函数都会调用它

void ndp_face_detection_complete(struct spdk_bdev_io *bdev_io, bool success,
                                        void *cb_arg);

#endif                                        