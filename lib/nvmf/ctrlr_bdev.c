/*-
 *   BSD LICENSE
 *
 *   Copyright (c) Intel Corporation. All rights reserved.
 *   Copyright (c) 2019 Mellanox Technologies LTD. All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "spdk/stdinc.h"

#include "nvmf_internal.h"

#include "spdk/bdev.h"
#include "spdk/endian.h"
#include "spdk/thread.h"
#include "spdk/likely.h"
#include "spdk/nvme.h"
#include "spdk/nvmf_cmd.h"
#include "spdk/nvmf_spec.h"
#include "spdk/trace.h"
#include "spdk/scsi_spec.h"
#include "spdk/string.h"
#include "spdk/util.h"

#include "spdk/log.h"
//extern "C"{
#include "ndp.h"
//}
extern  int ndp_decompress(const char *filename, uint8_t **result, int opt);
extern	int process_image(struct ndp_request *ndp_req, char *input_name);
extern int register_ndp_task(struct ndp_request *ndp_req);
extern struct ndp_request *get_ndp_req_from_id(int id);
extern int alloc_and_start_sub_req(struct ndp_request *ndp_req);

static bool
nvmf_subsystem_bdev_io_type_supported(struct spdk_nvmf_subsystem *subsystem,
				      enum spdk_bdev_io_type io_type)
{
	struct spdk_nvmf_ns *ns;

	for (ns = spdk_nvmf_subsystem_get_first_ns(subsystem); ns != NULL;
	     ns = spdk_nvmf_subsystem_get_next_ns(subsystem, ns)) {
		if (ns->bdev == NULL) {
			continue;
		}

		if (!spdk_bdev_io_type_supported(ns->bdev, io_type)) {
			SPDK_DEBUGLOG(nvmf,
				      "Subsystem %s namespace %u (%s) does not support io_type %d\n",
				      spdk_nvmf_subsystem_get_nqn(subsystem),
				      ns->opts.nsid, spdk_bdev_get_name(ns->bdev), (int)io_type);
			return false;
		}
	}

	SPDK_DEBUGLOG(nvmf, "All devices in Subsystem %s support io_type %d\n",
		      spdk_nvmf_subsystem_get_nqn(subsystem), (int)io_type);
	return true;
}

bool
nvmf_ctrlr_dsm_supported(struct spdk_nvmf_ctrlr *ctrlr)
{
	return nvmf_subsystem_bdev_io_type_supported(ctrlr->subsys, SPDK_BDEV_IO_TYPE_UNMAP);
}

bool
nvmf_ctrlr_write_zeroes_supported(struct spdk_nvmf_ctrlr *ctrlr)
{
	return nvmf_subsystem_bdev_io_type_supported(ctrlr->subsys, SPDK_BDEV_IO_TYPE_WRITE_ZEROES);
}

static void
nvmf_ndp_complete_cmd(struct spdk_bdev_io *bdev_io, bool success,
			     void *cb_arg)
{
	struct ndp_request *ndp_req = cb_arg;
	struct spdk_nvmf_request	*req = ndp_req->nvmf_req;
	struct spdk_nvme_cpl		*response = &req->rsp->nvme_cpl;
	int				first_sc = 0, first_sct = 0, sc = 0, sct = 0;
	uint32_t			cdw0 = 0;
	struct spdk_nvmf_request	*first_req = req->first_fused_req;
	int ret = 0;

	if (spdk_unlikely(first_req != NULL)) {
		/* fused commands - get status for both operations */
		struct spdk_nvme_cpl *first_response = &first_req->rsp->nvme_cpl;

		spdk_bdev_io_get_nvme_fused_status(bdev_io, &cdw0, &first_sct, &first_sc, &sct, &sc);
		first_response->cdw0 = cdw0;
		first_response->status.sc = first_sc;
		first_response->status.sct = first_sct;
		SPDK_NOTICELOG("unlikely thing happens\n");
		/* first request should be completed */
		spdk_nvmf_request_complete(first_req);
		req->first_fused_req = NULL;
	} else {
		spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc);
	}
	SPDK_NOTICELOG("sct=%d, sc=%d\n", sct, sc);
	response->cdw0 = cdw0;
	response->status.sc = sc;
	response->status.sct = sct;
	//统计读取的数据是否完成，完成才完成请求
	//SPDK_NOTICELOG("complete num_blocks[%d]\n", bdev_io->u.bdev.num_blocks);
	spdk_bdev_free_io(bdev_io);
	ndp_req->read_bdev_blocks += 8;
	if(ndp_req->read_bdev_blocks == 16)
	{
		uint8_t *buffer = NULL;
		SPDK_NOTICELOG("#######CUDA lib call#######\n");
		process_image(ndp_req, "/home/femu/spdk/lib/nvmf/data/test.jpeg");
		ret = ndp_decompress("liu", &buffer, 0);
		if(!ret)
			free(buffer);
		ret = ndp_decompress("liu", &buffer, 1);
		if(!ret)
			free(buffer);
		
		SPDK_NOTICELOG("#######CUDA lib call finished!#####\n");
		spdk_nvmf_request_complete(req);
		free(ndp_req);
	}

}

static void
nvmf_bdev_ctrlr_complete_cmd(struct spdk_bdev_io *bdev_io, bool success,
			     void *cb_arg)
{
	struct spdk_nvmf_request	*req = cb_arg;
	struct spdk_nvme_cpl		*response = &req->rsp->nvme_cpl;
	int				first_sc = 0, first_sct = 0, sc = 0, sct = 0;
	uint32_t			cdw0 = 0;
	struct spdk_nvmf_request	*first_req = req->first_fused_req;

	if (spdk_unlikely(first_req != NULL)) {
		/* fused commands - get status for both operations */
		struct spdk_nvme_cpl *first_response = &first_req->rsp->nvme_cpl;

		spdk_bdev_io_get_nvme_fused_status(bdev_io, &cdw0, &first_sct, &first_sc, &sct, &sc);
		first_response->cdw0 = cdw0;
		first_response->status.sc = first_sc;
		first_response->status.sct = first_sct;

		/* first request should be completed */
		spdk_nvmf_request_complete(first_req);
		req->first_fused_req = NULL;
	} else {
		spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc);
	}

	response->cdw0 = cdw0;
	response->status.sc = sc;
	response->status.sct = sct;

	spdk_nvmf_request_complete(req);
	spdk_bdev_free_io(bdev_io);
}

static void
nvmf_bdev_ctrlr_complete_admin_cmd(struct spdk_bdev_io *bdev_io, bool success,
				   void *cb_arg)
{
	struct spdk_nvmf_request *req = cb_arg;

	if (req->cmd_cb_fn) {
		req->cmd_cb_fn(req);
	}

	nvmf_bdev_ctrlr_complete_cmd(bdev_io, success, req);
}

void
nvmf_bdev_ctrlr_identify_ns(struct spdk_nvmf_ns *ns, struct spdk_nvme_ns_data *nsdata,
			    bool dif_insert_or_strip)
{
	struct spdk_bdev *bdev = ns->bdev;
	uint64_t num_blocks;
	uint32_t phys_blocklen;

	num_blocks = spdk_bdev_get_num_blocks(bdev);

	nsdata->nsze = num_blocks;
	nsdata->ncap = num_blocks;
	nsdata->nuse = num_blocks;
	nsdata->nlbaf = 0;
	nsdata->flbas.format = 0;
	nsdata->nacwu = spdk_bdev_get_acwu(bdev);
	if (!dif_insert_or_strip) {
		nsdata->lbaf[0].ms = spdk_bdev_get_md_size(bdev);
		nsdata->lbaf[0].lbads = spdk_u32log2(spdk_bdev_get_block_size(bdev));
		if (nsdata->lbaf[0].ms != 0) {
			nsdata->flbas.extended = 1;
			nsdata->mc.extended = 1;
			nsdata->mc.pointer = 0;
			nsdata->dps.md_start = spdk_bdev_is_dif_head_of_md(bdev);

			switch (spdk_bdev_get_dif_type(bdev)) {
			case SPDK_DIF_TYPE1:
				nsdata->dpc.pit1 = 1;
				nsdata->dps.pit = SPDK_NVME_FMT_NVM_PROTECTION_TYPE1;
				break;
			case SPDK_DIF_TYPE2:
				nsdata->dpc.pit2 = 1;
				nsdata->dps.pit = SPDK_NVME_FMT_NVM_PROTECTION_TYPE2;
				break;
			case SPDK_DIF_TYPE3:
				nsdata->dpc.pit3 = 1;
				nsdata->dps.pit = SPDK_NVME_FMT_NVM_PROTECTION_TYPE3;
				break;
			default:
				SPDK_DEBUGLOG(nvmf, "Protection Disabled\n");
				nsdata->dps.pit = SPDK_NVME_FMT_NVM_PROTECTION_DISABLE;
				break;
			}
		}
	} else {
		nsdata->lbaf[0].ms = 0;
		nsdata->lbaf[0].lbads = spdk_u32log2(spdk_bdev_get_data_block_size(bdev));
	}

	phys_blocklen = spdk_bdev_get_physical_block_size(bdev);
	assert(phys_blocklen > 0);
	/* Linux driver uses min(nawupf, npwg) to set physical_block_size */
	nsdata->nsfeat.optperf = 1;
	nsdata->nsfeat.ns_atomic_write_unit = 1;
	nsdata->npwg = (phys_blocklen >> nsdata->lbaf[0].lbads) - 1;
	nsdata->nawupf = nsdata->npwg;

	nsdata->noiob = spdk_bdev_get_optimal_io_boundary(bdev);
	nsdata->nmic.can_share = 1;
	if (ns->ptpl_file != NULL) {
		nsdata->nsrescap.rescap.persist = 1;
	}
	nsdata->nsrescap.rescap.write_exclusive = 1;
	nsdata->nsrescap.rescap.exclusive_access = 1;
	nsdata->nsrescap.rescap.write_exclusive_reg_only = 1;
	nsdata->nsrescap.rescap.exclusive_access_reg_only = 1;
	nsdata->nsrescap.rescap.write_exclusive_all_reg = 1;
	nsdata->nsrescap.rescap.exclusive_access_all_reg = 1;
	nsdata->nsrescap.rescap.ignore_existing_key = 1;

	SPDK_STATIC_ASSERT(sizeof(nsdata->nguid) == sizeof(ns->opts.nguid), "size mismatch");
	memcpy(nsdata->nguid, ns->opts.nguid, sizeof(nsdata->nguid));

	SPDK_STATIC_ASSERT(sizeof(nsdata->eui64) == sizeof(ns->opts.eui64), "size mismatch");
	memcpy(&nsdata->eui64, ns->opts.eui64, sizeof(nsdata->eui64));
}

static void
nvmf_bdev_ctrlr_get_rw_params(const struct spdk_nvme_cmd *cmd, uint64_t *start_lba,
			      uint64_t *num_blocks)
{
	/* SLBA: CDW10 and CDW11 */
	*start_lba = from_le64(&cmd->cdw10);

	/* NLB: CDW12 bits 15:00, 0's based */
	*num_blocks = (from_le32(&cmd->cdw12) & 0xFFFFu) + 1;
}

static bool
nvmf_bdev_ctrlr_lba_in_range(uint64_t bdev_num_blocks, uint64_t io_start_lba,
			     uint64_t io_num_blocks)
{
	if (io_start_lba + io_num_blocks > bdev_num_blocks ||
	    io_start_lba + io_num_blocks < io_start_lba) {
		return false;
	}

	return true;
}

static void
nvmf_ctrlr_process_io_cmd_resubmit(void *arg)
{
	struct spdk_nvmf_request *req = arg;

	nvmf_ctrlr_process_io_cmd(req);
}

static void
nvmf_ctrlr_process_admin_cmd_resubmit(void *arg)
{
	struct spdk_nvmf_request *req = arg;

	nvmf_ctrlr_process_admin_cmd(req);
}

static void
nvmf_bdev_ctrl_queue_io(struct spdk_nvmf_request *req, struct spdk_bdev *bdev,
			struct spdk_io_channel *ch, spdk_bdev_io_wait_cb cb_fn, void *cb_arg)
{
	int rc;

	req->bdev_io_wait.bdev = bdev;
	req->bdev_io_wait.cb_fn = cb_fn;
	req->bdev_io_wait.cb_arg = cb_arg;

	rc = spdk_bdev_queue_io_wait(bdev, ch, &req->bdev_io_wait);
	if (rc != 0) {
		assert(false);
	}
	req->qpair->group->stat.pending_bdev_io++;
}

bool
nvmf_bdev_zcopy_enabled(struct spdk_bdev *bdev)
{
	return spdk_bdev_io_type_supported(bdev, SPDK_BDEV_IO_TYPE_ZCOPY);
}

#include "json.h"
#include "json_object.h"
//用户发送了一个json文件, 对齐进行解析
/*
{
"Type": 0x0,  
"FileList": {
"FileRegex": false,			//文件名是否正则表达,只允许最后一个文件名可以正则表达。
"ReadFile": [[FilePath], [FilePath], ……], //如果所有文件都不存在那么，
"ReadDir": [[DirPath], [DirPath], ……],
},
"FaceDetection":
{
"AccelFlag": false,             // 十进制展示
"FeatureFlag": false,
},
"TimeOut": 200000
}

*/
// #define MAX_ITEM 1024
// #define MAX_LEN 256
// struct file_dir_alloc
// {
// 	char file_list[MAX_ITEM][MAX_LEN];//可以允许有正则表达式
// 	char dir_list[MAX_ITEM][MAX_LEN];//可以允许正则表达式
// 	bool alloc_file_flag[MAX_ITEM];
// 	bool alloc_dir_flag[MAX_ITEM];
// };
// struct file_dir_alloc g_dir_list;
#if 1
int tran_parse_ndp_task_json(const char *json_str, struct ndp_request *ndp_req);
int tran_parse_ndp_task_json(const char *json_str, struct ndp_request *ndp_req)
{
    struct json_object *jo_caps = NULL;
    struct json_object *jo = NULL;
	
    int ret = SPDK_NVME_SC_INVALID_FILE;
	
	ndp_req->dir_flag = false;
	ndp_req->reverse = false;
	
    if ((jo_caps = json_tokener_parse(json_str)) == NULL) {
        goto out;
    }

	if (json_object_object_get_ex(jo_caps, "TimeOut", &jo)) {//直接跳出来？
        if (json_object_get_type(jo) != json_type_int) {
            goto out;
        }

        errno = 0;
        ndp_req->timeout_ms = json_object_get_int(jo);

        if (errno != 0) {
            goto out;
        }
    }

	if (json_object_object_get_ex(jo_caps, "UserSpaceIOFlag", &jo)) {
        if (json_object_get_type(jo) != json_type_boolean) {
            goto out;
        }
        ndp_req->spdk_read_flag = json_object_get_boolean(jo);
		SPDK_NOTICELOG("SPDK FLAG: %d\n", ndp_req->spdk_read_flag);
    }

    if (json_object_object_get_ex(jo_caps, "ID", &jo)) {
        if (json_object_get_type(jo) != json_type_int) {
            goto out;
        }

        errno = 0;
        ndp_req->id = (int)json_object_get_int(jo);

        if (errno != 0) {
            goto out;
        }
    }

    if (json_object_object_get_ex(jo_caps, "Type", &jo)) {
        if (json_object_get_type(jo) != json_type_int) {
            goto out;
        }

        errno = 0;
        ndp_req->task_type = (int)json_object_get_int(jo);

        if (errno != 0 ||  ndp_req->task_type >= MAX_NDP_TASK) {//有问题
            goto out;
        }
    }
	
	if (json_object_object_get_ex(jo_caps, "Compress", &jo)) {
		struct json_object *jo2 = NULL;
		ndp_req->accel = false;
		ndp_req->task_type = COMPRESS;
		if (json_object_get_type(jo) != json_type_object) {
            goto out;
        }
		
		if(json_object_object_get_ex(jo, "AccelerateFlag", &jo2)) {//为什么在这里退出?
            //goto out;
			;
        }
		ndp_req->accel = (json_object_get_int(jo2) == 1);
	}

	if (json_object_object_get_ex(jo_caps, "Decompress", &jo)) {
		struct json_object *jo2 = NULL;
		ndp_req->accel = false;
		ndp_req->task_type = DECOMPRESS;
		if (json_object_get_type(jo) != json_type_object) {
            goto out;
        }
		
		if(json_object_object_get_ex(jo, "AccelerateFlag", &jo2)) {//为什么在这里退出?
            //goto out;
			;
        }
		ndp_req->accel = (json_object_get_int(jo2) == 1);
	}

	if (json_object_object_get_ex(jo_caps, "FaceDetection", &jo)) {
		struct json_object *jo2 = NULL;
		ndp_req->task_type = FACE_DETECTION;
		ndp_req->task.face_detection.cnn_flag = false;
		ndp_req->task.face_detection.face_feature_flag = false;
		if (json_object_get_type(jo) != json_type_object) {
            goto out;
        }
		
		if(json_object_object_get_ex(jo, "AccelerateFlag", &jo2)) {//为什么在这里退出?
            //goto out;
			;
        }
		// if (json_object_get_type(jo2) != json_type_boolean) {
        //     goto out;
        // }
		ndp_req->accel = ndp_req->task.face_detection.cnn_flag = (json_object_get_int(jo2) == 1);//json_object_get_boolean(jo2);
		

		if (json_object_object_get_ex(jo, "FeatureFlag", &jo2)) {
            //goto out;
			;
        }
		// if (json_object_get_type(jo2) != json_type_boolean) {
        //     goto out;
        // }
		ndp_req->task.face_detection.face_feature_flag = (json_object_get_int(jo2) == 1);//json_object_get_boolean(jo2);
    }
	

    if (json_object_object_get_ex(jo_caps, "FileList", &jo)) {//假定当前只处理一个文件或者目录
        struct json_object *jo2 = NULL;
		if (json_object_get_type(jo) != json_type_object) {
            goto out;
        }
        if (json_object_object_get_ex(jo, "ReadFile", &jo2)) {
            if (json_object_get_type(jo2) != json_type_array) {
                goto out;
            }
			
			int ele_num = json_object_array_length(jo2);
			printf("element number of file: %d\n", ele_num);
			for(int i=0; i<ele_num; i++)//读取文件数组，默认1个
			{
				struct json_object *temp = json_object_array_get_idx(jo2, i);
				const char *str = json_object_get_string(temp);
				strcpy(ndp_req->read_path, str);
				break;
			}
        }

		 if (json_object_object_get_ex(jo, "ReadDirectory", &jo2)) {
            if (json_object_get_type(jo2) != json_type_array) {
                goto out;
            }
		
			int ele_num = json_object_array_length(jo2);
            //*pgsizep = (size_t)json_object_get_int64(jo2);
			printf("element number of directory: %d\n", ele_num);
			for(int i=0; i<ele_num; i++)//读取数组
			{
				struct json_object *temp = json_object_array_get_idx(jo2, i);
				const char *str = json_object_get_string(temp);
				ndp_req->dir_flag = true;
				strcpy(ndp_req->read_path, str);
				break;
			}
        }
    }
	
	
	
    ret = 0;
out:
    /* We just need to put our top-level object. */
    json_object_put(jo_caps);
    return ret;
}
#endif

int
nvmf_ndp_execute_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
			 struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	struct spdk_nvme_cpl *response = &req->rsp->nvme_cpl;      
	
	struct spdk_nvme_cmd *cmd = &req->cmd->nvme_cmd;
	int ret;
	int taskid = 0;
	response->cdw0 = 0;
    ret = SPDK_NVME_SC_INVALID_FILE;//SPDK_NVME_SC_INVALID_FILE;
    response->status.sct = 0;

	taskid = cmd->cdw13;//任务ID
	struct ndp_request *ndp_req = get_ndp_req_from_id(taskid);
	if(ndp_req)
	{
		ndp_req->nvmf_req = req;
		ret = alloc_and_start_sub_req(ndp_req);
	}
	/* outdated way */
	// ndp_req->accel = ((cmd->cdw14 & 0x1) != 0);
	// ndp_req->spdk_read_flag =  ((cmd->cdw14 & 0x2) != 0);

	/*如果传过来了json文件，可以参考写命令读取这部分*/
	/*
	//直接执行ndp任务
	switch (taskid)
	{
	case FACE_DETECTION:
	{
		if((cmd->cdw14) & (1<<FEATURE_FLAG))
			ndp_req->task.face_detection.face_feature_flag = true;
		ret = process_image(ndp_req, "/mnt/ndp/test.bmp");
		break;
	}
	case COMPRESS:

		break;
	case DECOMPRESS:
		break;
	default:
		ret = 1;
		break;
	}
	*/
	response->status.sc = ret;
	if(ret != 0 || !ndp_req->spdk_read_flag)//异常行为或者同步IO时，需要结束请求
	{
		spdk_nvmf_request_complete(req);
		free(ndp_req);
		return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;//SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}
	

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}


int
nvmf_ndp_write_execute_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
			 struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	struct spdk_nvme_cpl *response = &req->rsp->nvme_cpl;      
	struct ndp_request *ndp_req = malloc(sizeof(struct ndp_request));
	struct spdk_nvme_cmd *cmd = &req->cmd->nvme_cmd;
	struct spdk_nvme_cpl *rsp = &req->rsp->nvme_cpl;
	uint64_t start_lba;
	uint64_t num_blocks;
	int ret;

	response->cdw0 = 0;
    response->status.sc = SPDK_NVME_SC_SUCCESS;//SPDK_NVME_SC_INVALID_FILE;
    response->status.sct = 0;

	// ndp_req->nvmf_req = req;
	ndp_req->desc = desc;
	ndp_req->io_ch = ch;
	/*如果传过来了json文件，可以参考写命令读取这部分*/
	//TODO 解析json字节流
	nvmf_bdev_ctrlr_get_rw_params(cmd, &start_lba, &num_blocks);
	if(req->iovcnt == 1)//保证传过来的json文件不超过4K大小！！
	{
		ret = tran_parse_ndp_task_json(req->iov[0].iov_base, ndp_req);
	}else{
		ret = SPDK_NVME_SC_INVALID_FILE;//错误码
		SPDK_NOTICELOG("json file is too long, io vector number [%d]\n", req->iovcnt);
		// spdk_nvmf_request_complete(req);
		// free(ndp_req);
		// return 0;

	}
	if(ret == 0)
	{
		ret = register_ndp_task(ndp_req); //查看是否重复的ID
		SPDK_NOTICELOG("register NDP task[%d] successfully\n", ndp_req->id);
	}
	response->status.sc = ret;
	//记录任务ID及对应的具体任务内容，如果json文件有问题，那么收到0x82命令时候直接返回错误
	SPDK_NOTICELOG("start lba: %ld, num blocks: %ld, sgl length %d\n", num_blocks, start_lba, req->length);
	SPDK_NOTICELOG("io vector count: %u, data: %x\n", req->iovcnt, ((int *)(req->iov[0].iov_base))[0]);
	spdk_nvmf_request_complete(req);
	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

int
nvmf_ndp_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
			 struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	uint64_t bdev_num_blocks = spdk_bdev_get_num_blocks(bdev);
	uint32_t block_size = spdk_bdev_get_block_size(bdev);
	struct spdk_nvme_cmd *cmd = &req->cmd->nvme_cmd;
	struct spdk_nvme_cpl *rsp = &req->rsp->nvme_cpl;
	uint64_t start_lba;
	uint64_t num_blocks;
	struct ndp_request *ndp_req = (struct ndp_request *)malloc(sizeof(struct ndp_request));
	int rc;
	ndp_req->nvmf_req = req;
	ndp_req->total_bdev_blocks = 2 * 8;
	ndp_req->read_bdev_blocks=0;

	nvmf_bdev_ctrlr_get_rw_params(cmd, &start_lba, &num_blocks);

	if (spdk_unlikely(!nvmf_bdev_ctrlr_lba_in_range(bdev_num_blocks, start_lba, num_blocks))) {
		SPDK_ERRLOG("end of media\n");
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_LBA_OUT_OF_RANGE;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	if (spdk_unlikely(num_blocks * block_size > req->length)) {
		SPDK_ERRLOG("Read NLB %" PRIu64 " * block size %" PRIu32 " > SGL length %" PRIu32 "\n",
			    num_blocks, block_size, req->length);
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_DATA_SGL_LENGTH_INVALID;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	if (req->zcopy_phase == NVMF_ZCOPY_PHASE_EXECUTE) {
		/* Return here after checking the lba etc */
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	assert(!spdk_nvmf_using_zcopy(req->zcopy_phase));
	start_lba = 0;
	//甚至可以不用iov
	num_blocks = 8;
	SPDK_NOTICELOG("io vector number [%d]\n", req->iovcnt);
	int iocnt_tmp = req->iovcnt/2 > 0 ? req->iovcnt/2 : 1;
	int read_len = num_blocks * block_size;
	if(req->iovcnt == 1)
	{
		req->iov[1].iov_len = read_len;
		req->iov[1].iov_base = req->iov[0].iov_base + read_len;
	}
	rc = spdk_bdev_readv_blocks(desc, ch, req->iov, iocnt_tmp, start_lba, num_blocks,
				    nvmf_ndp_complete_cmd, ndp_req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	start_lba = 16;
	num_blocks = 8;
	rc = spdk_bdev_readv_blocks(desc, ch, req->iov+iocnt_tmp, iocnt_tmp, start_lba, num_blocks,
				    nvmf_ndp_complete_cmd, ndp_req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

int
nvmf_bdev_ctrlr_read_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
			 struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	uint64_t bdev_num_blocks = spdk_bdev_get_num_blocks(bdev);
	uint32_t block_size = spdk_bdev_get_block_size(bdev);
	struct spdk_nvme_cmd *cmd = &req->cmd->nvme_cmd;
	struct spdk_nvme_cpl *rsp = &req->rsp->nvme_cpl;
	uint64_t start_lba;
	uint64_t num_blocks;
	int rc;

	nvmf_bdev_ctrlr_get_rw_params(cmd, &start_lba, &num_blocks);
	if (spdk_unlikely(!nvmf_bdev_ctrlr_lba_in_range(bdev_num_blocks, start_lba, num_blocks))) {
		SPDK_ERRLOG("end of media\n");
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_LBA_OUT_OF_RANGE;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}
#if 0	
	SPDK_NOTICELOG("Read NLB[%ld], block size[%u], SGL length[%u] \n",  
		num_blocks, block_size, req->length);
	SPDK_NOTICELOG("io vector number [%d], io vec info:\n", req->iovcnt);
	// for(int i=0; i<(int)req->iovcnt; i++)
	// {
	// 	printf("[%ld] ", req->iov[i].iov_len);
	// }
	SPDK_NOTICELOG("\n");
#endif	
#if 0	//不可以直接返回，会有BUG
//nvmf_tgt: tcp.c:365: nvmf_tcp_req_pdu_init: Assertion `tcp_req->pdu_in_use == false' failed.
	rsp->status.sct = SPDK_NVME_SCT_GENERIC;
	rsp->status.sc = SPDK_NVME_SC_SUCCESS;
	spdk_nvmf_request_complete(req);
	return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
#endif

	if (spdk_unlikely(num_blocks * block_size > req->length)) {
		SPDK_ERRLOG("Read NLB %" PRIu64 " * block size %" PRIu32 " > SGL length %" PRIu32 "\n",
			    num_blocks, block_size, req->length);
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_DATA_SGL_LENGTH_INVALID;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	if (req->zcopy_phase == NVMF_ZCOPY_PHASE_EXECUTE) {
		/* Return here after checking the lba etc */
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	assert(!spdk_nvmf_using_zcopy(req->zcopy_phase));

	rc = spdk_bdev_readv_blocks(desc, ch, req->iov, req->iovcnt, start_lba, num_blocks,
				    nvmf_bdev_ctrlr_complete_cmd, req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

int
nvmf_bdev_ctrlr_write_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
			  struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	uint64_t bdev_num_blocks = spdk_bdev_get_num_blocks(bdev);
	uint32_t block_size = spdk_bdev_get_block_size(bdev);
	struct spdk_nvme_cmd *cmd = &req->cmd->nvme_cmd;
	struct spdk_nvme_cpl *rsp = &req->rsp->nvme_cpl;
	uint64_t start_lba;
	uint64_t num_blocks;
	int rc;

	nvmf_bdev_ctrlr_get_rw_params(cmd, &start_lba, &num_blocks);

	if (spdk_unlikely(!nvmf_bdev_ctrlr_lba_in_range(bdev_num_blocks, start_lba, num_blocks))) {
		SPDK_ERRLOG("end of media\n");
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_LBA_OUT_OF_RANGE;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	if (spdk_unlikely(num_blocks * block_size > req->length)) {
		SPDK_ERRLOG("Write NLB %" PRIu64 " * block size %" PRIu32 " > SGL length %" PRIu32 "\n",
			    num_blocks, block_size, req->length);
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_DATA_SGL_LENGTH_INVALID;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	if (req->zcopy_phase == NVMF_ZCOPY_PHASE_EXECUTE) {
		/* Return here after checking the lba etc */
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	assert(!spdk_nvmf_using_zcopy(req->zcopy_phase));

	rc = spdk_bdev_writev_blocks(desc, ch, req->iov, req->iovcnt, start_lba, num_blocks,
				     nvmf_bdev_ctrlr_complete_cmd, req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

int
nvmf_bdev_ctrlr_compare_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
			    struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	uint64_t bdev_num_blocks = spdk_bdev_get_num_blocks(bdev);
	uint32_t block_size = spdk_bdev_get_block_size(bdev);
	struct spdk_nvme_cmd *cmd = &req->cmd->nvme_cmd;
	struct spdk_nvme_cpl *rsp = &req->rsp->nvme_cpl;
	uint64_t start_lba;
	uint64_t num_blocks;
	int rc;

	nvmf_bdev_ctrlr_get_rw_params(cmd, &start_lba, &num_blocks);

	if (spdk_unlikely(!nvmf_bdev_ctrlr_lba_in_range(bdev_num_blocks, start_lba, num_blocks))) {
		SPDK_ERRLOG("end of media\n");
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_LBA_OUT_OF_RANGE;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	if (spdk_unlikely(num_blocks * block_size > req->length)) {
		SPDK_ERRLOG("Compare NLB %" PRIu64 " * block size %" PRIu32 " > SGL length %" PRIu32 "\n",
			    num_blocks, block_size, req->length);
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_DATA_SGL_LENGTH_INVALID;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	rc = spdk_bdev_comparev_blocks(desc, ch, req->iov, req->iovcnt, start_lba, num_blocks,
				       nvmf_bdev_ctrlr_complete_cmd, req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

int
nvmf_bdev_ctrlr_compare_and_write_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
				      struct spdk_io_channel *ch, struct spdk_nvmf_request *cmp_req, struct spdk_nvmf_request *write_req)
{
	uint64_t bdev_num_blocks = spdk_bdev_get_num_blocks(bdev);
	uint32_t block_size = spdk_bdev_get_block_size(bdev);
	struct spdk_nvme_cmd *cmp_cmd = &cmp_req->cmd->nvme_cmd;
	struct spdk_nvme_cmd *write_cmd = &write_req->cmd->nvme_cmd;
	struct spdk_nvme_cpl *rsp = &write_req->rsp->nvme_cpl;
	uint64_t write_start_lba, cmp_start_lba;
	uint64_t write_num_blocks, cmp_num_blocks;
	int rc;

	nvmf_bdev_ctrlr_get_rw_params(cmp_cmd, &cmp_start_lba, &cmp_num_blocks);
	nvmf_bdev_ctrlr_get_rw_params(write_cmd, &write_start_lba, &write_num_blocks);

	if (spdk_unlikely(write_start_lba != cmp_start_lba || write_num_blocks != cmp_num_blocks)) {
		SPDK_ERRLOG("Fused command start lba / num blocks mismatch\n");
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_INVALID_FIELD;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	if (spdk_unlikely(!nvmf_bdev_ctrlr_lba_in_range(bdev_num_blocks, write_start_lba,
			  write_num_blocks))) {
		SPDK_ERRLOG("end of media\n");
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_LBA_OUT_OF_RANGE;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	if (spdk_unlikely(write_num_blocks * block_size > write_req->length)) {
		SPDK_ERRLOG("Write NLB %" PRIu64 " * block size %" PRIu32 " > SGL length %" PRIu32 "\n",
			    write_num_blocks, block_size, write_req->length);
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_DATA_SGL_LENGTH_INVALID;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	rc = spdk_bdev_comparev_and_writev_blocks(desc, ch, cmp_req->iov, cmp_req->iovcnt, write_req->iov,
			write_req->iovcnt, write_start_lba, write_num_blocks, nvmf_bdev_ctrlr_complete_cmd, write_req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(cmp_req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, cmp_req);
			nvmf_bdev_ctrl_queue_io(write_req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, write_req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

int
nvmf_bdev_ctrlr_write_zeroes_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
				 struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	uint64_t bdev_num_blocks = spdk_bdev_get_num_blocks(bdev);
	struct spdk_nvme_cmd *cmd = &req->cmd->nvme_cmd;
	struct spdk_nvme_cpl *rsp = &req->rsp->nvme_cpl;
	uint64_t start_lba;
	uint64_t num_blocks;
	int rc;

	nvmf_bdev_ctrlr_get_rw_params(cmd, &start_lba, &num_blocks);

	if (spdk_unlikely(!nvmf_bdev_ctrlr_lba_in_range(bdev_num_blocks, start_lba, num_blocks))) {
		SPDK_ERRLOG("end of media\n");
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_LBA_OUT_OF_RANGE;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	rc = spdk_bdev_write_zeroes_blocks(desc, ch, start_lba, num_blocks,
					   nvmf_bdev_ctrlr_complete_cmd, req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		rsp->status.sct = SPDK_NVME_SCT_GENERIC;
		rsp->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

int
nvmf_bdev_ctrlr_flush_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
			  struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	struct spdk_nvme_cpl *response = &req->rsp->nvme_cpl;
	int rc;

	/* As for NVMeoF controller, SPDK always set volatile write
	 * cache bit to 1, return success for those block devices
	 * which can't support FLUSH command.
	 */
	if (!spdk_bdev_io_type_supported(bdev, SPDK_BDEV_IO_TYPE_FLUSH)) {
		response->status.sct = SPDK_NVME_SCT_GENERIC;
		response->status.sc = SPDK_NVME_SC_SUCCESS;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	rc = spdk_bdev_flush_blocks(desc, ch, 0, spdk_bdev_get_num_blocks(bdev),
				    nvmf_bdev_ctrlr_complete_cmd, req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		response->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}
	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

struct nvmf_bdev_ctrlr_unmap {
	struct spdk_nvmf_request	*req;
	uint32_t			count;
	struct spdk_bdev_desc		*desc;
	struct spdk_bdev		*bdev;
	struct spdk_io_channel		*ch;
	uint32_t			range_index;
};

static void
nvmf_bdev_ctrlr_unmap_cpl(struct spdk_bdev_io *bdev_io, bool success,
			  void *cb_arg)
{
	struct nvmf_bdev_ctrlr_unmap *unmap_ctx = cb_arg;
	struct spdk_nvmf_request	*req = unmap_ctx->req;
	struct spdk_nvme_cpl		*response = &req->rsp->nvme_cpl;
	int				sc, sct;
	uint32_t			cdw0;

	unmap_ctx->count--;

	if (response->status.sct == SPDK_NVME_SCT_GENERIC &&
	    response->status.sc == SPDK_NVME_SC_SUCCESS) {
		spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc);
		response->cdw0 = cdw0;
		response->status.sc = sc;
		response->status.sct = sct;
	}

	if (unmap_ctx->count == 0) {
		spdk_nvmf_request_complete(req);
		free(unmap_ctx);
	}
	spdk_bdev_free_io(bdev_io);
}

static int
nvmf_bdev_ctrlr_unmap(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
		      struct spdk_io_channel *ch, struct spdk_nvmf_request *req,
		      struct nvmf_bdev_ctrlr_unmap *unmap_ctx);
static void
nvmf_bdev_ctrlr_unmap_resubmit(void *arg)
{
	struct nvmf_bdev_ctrlr_unmap *unmap_ctx = arg;
	struct spdk_nvmf_request *req = unmap_ctx->req;
	struct spdk_bdev_desc *desc = unmap_ctx->desc;
	struct spdk_bdev *bdev = unmap_ctx->bdev;
	struct spdk_io_channel *ch = unmap_ctx->ch;

	nvmf_bdev_ctrlr_unmap(bdev, desc, ch, req, unmap_ctx);
}

static int
nvmf_bdev_ctrlr_unmap(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
		      struct spdk_io_channel *ch, struct spdk_nvmf_request *req,
		      struct nvmf_bdev_ctrlr_unmap *unmap_ctx)
{
	uint16_t nr, i;
	struct spdk_nvme_cmd *cmd = &req->cmd->nvme_cmd;
	struct spdk_nvme_cpl *response = &req->rsp->nvme_cpl;
	struct spdk_nvme_dsm_range *dsm_range;
	uint64_t lba;
	uint32_t lba_count;
	int rc;

	nr = cmd->cdw10_bits.dsm.nr + 1;
	if (nr * sizeof(struct spdk_nvme_dsm_range) > req->length) {
		SPDK_ERRLOG("Dataset Management number of ranges > SGL length\n");
		response->status.sc = SPDK_NVME_SC_DATA_SGL_LENGTH_INVALID;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	if (unmap_ctx == NULL) {
		unmap_ctx = calloc(1, sizeof(*unmap_ctx));
		if (!unmap_ctx) {
			response->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
			return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
		}

		unmap_ctx->req = req;
		unmap_ctx->desc = desc;
		unmap_ctx->ch = ch;
		unmap_ctx->bdev = bdev;

		response->status.sct = SPDK_NVME_SCT_GENERIC;
		response->status.sc = SPDK_NVME_SC_SUCCESS;
	} else {
		unmap_ctx->count--;	/* dequeued */
	}

	dsm_range = (struct spdk_nvme_dsm_range *)req->data;
	for (i = unmap_ctx->range_index; i < nr; i++) {
		lba = dsm_range[i].starting_lba;
		lba_count = dsm_range[i].length;

		unmap_ctx->count++;

		rc = spdk_bdev_unmap_blocks(desc, ch, lba, lba_count,
					    nvmf_bdev_ctrlr_unmap_cpl, unmap_ctx);
		if (rc) {
			if (rc == -ENOMEM) {
				nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_bdev_ctrlr_unmap_resubmit, unmap_ctx);
				/* Unmap was not yet submitted to bdev */
				/* unmap_ctx->count will be decremented when the request is dequeued */
				return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
			}
			response->status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
			unmap_ctx->count--;
			/* We can't return here - we may have to wait for any other
				* unmaps already sent to complete */
			break;
		}
		unmap_ctx->range_index++;
	}

	if (unmap_ctx->count == 0) {
		free(unmap_ctx);
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

int
nvmf_bdev_ctrlr_dsm_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
			struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	struct spdk_nvme_cmd *cmd = &req->cmd->nvme_cmd;
	struct spdk_nvme_cpl *response = &req->rsp->nvme_cpl;

	if (cmd->cdw11_bits.dsm.ad) {
		return nvmf_bdev_ctrlr_unmap(bdev, desc, ch, req, NULL);
	}

	response->status.sct = SPDK_NVME_SCT_GENERIC;
	response->status.sc = SPDK_NVME_SC_SUCCESS;
	return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
}

int
nvmf_bdev_ctrlr_nvme_passthru_io(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
				 struct spdk_io_channel *ch, struct spdk_nvmf_request *req)
{
	int rc;

	rc = spdk_bdev_nvme_io_passthru(desc, ch, &req->cmd->nvme_cmd, req->data, req->length,
					nvmf_bdev_ctrlr_complete_cmd, req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_io_cmd_resubmit, req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		req->rsp->nvme_cpl.status.sct = SPDK_NVME_SCT_GENERIC;
		req->rsp->nvme_cpl.status.sc = SPDK_NVME_SC_INVALID_OPCODE;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

int
spdk_nvmf_bdev_ctrlr_nvme_passthru_admin(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
		struct spdk_io_channel *ch, struct spdk_nvmf_request *req,
		spdk_nvmf_nvme_passthru_cmd_cb cb_fn)
{
	int rc;

	req->cmd_cb_fn = cb_fn;

	rc = spdk_bdev_nvme_admin_passthru(desc, ch, &req->cmd->nvme_cmd, req->data, req->length,
					   nvmf_bdev_ctrlr_complete_admin_cmd, req);
	if (spdk_unlikely(rc)) {
		if (rc == -ENOMEM) {
			nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_admin_cmd_resubmit, req);
			return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
		}
		req->rsp->nvme_cpl.status.sct = SPDK_NVME_SCT_GENERIC;
		req->rsp->nvme_cpl.status.sc = SPDK_NVME_SC_INTERNAL_DEVICE_ERROR;
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}

	return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
}

static void
nvmf_bdev_ctrlr_complete_abort_cmd(struct spdk_bdev_io *bdev_io, bool success, void *cb_arg)
{
	struct spdk_nvmf_request *req = cb_arg;

	if (success) {
		req->rsp->nvme_cpl.cdw0 &= ~1U;
	}

	spdk_nvmf_request_complete(req);
	spdk_bdev_free_io(bdev_io);
}

int
spdk_nvmf_bdev_ctrlr_abort_cmd(struct spdk_bdev *bdev, struct spdk_bdev_desc *desc,
			       struct spdk_io_channel *ch, struct spdk_nvmf_request *req,
			       struct spdk_nvmf_request *req_to_abort)
{
	int rc;

	assert((req->rsp->nvme_cpl.cdw0 & 1U) != 0);

	rc = spdk_bdev_abort(desc, ch, req_to_abort, nvmf_bdev_ctrlr_complete_abort_cmd, req);
	if (spdk_likely(rc == 0)) {
		return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
	} else if (rc == -ENOMEM) {
		nvmf_bdev_ctrl_queue_io(req, bdev, ch, nvmf_ctrlr_process_admin_cmd_resubmit, req);
		return SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS;
	} else {
		return SPDK_NVMF_REQUEST_EXEC_STATUS_COMPLETE;
	}
}

bool
nvmf_bdev_ctrlr_get_dif_ctx(struct spdk_bdev *bdev, struct spdk_nvme_cmd *cmd,
			    struct spdk_dif_ctx *dif_ctx)
{
	uint32_t init_ref_tag, dif_check_flags = 0;
	int rc;

	if (spdk_bdev_get_md_size(bdev) == 0) {
		return false;
	}

	/* Initial Reference Tag is the lower 32 bits of the start LBA. */
	init_ref_tag = (uint32_t)from_le64(&cmd->cdw10);

	if (spdk_bdev_is_dif_check_enabled(bdev, SPDK_DIF_CHECK_TYPE_REFTAG)) {
		dif_check_flags |= SPDK_DIF_FLAGS_REFTAG_CHECK;
	}

	if (spdk_bdev_is_dif_check_enabled(bdev, SPDK_DIF_CHECK_TYPE_GUARD)) {
		dif_check_flags |= SPDK_DIF_FLAGS_GUARD_CHECK;
	}

	rc = spdk_dif_ctx_init(dif_ctx,
			       spdk_bdev_get_block_size(bdev),
			       spdk_bdev_get_md_size(bdev),
			       spdk_bdev_is_md_interleaved(bdev),
			       spdk_bdev_is_dif_head_of_md(bdev),
			       spdk_bdev_get_dif_type(bdev),
			       dif_check_flags,
			       init_ref_tag, 0, 0, 0, 0);

	return (rc == 0) ? true : false;
}

static void
nvmf_bdev_ctrlr_start_zcopy_complete(struct spdk_bdev_io *bdev_io, bool success,
				     void *cb_arg)
{
	struct spdk_nvmf_request	*req = cb_arg;
	struct iovec *iov;
	int iovcnt;

	if (spdk_unlikely(!success)) {
		int                     sc = 0, sct = 0;
		uint32_t                cdw0 = 0;
		struct spdk_nvme_cpl    *response = &req->rsp->nvme_cpl;
		spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc);

		response->cdw0 = cdw0;
		response->status.sc = sc;
		response->status.sct = sct;

		spdk_bdev_free_io(bdev_io);
		spdk_nvmf_request_complete(req);
		return;
	}

	spdk_bdev_io_get_iovec(bdev_io, &iov, &iovcnt);

	assert(iovcnt <= NVMF_REQ_MAX_BUFFERS);
	assert(iovcnt > 0);

	req->iovcnt = iovcnt;

	assert(req->iov == iov);

	/* backward compatible */
	req->data = req->iov[0].iov_base;

	req->zcopy_bdev_io = bdev_io; /* Preserve the bdev_io for the end zcopy */

	spdk_nvmf_request_complete(req);
	/* Don't free the bdev_io here as it is needed for the END ZCOPY */
}

int
nvmf_bdev_ctrlr_start_zcopy(struct spdk_bdev *bdev,
			    struct spdk_bdev_desc *desc,
			    struct spdk_io_channel *ch,
			    struct spdk_nvmf_request *req)
{
	uint64_t bdev_num_blocks = spdk_bdev_get_num_blocks(bdev);
	uint32_t block_size = spdk_bdev_get_block_size(bdev);
	uint64_t start_lba;
	uint64_t num_blocks;

	nvmf_bdev_ctrlr_get_rw_params(&req->cmd->nvme_cmd, &start_lba, &num_blocks);

	if (spdk_unlikely(!nvmf_bdev_ctrlr_lba_in_range(bdev_num_blocks, start_lba, num_blocks))) {
		SPDK_ERRLOG("end of media\n");
		return -ENXIO;
	}

	if (spdk_unlikely(num_blocks * block_size > req->length)) {
		SPDK_ERRLOG("Read NLB %" PRIu64 " * block size %" PRIu32 " > SGL length %" PRIu32 "\n",
			    num_blocks, block_size, req->length);
		return -ENXIO;
	}

	bool populate = (req->cmd->nvme_cmd.opc == SPDK_NVME_OPC_READ) ? true : false;

	return spdk_bdev_zcopy_start(desc, ch, req->iov, req->iovcnt, start_lba,
				     num_blocks, populate, nvmf_bdev_ctrlr_start_zcopy_complete, req);
}

static void
nvmf_bdev_ctrlr_end_zcopy_complete(struct spdk_bdev_io *bdev_io, bool success,
				   void *cb_arg)
{
	struct spdk_nvmf_request	*req = cb_arg;

	if (spdk_unlikely(!success)) {
		int                     sc = 0, sct = 0;
		uint32_t                cdw0 = 0;
		struct spdk_nvme_cpl    *response = &req->rsp->nvme_cpl;
		spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc);

		response->cdw0 = cdw0;
		response->status.sc = sc;
		response->status.sct = sct;
	}

	spdk_bdev_free_io(bdev_io);
	req->zcopy_bdev_io = NULL;
	spdk_nvmf_request_complete(req);
}

int
nvmf_bdev_ctrlr_end_zcopy(struct spdk_nvmf_request *req, bool commit)
{
	return spdk_bdev_zcopy_end(req->zcopy_bdev_io, commit, nvmf_bdev_ctrlr_end_zcopy_complete, req);
}
