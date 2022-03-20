_base_ = './dw_r50_fpn_1x_coco.py'
model = dict(
    bbox_head=dict(
        reg_refine=False))
