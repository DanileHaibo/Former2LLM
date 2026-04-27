_base_ = ["./unidrivevla_b2d_stage2_unified_2b.py"]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LeadVehicleGlare",
        cam_index=0,
        min_forward_y=2.0,
        max_forward_y=80.0,
        max_lateral_x=14.0,
        margin_px=18,
        sigma_scale=0.55,
        peak_bgr=(255.0, 255.0, 255.0),
        solid_blend=0.35,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
    dict(
        type="InstanceNameFilter",
        classes=["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"],
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "timestamp",
            "projection_mat",
            "image_wh",
            "ego_status",
            "gt_ego_fut_cmd",
            "gt_ego_fut_trajs_2hz",
            "gt_ego_fut_masks_2hz",
            "gt_attr_labels",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token"],
    ),
]

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

