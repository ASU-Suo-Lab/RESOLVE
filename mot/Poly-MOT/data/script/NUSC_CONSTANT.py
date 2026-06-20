"""
Public constant accessible to all files
"""

import numpy as np

# metrics with two return values
METRIC = ['iou_3d', 'giou_3d']
FAST_METRIC = ['giou_3d', 'giou_bev']

# category name(str) <-> category label(int)
CLASS_STR_TO_LABEL = {
    'barrier': 0,
    'bicycle': 1,
    'bus': 2,
    'car': 3,
    'construction_vehicle': 4,
    'motorcycle': 5,
    'pedestrian': 6,
    'traffic_cone': 7,
    'trailer': 8,
    'truck': 9,
}

CLASS_LABEL_TO_STR = {v: k for k, v in CLASS_STR_TO_LABEL.items()}
# category name(str) <-> category label(int)
CLASS_SEG_TO_STR_CLASS = CLASS_STR_TO_LABEL      # 仍然 str->int
CLASS_STR_TO_SEG_CLASS = CLASS_LABEL_TO_STR      # 仍然 int->str


# math
PI, TWO_PI = np.pi, 2 * np.pi

# init EKFP for different non-linear motion model
CTRA_INIT_EFKP = {
    # [x, y, z, w, l, h, v, a, theta, omega]
    'bus': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'car': [4, 4, 4, 4, 4, 4, 1000, 4, 1, 0.1],
    'trailer': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'truck': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'pedestrian': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10]
}
CTRV_INIT_EFKP = {
    # [x, y, z, w, l, h, v, theta, omega]
    'bus': [10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'car': [4, 4, 4, 4, 4, 4, 1000, 1, 0.1],
    'trailer': [10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'truck': [10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'pedestrian': [10, 10, 10, 10, 10, 10, 10, 1000, 10]
}
BIC_INIT_EKFP = {
    # [x, y, z, w, l, h, v, a, theta, sigma]
    'bicycle': [10, 10, 10, 10, 10, 10, 10000, 10, 10, 10],
    'motorcycle': [4, 4, 4, 4, 4, 4, 100, 4, 4, 1],
}

def _scale_init(v, s):
    """scale a list/tuple init vector by factor s (elementwise)."""
    return [float(x) * float(s) for x in v]

# base templates
_car_ctra = CTRA_INIT_EFKP.get('car', [4,4,4,4,4,4,1000,4,1,0.1])
_truck_ctra = CTRA_INIT_EFKP.get('truck', [10,10,10,10,10,10,10,10,1000,10])

_car_ctrv = CTRV_INIT_EFKP.get('car', [4,4,4,4,4,4,1000,1,0.1])
_truck_ctrv = CTRV_INIT_EFKP.get('truck', [10,10,10,10,10,10,10,1000,10])

# 1) CTRA init补齐
CTRA_INIT_EFKP.setdefault('construction_vehicle', _truck_ctra)

# barrier = van : usually like car but a bit larger uncertainty on dims/vel
CTRA_INIT_EFKP.setdefault('barrier', _scale_init(_car_ctra, 1.5))

# traffic_cone = golf cart : slower, but turning more frequent; keep moderate uncertainty
# (这里给一个偏“稳定”的版本：速度/加速度相关项稍小一些)
_golf_ctra = list(_car_ctra)
# indices: [x,y,z,w,l,h,v,a,theta,omega]
_golf_ctra[6] = min(_golf_ctra[6], 500.0)   # v
_golf_ctra[7] = min(_golf_ctra[7], 2.0)     # a
_golf_ctra[9] = max(_golf_ctra[9], 0.2)     # omega (允许更灵活转向)
CTRA_INIT_EFKP.setdefault('traffic_cone', _golf_ctra)

# 2) CTRV init补齐（如果你某些类用 CTRV）
CTRV_INIT_EFKP.setdefault('construction_vehicle', _truck_ctrv)
CTRV_INIT_EFKP.setdefault('barrier', _scale_init(_car_ctrv, 1.5))

_golf_ctrv = list(_car_ctrv)
# indices: [x,y,z,w,l,h,v,theta,omega]
_golf_ctrv[6] = min(_golf_ctrv[6], 500.0)   # v
_golf_ctrv[8] = max(_golf_ctrv[8], 0.2)     # omega
CTRV_INIT_EFKP.setdefault('traffic_cone', _golf_ctrv)

# 3) BICYCLE model init表（通常只给 bicycle/motorcycle）
# 如果你绝不会把 barrier/traffic_cone 配成 BICYCLE，可以不加；
# 但为了“永不 KeyError”，给一个兜底（用 bicycle 的参数）
_bic_default = BIC_INIT_EKFP.get('bicycle', [10,10,10,10,10,10,10000,10,10,10])
BIC_INIT_EKFP.setdefault('barrier', _bic_default)
BIC_INIT_EKFP.setdefault('traffic_cone', _bic_default)
BIC_INIT_EKFP.setdefault('construction_vehicle', _bic_default)
BIC_INIT_EKFP.setdefault('trailer', _bic_default)
BIC_INIT_EKFP.setdefault('truck', _bic_default)
BIC_INIT_EKFP.setdefault('bus', _bic_default)
BIC_INIT_EKFP.setdefault('car', _bic_default)
BIC_INIT_EKFP.setdefault('pedestrian', _bic_default)