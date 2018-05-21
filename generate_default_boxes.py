def get_ctrs(y1, x1, y2, x2):
    y_ctr = y1 + 0.5 * (y2 - y1)
    x_ctr = x1 + 0.5 * (x2 - x1)
    return y_ctr, x_ctr


def generate_box(y_ctr, x_ctr, h, w):
    y1 = y_ctr - 0.5 * (h - 1)
    x1 = x_ctr - 0.5 * (w - 1)
    y2 = y1 + h - 1
    x2 = x1 + w - 1
    return y1, x1, y2, x2


def generate_feature_boxes(feature_shape, feature_stride, scales):
    """
    注意这里的生成顺序是先是scale,然后是列维度，最后是行维度
    """
    feature_height = feature_shape[0]
    feature_width = feature_shape[1]
    feature_boxes = []
    for i in range(feature_height):
        for j in range(feature_width):
            for s in scales:
                y1 = i * feature_stride
                x1 = j * feature_stride
                y2 = (i + 1) * feature_stride - 1
                x2 = (j + 1) * feature_stride - 1
                y_ctr, x_ctr = get_ctrs(y1, x1, y2, x2)
                box = generate_box(y_ctr, x_ctr, s, s)
                feature_boxes.append(box)
    print('Have generated {:d} default boxes on {:d}x{:d} feature map'.
          format(len(feature_boxes), feature_height, feature_width))
    return feature_boxes


def generate_default_boxes(config):

    image_shape = config.IMAGE_SHAPE
    all_scales = config.ALL_SCALES
    downsample_factors = config.DOWNSAMPLE_FACTORS
    assert len(all_scales) == len(downsample_factors)
    # 表明图像的尺寸必须是最大下采样的整数倍
    assert image_shape[0] % 2**downsample_factors[-1] == 0 and image_shape[1] % 2**downsample_factors[-1] == 0

    default_boxes = []

    print('Begin to generate default boxes from high resolution to low resolution')
    for i, factor in enumerate(downsample_factors):
        feature_stride = 2 ** factor
        feature_height = image_shape[0] // feature_stride
        feature_width = image_shape[1] // feature_stride
        feature_shape = [feature_height, feature_width]
        feature_boxes = generate_feature_boxes(feature_shape, feature_stride, all_scales[i])
        default_boxes.extend(feature_boxes)

    print('Have generated {:d} default boxes in total'.format(len(default_boxes)))
    # 返回的是列表
    return default_boxes
