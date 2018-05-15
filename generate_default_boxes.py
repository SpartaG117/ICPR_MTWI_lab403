def get_ctrs(x1, y1, x2, y2):
    x_ctr = x1 + 0.5 * (x2 - x1)
    y_ctr = y1 + 0.5 * (y2 - y1)
    return x_ctr, y_ctr


def make_box(w, h, x_ctr, y_ctr):
    x1 = x_ctr - 0.5 * (w - 1)
    y1 = y_ctr - 0.5 * (h - 1)
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    return x1, y1, x2, y2


def make_feature_boxes(feature_shape, feature_stride, scales):
    feature_height = feature_shape[0]
    feature_width = feature_shape[1]
    feature_boxes = []
    print(feature_shape)
    for i in range(feature_height):
        for j in range(feature_width):
            for s in scales:
                x1 = j * feature_stride
                y1 = i * feature_stride
                x2 = (j + 1) * feature_stride - 1
                y2 = (i + 1) * feature_stride - 1
                x_ctr, y_ctr = get_ctrs(x1, y1, x2, y2)
                box = make_box(s, s, x_ctr, y_ctr)
                feature_boxes.append(box)
                
    return feature_boxes


def make_default_boxes(config):
    image_shape = config.IMAGE_SHAPE
    assert image_shape[0] // (2**8) == int(image_shape[0] / (2**8)) and \
        image_shape[1] // (2**8) == int(image_shape[1] / (2**8))
    
    all_scales = config.ALL_SCALES
    assert len(all_scales) == 7
    
    default_boxes = []
    
    for i in range(2, 9):
        feature_stride = 2**i
        feature_shape = []
        feature_shape.append(image_shape[0] // feature_stride)
        feature_shape.append(image_shape[1] // feature_stride)
        feature_boxes = make_feature_boxes(feature_shape, feature_stride, all_scales[i-2])
        default_boxes.extend(feature_boxes)
    return default_boxes
