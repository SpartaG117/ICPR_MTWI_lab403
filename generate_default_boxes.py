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
    feature_height = feature_shape[0]
    feature_width = feature_shape[1]
    feature_boxes = []
    print(feature_shape)
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

    return feature_boxes


def generate_default_boxes(config):
    image_shape = config.IMAGE_SHAPE
    assert image_shape[0] // (2 ** 8) == int(image_shape[0] / (2 ** 8)) and \
        image_shape[1] // (2 ** 8) == int(image_shape[1] / (2 ** 8))

    all_scales = config.ALL_SCALES
    assert len(all_scales) == 7

    default_boxes = []

    for i in range(2, 9):
        feature_stride = 2 ** i
        feature_height = image_shape[0] // feature_stride
        feature_width = image_shape[1] // feature_stride
        feature_shape = [feature_height, feature_width]
        feature_boxes = generate_feature_boxes(feature_shape, feature_stride, all_scales[i - 2])
        default_boxes.extend(feature_boxes)

    return default_boxes

