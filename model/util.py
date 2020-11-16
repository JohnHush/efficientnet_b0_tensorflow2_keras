import tensorflow as tf

IMAGE_SIZE = 224
CROP_PADDING = 32

def _resize_image(image, image_size, method=None):
  # if method is not None:
  #   tf.logging.info('Use customized resize method {}'.format(method))
  #   return tf.image.resize([image], [image_size, image_size], method)[0]
  # tf.logging.info('Use default resize_bicubic.')
    return tf.image.resize([image], [image_size, image_size])[0]

def preprocess_for_eval(image_bytes,
                        use_bfloat16=False,
                        image_size=IMAGE_SIZE,
                        resize_method=None):
    """Preprocesses the given image for evaluation.

    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      use_bfloat16: `bool` for whether to use bfloat16.
      image_size: image size.
      resize_method: if None, use bicubic.

    Returns:
      A preprocessed image `Tensor`.
    """
    image = _decode_and_center_crop(image_bytes, image_size, resize_method)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.image.convert_image_dtype(
        image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
    return image

def _decode_and_center_crop(image_bytes, image_size, resize_method=None):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = _resize_image(image, image_size, resize_method)

    return image
