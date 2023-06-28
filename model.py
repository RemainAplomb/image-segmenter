import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers as L

from tensorflow.keras.utils import plot_model


def convBlock (priorNode, LAYER1= (96, (1,1), (1,1)), LAYER2=((3,3), (1,1)), LAYER3 = (64, (1,1), (1,1))):
  conv2d_1 = L.Conv2D(LAYER1[0], LAYER1[1], strides= LAYER1[2] , padding='same', use_bias=True, activation='relu')(priorNode)
  depth_conv2d_1 = L.DepthwiseConv2D(LAYER2[0], strides=LAYER2[1], padding='same', use_bias=True, activation='relu')(conv2d_1)
  conv2d_2 = L.Conv2D(LAYER3[0], LAYER3[1], strides=LAYER3[2], padding='same', use_bias=True)(depth_conv2d_1)
  return conv2d_2

def reshapeTransposeBlock(priorNode, RESHAPE=(8, 4, 8, 512), TRANSPOSE_PERM=[0, 2, 1, 3]):
    print("HEY")
    print(priorNode.shape)
    batch_size = tf.shape(priorNode)[0]
    reshape = tf.reshape(priorNode, RESHAPE)
    transpose = tf.transpose(reshape, perm=TRANSPOSE_PERM)
    print(transpose.shape)
    return transpose


def transposeReshapeBlock (priorNode, RESHAPE=(8, 4, 8, 512), TRANSPOSE_PERM=[0, 2, 1, 3]):
  transpose = tf.transpose(priorNode, perm=TRANSPOSE_PERM)
  batch_size = tf.shape(transpose)[0]
  reshape = tf.reshape(transpose, RESHAPE)
  return reshape

def largeBlock(priorNode, MUL=128):
  # Block 2-1
  block_2_1 = tf.multiply(priorNode, priorNode)  # Multiply with priorNode itself
  block_2_1 = tf.add(block_2_1, block_2_1)  # Add the result with itself

  # Block 2-2
  block_2_2 = L.Conv2D(1, (1, 1), strides=(1, 1), padding='same', use_bias=True)(block_2_1)
  
  batch_size = tf.shape(block_2_2)[0]
  if MUL == 128:
    block_2_2 = tf.reshape(block_2_2, (batch_size, 16, 1, 64))
  elif MUL == 192:
    block_2_2 = tf.reshape(block_2_2, (batch_size, 16, 1, 16))

  block_2_2 = L.Softmax(axis=-1)(block_2_2)

  batch_size = tf.shape(block_2_2)[0]
  if MUL == 128:
    block_2_2 = tf.reshape(block_2_2, (batch_size, 16, 64, 1))
  elif MUL == 192:
    block_2_2 = tf.reshape(block_2_2, (batch_size, 16, 16, 1))

  # Block 2-3
  block_2_3 = L.Conv2D(MUL, (1, 1), strides=(1, 1), padding='same', use_bias=True)(block_2_1)
  block_2_3 = L.Multiply()([block_2_2, block_2_3])
  block_2_3 = L.Lambda(lambda x: tf.reduce_sum(x, axis=2, keepdims=True))(block_2_3)

  # Block 2-4
  block_2_4 = L.Conv2D(MUL, (1, 1), strides=(1, 1), padding='same', use_bias=True, activation='relu')(block_2_1)
  block_2_4 = L.Multiply()([block_2_3, block_2_4])
  block_2_4 = L.Conv2D(MUL, (1, 1), strides=(1, 1), padding='same', use_bias=True)(block_2_4)
  block_2_4 = L.Add()([block_2_4, priorNode])  # <-- Corrected typo: block2 -> priorNode

  # Block 2-5
  block_2_5 = L.Conv2D(MUL*2, (1, 1), strides=(1, 1), padding='same', use_bias=True)(block_2_1)
  block_2_5 = L.Conv2D(MUL, (1, 1), strides=(1, 1), padding='same', use_bias=True)(block_2_5)

  # Output
  return L.Add()([block_2_5, block_2_4])


# BUILD MODEL
def build_model(input_shape, classes = 6):
  # Input layer
  inputs = L.Input(input_shape)
  x = L.Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=True, activation='relu')(inputs)

  # Block 1 (CONV)
  x1 = convBlock(x, LAYER1= (96, (1,1), (1,1)), LAYER2=((3,3), (1,1)), LAYER3 = (64, (1,1), (1,1)))
  x1 = convBlock(x1, LAYER1= (384, (1,1), (1,1)), LAYER2=((3,3), (2,2)), LAYER3 = (128, (1,1), (1,1)))

  x1_1 = convBlock(x1, LAYER1= (768, (1,1), (1,1)), LAYER2=((3,3), (1,1)), LAYER3 = (128, (1,1), (1,1)))
  output1 = L.Add(name= "output1")([x1_1, x1])
  batch_size = tf.shape(output1)[0]
  # output1 = tf.reshape(output1, (batch_size, 64, 64, 128))

  # Block 2
  x2 = convBlock(output1, LAYER1= (768, (1,1), (1,1)), LAYER2=((3,3), (2,2)), LAYER3 = (256, (1,1), (1,1)))
  x2 = L.Conv2D(128, (3,3), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x2)
  x2 = L.Conv2D(128, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x2)
  x2 = reshapeTransposeBlock(x2, RESHAPE= (8, 4, 8, 512), TRANSPOSE_PERM=[0, 2, 1, 3])
  output2 = reshapeTransposeBlock(x2, RESHAPE= (1, 64, 16, 128), TRANSPOSE_PERM=[0, 2, 1, 3])

  # Block 3
  x3 = largeBlock(output2)
  x3 = largeBlock(x3)

  x3 = transposeReshapeBlock(x3, RESHAPE= (8, 8, 4, 512), TRANSPOSE_PERM=[0, 2, 1, 3])
  x3 = transposeReshapeBlock(x3, RESHAPE= (1, 32, 32, 128), TRANSPOSE_PERM=[0, 2, 1, 3])
  output3 = L.Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=True, name="output3")(x3)

  # Block 4
  x4 = convBlock(output3, LAYER1= (1536, (1,1), (1,1)), LAYER2=((3,3), (2,2)), LAYER3 = (384, (1,1), (1,1)))
  x4 = L.Conv2D(192, (3,3), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x4)
  x4 = L.Conv2D(192, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x4)
  x4.shape

  x4 = reshapeTransposeBlock(x4, RESHAPE= (4, 4, 4, 768), TRANSPOSE_PERM=[0, 2, 1, 3])
  x4 = reshapeTransposeBlock(x4, RESHAPE= (1, 16, 16, 192), TRANSPOSE_PERM=[0, 2, 1, 3])
  # x4.shape

  x4 = largeBlock(x4, MUL=192)
  x4 = largeBlock(x4, MUL=192)
  x4 = largeBlock(x4, MUL=192)
  x4 = largeBlock(x4, MUL=192)

  x4 = transposeReshapeBlock(x4, RESHAPE= (4, 4, 4, 768), TRANSPOSE_PERM=[0, 2, 1, 3])
  x4 = transposeReshapeBlock(x4, RESHAPE= (1, 16, 16, 192), TRANSPOSE_PERM=[0, 2, 1, 3])
  x4 = L.Conv2D(384, (1,1), strides= (1,1) , padding='same', use_bias=True)(x4)
  x4 = L.DepthwiseConv2D((1,1), strides=(1,1), padding='same', use_bias=True)(x4)
  output4 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True, name= "output4")(x4)

  # Block 5
  x5 = L.DepthwiseConv2D((1,1), strides=(1,1), padding='same', use_bias=True)(output3)
  x5 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x5)

  x5_output4 = tf.image.resize(output4, size=(32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output5 = L.Add(name= "output5")([x5, x5_output4])

  # Block 6
  x6 = L.DepthwiseConv2D((1,1), strides=(1,1), padding='same', use_bias=True)(output1)
  x6 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x6)

  x6_output5 = tf.image.resize(output5, size=(64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output6 = L.Add(name= "output6")([x6, x6_output5])

  # Block 7 
  x7 = L.DepthwiseConv2D((1,1), strides=(1,1), padding='same', use_bias=True)(x)
  x7 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x7)

  x7_output6 = tf.image.resize(output6, size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output7 = L.Add(name= "output7")([x7, x7_output6])

  # Block 8 (No longer need resize nearest neighbor)
  x8 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output7)
  x8 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x8)

  x8_output6 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output6)
  x8_output6 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x8_output6)
  x8_output6 = tf.image.resize(x8_output6, size=(128, 128), method=tf.image.ResizeMethod.BILINEAR)
  output8 = L.Add(name= "output8")([x8, x8_output6])

  # Block 9
  x9 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output8)
  x9 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x9)

  x9_output5 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output5)
  x9_output5 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x9_output5)
  x9_output5 = tf.image.resize(x9_output5, size=(128, 128), method=tf.image.ResizeMethod.BILINEAR)
  output9 = L.Add(name= "output9")([x9, x9_output5])

  # Block 10
  x10 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output9)
  x10 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x10)

  x10_output4 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output4)
  x10_output4 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x10_output4)
  x10_output4 = tf.image.resize(x10_output4, size=(128, 128), method=tf.image.ResizeMethod.BILINEAR)
  output10 = L.Add(name= "output10")([x10, x10_output4])

  # Block 11 (Head)
  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(output10)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(x11)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(x11)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(x11)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(x11)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  x11 = L.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same', use_bias=True)(x11)

  output11 = L.Conv2D(NUM_CLASSES, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)


  # Build model
  return tf.keras.Model(inputs=inputs, outputs=output11)


if __name__ == "__main__":
  INPUT_SHAPE = (256, 256, 3)
  NUM_CLASSES = 6
  model = build_model(INPUT_SHAPE)
  model.summary()
  # Save model architecture plot
  # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
