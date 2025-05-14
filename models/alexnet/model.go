/*
Package AlexNet provides a structure for the AlexNet model

Reference:
- ImageNet Classification with Deep Convolutional Neural Networks, https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf (NIPS 2012)
*/
package alexnet

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/types/tensors/images"
)

const (
	BuildScope = "AlexNet"

	//
	NumClasses = 10

	// - The image size accepted by AlexNet is 227x227.
	ModelImageShortSize = 227

	ChannelAxisConfig = images.ChannelsLast
)

func AlexNetModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec                 // Not needed.
	ctx = ctx.In(BuildScope) // Create the model by default under the "/model" scope.
	image := inputs[0]

	embeddings := AlexNetEmbeddings(ctx, image)
	logits := fnn.New(ctx.In("Output"), embeddings, NumClasses).Done()

	return []*Node{logits}
}

func AlexNetEmbeddings(ctx *context.Context, image *Node) *Node {
	batchSize := image.Shape().Dimensions[0]
	image = PreprocessImage(image, ModelImageShortSize, ChannelAxisConfig)

	// Build model:

	// Layer 1: Conv2D + MaxPool2D
	// Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')
	image = layers.Convolution(ctx.In("layer1"), image).Filters(96).KernelSize(11).Strides(4).Done()
	image = activations.Relu(image)
	image.AssertDims(batchSize, 55, 55, 96)
	// MaxPool2D(pool_size=3, strides=2)
	image = MaxPool(image).Window(3).Strides(2).Done()
	image.AssertDims(batchSize, 27, 27, 96)

	// Layer 2: Conv2D + MaxPool2D
	// Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')
	image = layers.Convolution(ctx.In("layer2"), image).Filters(256).KernelSize(5).PadSame().Done()
	image = activations.Relu(image)
	image.AssertDims(batchSize, 27, 27, 256)
	// MaxPool2D(pool_size=3, strides=2)
	image = MaxPool(image).Window(3).Strides(2).Done()
	image.AssertDims(batchSize, 13, 13, 256)

	// Layer 3: Conv2D
	// Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
	image = layers.Convolution(ctx.In("layer3"), image).Filters(384).KernelSize(3).PadSame().Done()
	image = activations.Relu(image)
	image.AssertDims(batchSize, 13, 13, 384)

	// Layer 4: Conv2D
	// Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
	image = layers.Convolution(ctx.In("layer4"), image).Filters(384).KernelSize(3).PadSame().Done()
	image = activations.Relu(image)
	image.AssertDims(batchSize, 13, 13, 384)

	// Layer 5: Conv2D + MaxPool2D
	// Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')
	image = layers.Convolution(ctx.In("layer5"), image).Filters(256).KernelSize(3).PadSame().Done()
	image = activations.Relu(image)
	image.AssertDims(batchSize, 13, 13, 256)
	// MaxPool2D(pool_size=3, strides=2)
	image = MaxPool(image).Window(3).Strides(2).Done()
	image.AssertDims(batchSize, 6, 6, 256)

	// Flatten
	image = Reshape(image, batchSize, -1)
	image.AssertDims(batchSize, 4096, 1, 1)

	// Layer 6: Dense + Dropout
	// Dense(4096, activation='relu')
	image = layers.Dense(ctx.In("layer6"), image, true, 4096)
	image = activations.Relu(image)
	// Dropout(0.5)
	image = layers.DropoutStatic(ctx, image, 0.5)

	// Layer 7: Dense + Dropout
	// Dense(4096, activation='relu')
	image = layers.Dense(ctx.In("layer7"), image, true, 4096)
	image = activations.Relu(image)
	// Dropout(0.5)
	image = layers.DropoutStatic(ctx, image, 0.5)

	return image
}
