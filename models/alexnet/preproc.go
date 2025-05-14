package alexnet

import (
	"math"
	"slices"

	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gopjrt/dtypes"
)

// PreprocessImage makes the image in a format usable to AlexNet model.
//
// It performs 3 tasks:
//   - It scales the image values from 0.0 to 1.0 for int dtypes.
//     For float dtypes, it assumes the values are already in that range.
//   - It removes the alpha channel, in case it is provided.
//   - Resize image according to square dimensions, while preserving the aspect ratio.
//
// Input `image` must have a batch dimension (rank=4), be either 3 or 4 channels, and its
func PreprocessImage(image *Node, shortSize int, channelsConfig images.ChannelsAxisConfig) *Node {
	if image.Rank() != 4 {
		exceptions.Panicf("alexnet.PreprocessImage requires image to be rank-4, got rank-%d instead", image.Rank())
	}

	// Scale image values to 0.0 to 1.0.
	if image.DType().IsInt() {
		image = ConvertDType(image, dtypes.F32)
		image = MulScalar(image, 1.0/255.0)
	}

	// Remove alpha-channel, if given.
	shape := image.Shape()
	channelsAxis := images.GetChannelsAxis(image, channelsConfig)
	if shape.Dimensions[channelsAxis] == 4 {
		axesRanges := make([]SliceAxisSpec, image.Rank())
		for ii := range axesRanges {
			if ii == channelsAxis {
				axesRanges[ii] = AxisRange(0, 3)
			} else {
				axesRanges[ii] = AxisRange()
			}
		}
		image = Slice(image, axesRanges...)
	}

	// Scale the image so the shorter spatial dimension becomes shortSize, preserving aspect ratio.
	newShape := image.Shape().Clone()
	spatialDims := images.GetSpatialAxes(image, channelsConfig)
	shortAxis := spatialDims[0]
	for _, axis := range spatialDims[1:] {
		if newShape.Dimensions[axis] < newShape.Dimensions[shortAxis] {
			shortAxis = axis
		}
	}
	scale := float64(shortSize) / float64(newShape.Dimensions[shortAxis])
	for _, axis := range spatialDims {
		if axis == shortAxis {
			newShape.Dimensions[axis] = shortSize
		} else {
			newShape.Dimensions[axis] = int(math.Round(float64(newShape.Dimensions[axis]) * scale))
		}
	}

	image = Interpolate(image, newShape.Dimensions...).Done()

	// Crop the image to a square of shortSize x shortSize.
	cropAxesRanges := make([]SliceAxisSpec, image.Rank())
	for i := range image.Rank() {
		if slices.Contains(spatialDims, i) {
			currentDimSize := newShape.Dimensions[i]
			if currentDimSize > shortSize {
				offset := (currentDimSize - shortSize) / 2
				cropAxesRanges[i] = AxisRange(offset, offset+shortSize)
			} else {
				// If dimension is already shortSize (or smaller, though scaling should prevent this)
				cropAxesRanges[i] = AxisRange(0, currentDimSize)
			}
		} else {
			// Keep the other dimensions as is.
			cropAxesRanges[i] = AxisRange()
		}
	}
	image = Slice(image, cropAxesRanges...)

	return image
}
