/*
 *	Copyright 2025 Rener Castro
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package mnist

// This file implements the baseline CNN model, including the FNN layers on top.

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
)

// LinearModelGraph builds a simple  model logistic model
// It returns the logit, not the predictions, which works with most losses with shape `[batch_size, NumClasses]`.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
func LinearModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	ctx = ctx.In("model") // Create the model by default under the "/model" scope.
	batchSize := inputs[0].Shape().Dimensions[0]
	embeddings := Reshape(inputs[0], batchSize, -1)
	logits := layers.DenseWithBias(ctx, embeddings, NumClasses)
	return []*Node{logits}
}

// CnnModelGraph builds the CNN model for our demo.
// It returns the logit, not the predictions, which works with most losses with shape `[batch_size, NumClasses]`.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
func CnnModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	ctx = ctx.In("model") // Create the model by default under the "/model" scope.
	embeddings := CnnEmbeddings(ctx, inputs[0])
	logits := layers.Dense(ctx, embeddings, true, NumClasses)
	return []*Node{logits}
}

func CnnEmbeddings(ctx *context.Context, images *Node) *Node {
	batchSize := images.Shape().Dimensions[0]

	layerIdx := 0
	nextCtx := func(name string) *context.Context {
		newCtx := ctx.Inf("%03d_%s", layerIdx, name)
		layerIdx++
		return newCtx
	}

	images = layers.Convolution(nextCtx("conv"), images).Filters(32).KernelSize(3).PadSame().Done()
	images.AssertDims(batchSize, 28, 28, 32)
	images = activations.Relu(images)
	images = layers.NormalizeFromContext(nextCtx("norm"), images)
	images = MaxPool(images).Window(2).Done()
	images = layers.DropoutFromContext(nextCtx("dropout"), images)
	images.AssertDims(batchSize, 14, 14, 32)

	images = layers.Convolution(nextCtx("conv"), images).Filters(64).KernelSize(3).PadSame().Done()
	images.AssertDims(batchSize, 14, 14, 64)
	images = activations.Relu(images)
	images = layers.NormalizeFromContext(nextCtx("norm"), images)
	images = MaxPool(images).Window(2).Done()
	images = layers.DropoutFromContext(nextCtx("dropout"), images)
	images.AssertDims(batchSize, 7, 7, 64)

	// Flatten images
	images = Reshape(images, batchSize, -1)
	return images
}
