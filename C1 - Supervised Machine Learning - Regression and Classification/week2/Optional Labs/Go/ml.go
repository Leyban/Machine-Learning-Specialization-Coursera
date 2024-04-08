package main

import (
	"fmt"
	"math"
)

type LossFunc func(x [][]float64, y, w []float64, b float64) float64

func dotProduct(a, b []float64) float64 {
	var r float64
	for i := range a {
		r = r + a[i]*b[i]
	}
	return r
}

/*
		f_wb(x) for multiple features x

	    f_wb(x) = (w1x1 + w2x2 +...+wnxn) + b
*/
func predictSingleLoop(x, w []float64, b float64) float64 {
	// prediction
	var p float64

	p = dotProduct(w, x)
	p += b

	return p
}

/*
Loss Function
*/
func computeLoss(x [][]float64, y, w []float64, b float64) float64 {
	m := len(x)

	loss := 0.0

	for i := range x {
		f_wb_i := dotProduct(x[i], w) + b
		loss += math.Pow((f_wb_i - y[i]), 2)
	}
	loss = loss / (2 * float64(m))

	return loss
}

type GradientFunction func(x [][]float64, y, w []float64, b float64) ([]float64, float64)

/*
Gradient Descent for multiple inputs
*/
func computeGradient(x [][]float64, y, w []float64, b float64) ([]float64, float64) {
	m, n := len(x), len(x[0])

	dj_dw := make([]float64, n)
	dj_db := 0.

	for i := range x {
		diff := dotProduct(x[i], w) + b - y[i]

		for j := range x[0] {
			dj_dw[j] += diff * x[i][j]
		}
		dj_db += diff
	}

	for j := range dj_dw {
		dj_dw[j] = dj_dw[j] / float64(m)
	}
	dj_db = dj_db / float64(m)

	return dj_dw, dj_db
}

func batchGradientDescent(
	X [][]float64,
	y, w_init []float64,
	b_init float64,
	loss_function LossFunc,
	gradient_function GradientFunction,
	alpha float64,
	num_iters int64,
) (w []float64, b float64, J_history []float64) {

	w = w_init
	b = b_init

	for i := range num_iters {

		dj_dw, dj_db := gradient_function(X, y, w, b)

		for j := range w {
			w[j] = w[j] - (alpha * dj_dw[j])
		}
		b = b - (alpha * dj_db)

		J_history = append(J_history, loss_function(X, y, w, b))

		if i%int64(math.Ceil(float64(num_iters)/10)) == 0 {
			fmt.Println("Iteration: ", i, "Loss: ", J_history[len(J_history)-1])
		}
	}

	return w, b, J_history
}

func zScoreNormalization(X [][]float64) [][]float64 {
	mean := make([]float64, len(X[0]))
	std := make([]float64, len(X[0]))
	cols := make([][]float64, len(X[0]))

	for i := range X {
		for j := range X[0] {
			cols[j] = append(cols[j], X[i][j])
		}
	}

	for j := range cols {
		mean[j] = getMean(cols[j])
		std[j] = getStd(cols[j])
	}

	for i := range X {
		for j := range X[0] {
			X[i][j] = (X[i][j] - mean[j]) / std[j]
		}
	}

	return X
}
