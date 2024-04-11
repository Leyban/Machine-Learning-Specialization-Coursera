package main

import (
	"fmt"
	"math"
)

func dotProduct(a, b []float64) float64 {
	var r float64
	for i := range a {
		r = r + a[i]*b[i]
	}
	return r
}

func sigmoid(z float64) float64 {
	g := 1 / (1 + math.Exp(-z))

	return g
}

func compute_loss_logistic(X [][]float64, y, w []float64, b float64) (loss float64) {
	m := len(X)
	for i := range X {
		z_i := dotProduct(X[i], w) + b
		f_wb_i := 1 / (1 + math.Exp(-z_i))
		loss += (y[i] * math.Log(f_wb_i)) + ((1 - y[i]) * math.Log(1-f_wb_i))
	}

	loss = -loss / float64(m)

	return loss
}

func compute_loss_logistic_reg(X [][]float64, y, w []float64, b, lambda float64) (loss float64) {
	m := len(X)
	for i := range X {
		z_i := dotProduct(X[i], w) + b
		f_wb_i := 1 / (1 + math.Exp(-z_i))
		loss += (y[i] * math.Log(f_wb_i)) + ((1 - y[i]) * math.Log(1-f_wb_i))
	}

	loss = -loss / float64(m)

	// regularization bit
	reg_loss := 0.
	for j := range w {
		reg_loss += math.Pow(w[j], 2)
	}
	reg_loss = (lambda / (2 * float64(m))) * reg_loss

	loss += reg_loss
	return loss
}

func compute_gradient_logistic(X [][]float64, y, w []float64, b float64) (dj_dw []float64, dj_db float64) {
	m := float64(len(X))

	dj_dw = make([]float64, len(X[0]))

	for i := range X {
		z_i := dotProduct(X[i], w) + b
		f_wb_i := sigmoid(z_i)

		for j := range X[i] {
			dj_dw[j] += (f_wb_i - y[i]) * X[i][j]
		}
		dj_db += f_wb_i - y[i]
	}

	for j := range dj_dw {
		dj_dw[j] = dj_dw[j] / m
	}
	dj_db = dj_db / m

	return dj_dw, dj_db
}

func compute_gradient_logistic_reg(X [][]float64, y, w []float64, b, lambda float64) (dj_dw []float64, dj_db float64) {
	m := float64(len(X))

	dj_dw = make([]float64, len(X[0]))

	for i := range X {
		z_i := dotProduct(X[i], w) + b
		f_wb_i := sigmoid(z_i)

		for j := range X[i] {
			dj_dw[j] += (f_wb_i - y[i]) * X[i][j]
		}
		dj_db += f_wb_i - y[i]
	}

	for j := range dj_dw {
		dj_dw[j] = dj_dw[j] / m
	}
	dj_db = dj_db / m

	// regularization bit
	for j := range dj_dw {
		dj_dw[j] += (lambda / m) * w[j]
	}

	return dj_dw, dj_db
}

func gradient_descent(
	X [][]float64,
	y, w_init []float64,
	b_init float64,
	alpha float64,
	iterations int64,
) (
	w []float64,
	b float64,
	J_history []float64,
) {

	w = w_init
	b = b_init

	for i := range iterations {
		dj_dw, dj_db := compute_gradient_logistic(X, y, w, b)

		for j := range w {
			w[j] = w[j] - (alpha * dj_dw[j])
		}
		b = b - (alpha * dj_db)

		J_history = append(J_history, compute_loss_logistic(X, y, w, b))

		if i%int64(math.Ceil(float64(iterations/10))) == 0 {
			fmt.Println("Iteration: ", i, "Cost: ", J_history[len(J_history)-1])
		}
	}

	return w, b, J_history
}
