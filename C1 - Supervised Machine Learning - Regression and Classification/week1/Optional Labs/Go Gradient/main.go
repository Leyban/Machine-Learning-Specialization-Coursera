package main

import (
	"fmt"
	"math"
	"time"
)

type LossFunc func(x, y []float64, w, b float64) float64

// mean squared error loss
func compute_loss(x, y []float64, w, b float64) (total_loss float64) {
	m := len(x)

	var loss float64

	for i := range x {
		f_wb := (w * x[i]) + b
		loss += math.Pow((f_wb - y[i]), 2)
	}

	total_loss = loss / (2 * float64(m))

	return total_loss
}

type GradFunc func(x, y []float64, w, b float64) (float64, float64)

func compute_gradient(x, y []float64, w, b float64) (d_dw, d_db float64) {
	m := len(x)

	for i := range m {
		d_dw_i := (w*x[i] + b - y[i]) * x[i]
		d_db_i := (w*x[i] + b - y[i])
		d_dw += d_dw_i
		d_db += d_db_i
	}

	d_dw = d_dw / float64(m)
	d_db = d_db / float64(m)

	return d_dw, d_db
}

func gradient_descent(
	x, y []float64,
	w_init, b_init float64,
	alpha float64,
	num_iters int64,
	loss_function LossFunc,
	grad_function GradFunc,
) (w, b float64, j_history []float64, p_history [][]float64) {
	w = w_init
	b = b_init

	loss := loss_function(x, y, w, b)

	for i := range num_iters {
		d_dw, d_db := grad_function(x, y, w, b)

		w = w - (alpha * d_dw)
		b = b - (alpha * d_db)

		loss = loss_function(x, y, w, b)

		j_history = append(j_history, loss)
		p_history = append(p_history, []float64{w, b})

		if i%int64(math.Ceil(float64(num_iters)/10)) == 0 {
			fmt.Printf("Iteration: %v Loss: %v d_dw: %v d_db: %v w: %v b: %v \n", i, loss, d_dw, d_db, w, b)
		}
	}
	return w, b, j_history, p_history
}

func main() {

	x_train := []float64{1, 2}
	y_train := []float64{300, 500}

	var (
		w_init     float64
		b_init     float64
		iterations int64 = 10000
	)

	alpha := 0.01

	start := time.Now()
	w_final, b_final, _, _ := gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations, compute_loss, compute_gradient)
	fmt.Println(time.Since(start))

	fmt.Printf("final w: %v b: %v \n", w_final, b_final)
}
