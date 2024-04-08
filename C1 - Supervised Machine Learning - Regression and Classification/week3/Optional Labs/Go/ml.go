package main

import "math"

func sigmoid(z float64) float64 {
	g := 1 / (1 + math.Exp(-z))

	return g
}
