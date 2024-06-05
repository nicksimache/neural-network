package main

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	layers  int
	sizes   []int
	biases  []*mat.Dense
	weights []*mat.Dense
}

type Tuple[U any, V any] struct {
	first  U
	second V
}

func newNetwork(sizes []int) *NeuralNetwork {

	randSource := rand.NewSource(time.Now().UnixNano())
	random := rand.New(randSource)

	weights := make([]*mat.Dense, len(sizes)-1)
	biases := make([]*mat.Dense, len(sizes)-1)

	for i := 0; i < len(sizes)-1; i++ {

		//weights
		rowsW := sizes[i+1]
		colsW := sizes[i]
		dataW := make([]float64, rowsW*colsW)
		for j := range dataW {
			dataW[j] = random.Float64()
		}

		//biases
		rowsB := sizes[i+1]
		dataB := make([]float64, rowsB)
		for j := range dataB {
			dataB[j] = rand.NormFloat64()
		}

		weights[i] = mat.NewDense(rowsW, colsW, dataW)
		biases[i] = mat.NewDense(rowsB, 1, dataB)
	}

	return &NeuralNetwork{
		layers:  len(sizes),
		sizes:   sizes,
		weights: weights,
		biases:  biases,
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func (nn *NeuralNetwork) Feedforward(a *mat.Dense) *mat.Dense {

	for i := range nn.biases {

		z := mat.NewDense(nn.biases[i].RawMatrix().Rows, 1, nil)
		z.Product(nn.weights[i], a)

		z.Add(z, nn.biases[i])

		// applying sigmoid func
		rows, _ := z.Dims()
		data := make([]float64, rows)
		for i := range data {
			data[i] = sigmoid(z.At(i, 0))
		}

		a = mat.NewDense(rows, 1, data)

	}

	return a
}

func (nn *NeuralNetwork) SGD(trainingData []Tuple[*mat.Dense, *mat.Dense], epochs int, eta float64, miniBatchSize int) {
	n := len(trainingData)

	for j := 0; j < epochs; j++ {
		rand.Shuffle(len(trainingData), func(i, j int) {
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		})

		var miniBatches [][]Tuple[*mat.Dense, *mat.Dense]

		for k := 0; k < n; k += miniBatchSize {
			miniBatch := trainingData[k : k+miniBatchSize]
			miniBatches = append(miniBatches, miniBatch)

		}

		for _, miniBatch := range miniBatches {
			nn.updateMiniBatch(miniBatch, eta)
		}
	}

}

func (nn *NeuralNetwork) updateMiniBatch(miniBatch []Tuple[*mat.Dense, *mat.Dense], eta float64) {
	nablaB := make([]*mat.Dense, len(nn.biases))
	for i, b := range nn.biases {
		r, c := b.Dims()
		nablaB[i] = mat.NewDense(r, c, nil)
	}

	nablaW := make([]*mat.Dense, len(nn.weights))
	for i, w := range nn.weights {
		r, c := w.Dims()
		nablaW[i] = mat.NewDense(r, c, nil)
	}

	for _, tup := range miniBatch {
		x := tup.first
		y := tup.second

		tuple := nn.backprop(x, y)
		deltaNablaB := tuple.first
		deltaNablaW := tuple.second

		for i := range nablaB {
			nablaB[i].Add(nablaB[i], deltaNablaB[i])
		}

		for i := range nablaW {
			nablaW[i].Add(nablaW[i], deltaNablaW[i])
		}

		for i := range nn.weights {
			scaledMatrix := mat.NewDense(nablaW[i].RawMatrix().Rows, nablaW[i].RawMatrix().Cols, nil)
			scaledMatrix.Scale((eta / float64(len(miniBatch))), nablaW[i])
			nn.weights[i].Sub(nn.weights[i], scaledMatrix)
		}

		for i := range nn.biases {
			scaledMatrix := mat.NewDense(nablaB[i].RawMatrix().Rows, nablaB[i].RawMatrix().Cols, nil)
			scaledMatrix.Scale((eta / float64(len(miniBatch))), nablaB[i])
			nn.biases[i].Sub(nn.biases[i], scaledMatrix)
		}

	}

}

func (nn *NeuralNetwork) backprop(x, y *mat.Dense) Tuple[[]*mat.Dense, []*mat.Dense] {

	nablaB := make([]*mat.Dense, len(nn.biases))
	nablaW := make([]*mat.Dense, len(nn.weights))
	for i, b := range nn.biases {
		r, c := b.Dims()
		nablaB[i] = mat.NewDense(r, c, nil)
	}
	for i, w := range nn.weights {
		r, c := w.Dims()
		nablaW[i] = mat.NewDense(r, c, nil)
	}

	activation := x
	activations := []*mat.Dense{x}
	zs := []*mat.Dense{}

	for i := range nn.biases {
		z := mat.NewDense(0, 0, nil)
		z.Mul(nn.weights[i], activation)
		z.Add(z, nn.biases[i])
		zs = append(zs, z)

		activation = mat.NewDense(z.RawMatrix().Rows, z.RawMatrix().Cols, nil)

		r, c := z.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				activation.Set(i, j, sigmoid(z.At(i, j)))
			}
		}
		activations = append(activations, activation)
	}

	delta := mat.NewDense(0, 0, nil)

	sigmoidPrimeMat := mat.NewDense(zs[len(zs)-1].RawMatrix().Rows, zs[len(zs)-1].RawMatrix().Cols, nil)

	r, c := zs[len(zs)-1].Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sigmoidPrimeMat.Set(i, j, sigmoidPrime(zs[len(zs)-1].At(i, j)))
		}
	}

	delta.MulElem(nn.costDerivative(activations[len(activations)-1], y), sigmoidPrimeMat)
	nablaB[len(nablaB)-1] = delta

	delta_w := mat.NewDense(0, 0, nil)
	delta_w.Mul(delta, activations[len(activations)-2].T())
	nablaW[len(nablaW)-1] = delta_w

	for l := 2; l < nn.layers; l++ {
		z := zs[len(zs)-l]

		sp := mat.NewDense(z.RawMatrix().Rows, z.RawMatrix().Cols, nil)

		r, c := z.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				sp.Set(i, j, sigmoidPrime(z.At(i, j)))
			}
		}

		delta_next := mat.NewDense(0, 0, nil)
		delta_next.Mul(nn.weights[len(nn.weights)-l+1].T(), delta)
		delta_next.MulElem(delta_next, sp)
		delta = delta_next

		nablaB[len(nablaB)-l] = delta

		delta_w.Mul(delta, activations[len(activations)-l-1].T())
		nablaW[len(nablaW)-l] = delta_w
	}

	return Tuple[[]*mat.Dense, []*mat.Dense]{first: nablaB, second: nablaW}
}

func (nn *NeuralNetwork) costDerivative(outputActivations, y *mat.Dense) *mat.Dense {

	r, c := outputActivations.Dims()
	result := mat.NewDense(r, c, nil)

	result.Sub(outputActivations, y)

	return result
}
