package optimizers

import (
	"github.com/user/go-cnn/matrix"
)

type SGD struct {
	learningRate float64
}

func NewSGD(learningRate float64) *SGD {
	return &SGD{
		learningRate: learningRate,
	}
}

func (opt *SGD) Update(param *matrix.Matrix, grad *matrix.Matrix) {
	scaledGrad := grad.Scale(opt.learningRate)
	param.SubInPlace(scaledGrad)
}

func (opt *SGD) SetLearningRate(lr float64) {
	opt.learningRate = lr
}

func (opt *SGD) GetLearningRate() float64 {
	return opt.learningRate
}
