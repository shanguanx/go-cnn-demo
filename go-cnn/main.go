package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]

	switch command {
	case "train":
		RunTraining()
	case "inference":
		RunInference()
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Printf("未知命令: %s\n\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("MNIST手写数字识别 CNN - Go实现")
	fmt.Println("================================")
	fmt.Println()
	fmt.Println("用法:")
	fmt.Println("  go run . <command>")
	fmt.Println()
	fmt.Println("可用命令:")
	fmt.Println("  train      训练MNIST数字识别模型")
	fmt.Println("  inference  使用训练好的模型进行推理")
	fmt.Println("  help       显示此帮助信息")
	fmt.Println()
	fmt.Println("示例:")
	fmt.Println("  go run . train      # 开始训练")
	fmt.Println("  go run . inference  # 开始推理")
}
