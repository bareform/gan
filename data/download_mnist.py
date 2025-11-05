import os

import torchvision

def main():
  mnist_destination = os.path.join("data", "datasets")
  if not (os.path.exists(mnist_destination) and os.path.isdir(mnist_destination)):
    os.makedirs(mnist_destination, exist_ok=True)
  torchvision.datasets.MNIST(root=mnist_destination, train=True, download=True)
  print("Finished downloading MNIST dataset!")

if __name__ == "__main__":
  main()
