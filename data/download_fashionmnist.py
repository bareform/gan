import os

import torchvision

def main():
  fashionmnist_destination = os.path.join("data", "datasets")
  if not (os.path.exists(fashionmnist_destination) and os.path.isdir(fashionmnist_destination)):
    os.makedirs(fashionmnist_destination, exist_ok=True)
  torchvision.datasets.FashionMNIST(root=fashionmnist_destination, train=True, download=True)
  print("Finished downloading FashionMNIST dataset!")

if __name__ == "__main__":
  main()
