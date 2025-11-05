from nn import (
  Generator
)

import torch

def test_dimensions(model: torch.nn.Module, latent_dim: int=100) -> bool:
  input = torch.randn(1, latent_dim)
  model.eval()
  try:
    model(input)
  except:
    return False
  return True

def main() -> bool:
  test_cases = {
    "DIMENSION ALIGNMENT": test_dimensions
  }
  img_size = (1, 1, 28, 28)
  latent_dim = 100
  generators = {
    "[128, 256, 512]": Generator(img_size=img_size, in_features=[128, 256, 512], latent_dim=latent_dim)
  }
  results = []
  print("Testing Generator implementation...")
  print("DETAILED REPORT")
  print("---------------")
  for test_case_name, test_case_func in test_cases.items():
    print(f"Problem: {test_case_name}")
    print("Feedback")
    for in_features, model in generators.items():
      test_case_status = test_case_func(model, latent_dim=latent_dim)
      if test_case_status:
        print(f"Dimensions of Generator(in_features={in_features}) are aligned")
      else:
        print(f"\033[91mDimensions of Generator(in_features={in_features}) are misaligned\033[0m")
      results.append(test_case_status)
  if all(results):
    print("All Generator implementation are dimension aligned!")
  return all(results)

if __name__ == "__main__":
  main()
