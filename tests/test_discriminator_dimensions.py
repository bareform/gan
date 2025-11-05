from nn import (
  Discriminator
)

import torch

def test_dimensions(model: torch.nn.Module, img_size: tuple[int, int, int]=(1, 28, 28)) -> bool:
  input = torch.randn(img_size)
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
  discriminators = {
    "[512, 256, 128]": Discriminator(img_size=img_size, in_features=[512, 256, 128], use_spectral_norm=False)
  }
  results = []
  print("Testing Discriminator implementation...")
  print("DETAILED REPORT")
  print("---------------")
  for test_case_name, test_case_func in test_cases.items():
    print(f"Problem: {test_case_name}")
    print("Feedback")
    for in_features, model in discriminators.items():
      test_case_status = test_case_func(model, img_size=img_size)
      if test_case_status:
        print(f"Dimensions of Discriminator(in_features={in_features}) are aligned")
      else:
        print(f"\033[91mDimensions of Discriminator(in_features={in_features}) are misaligned\033[0m")
      results.append(test_case_status)
  if all(results):
    print("All Discriminator implementation are dimension aligned!")
  return all(results)

if __name__ == "__main__":
  main()
