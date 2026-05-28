from PIL import Image
from my_scorers import build_scorers

scorers = build_scorers("cuda:0")
img = Image.new("RGB", (224, 224), color="white")

for name, scorer in scorers.items():
    s = scorer.score("a white image", [img])
    print(name, s)