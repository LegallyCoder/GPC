# GPC (Generative Pre-trained Coder based GPT-2)
![indir (1)](https://i.hizliresim.com/hdcugjz.png)

GPC is an open-source and freely available tool, meaning that developers can access its code and use it without any cost. This can be particularly beneficial for those who are working on a tight budget or who may not have access to other expensive coding assistance tools.

While GPC has many advantages, it is important to note that it is not a silver bullet solution for coding. While it can offer helpful suggestions and make coding faster, it is still up to the programmer to understand and verify the code they are writing. Additionally, while the pre-training on 150,000 Python code files is a valuable feature, it does not necessarily guarantee accurate suggestions for every coding scenario.

Overall, GPC is a valuable tool for developers looking to improve their coding workflow, particularly those working with Python. Its open-source and freely available nature make it accessible to a wide range of developers, while its potential for customization and improvement make it a flexible and adaptable tool for specific coding tasks.
# Setup
You can download the pre-trained GPC-1 version using the link below and use it in your projects.
Provider | URL
--- | --- |
Google Drive | [Google Drive](https://drive.google.com/drive/folders/1l2rpWAgTldKkjKPFqws5LRwzqMxG8a_w?usp=sharing) |

# Using

After downloading the model, you can use the code below.
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpc-1')
def predict(n):
  generated_text = model.generate(
      input_ids=torch.tensor(tokenizer.encode(n)).unsqueeze(0),
      max_length=50,
      num_beams=5,
      no_repeat_ngram_size=2,
      early_stopping=True,
      
  )
  generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
  return generated_text
predict("")#Enter what you want predicted here.

```
# Sources
https://www.kaggle.com/datasets/pranithchowdary/python-150k-code
https://github.com/karndeepsingh/Custom-Next-Sentence-Prediction
