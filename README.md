# GPC (Generative Pre-trained Coder based GPT-2)
![indir (1)](https://i.hizliresim.com/hdcugjz.png)

GPC (Generative Pre-trained Coder based on GPT-2) is an open-source tool and its ability to be improved is a significant advantage.

Being open-source means that users have the ability to review and modify the code of the tool. This can lead to community-driven improvements and bug fixes that can benefit all users. In addition, open-source tools tend to be more transparent and trustworthy than closed-source tools.

Furthermore, GPC's ability to be improved is also a valuable feature. Developers can fine-tune the model for specific coding tasks or integrate it with other tools and platforms. This can result in even more accurate and useful suggestions for code completion, as well as new features and capabilities for the tool.

In summary, the fact that GPC is open-source and can be improved is a major advantage for developers looking for a powerful and flexible coding assistance tool.
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
