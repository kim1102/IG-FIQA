# IG-FIQA: Improving Face Image Quality Assessment through Intra-class Variance Guidance robust to Inaccurate Pseudo-Labels

>## Abstract
In the realm of face image quality assesment (FIQA), method based on sample relative classification have shown impressive performance. However, the quality scores used as pseudo-labels assigned from images of classes with low intra-class variance could be unrelated to the actual quality in this method. To address this issue, we present IG-FIQA, a novel approach to guide FIQA training, introducing a weight parameter to alleviate the adverse impact of these classes. This method involves estimating sample intra-class variance at each iteration during training, ensuring minimal computational overhead and straightforward implementation. Furthermore, this paper proposes an on-the-fly data augmentation methodology for improved generalization performance in FIQA. On various benchmark datasets, our proposed method, IG-FIQA, achieved novel state-of-the-art (SOTA) performance.
<img src="assets/main_figure.png"/>

>## Pseudo-code (Pytorch)

```python
import torch
from backbone import iresnet50
from losses import ArcFace, CR_FIQA

# Loss definition
CELoss = torch.nn.CrossEntropyLoss()

# intra-class variance weights for each classes
label_weight = torch.ones(trainset.class_num)

# label weight update
for epoch in range(0, end_epoch):
  loss_momentum = 0.9 + (0.1 * epoch / end_epoch)
  for learning_iter, (img, label) in enumerte(train_loader):
    # Loss calculation for Backbone update
    FR_loss = ArcFace(img, label)
    augmented_img  = augment(img)
  
    # Loss calculation for Regression layer update
    CR_loss, ccs, nnccs = CR_FIQA(augmented_img, label)
    with torch.no_grad():
      label_weight[label] = (loss_momentum * label_weight[label]) + ((1 - loss_momentum) * ccs)
      normalized_label_weight = (label_weight[label] - torch.mean(label_weight)) / (torch.std(label_weight) + 1e-6)
      normalized_weight = 1 + torch.clip(normalized_label_weight, -1.0, 0.0)

    IG_loss = torch.mean(CR_loss * normalized_weight)
    total_loss = FR_loss + (alpha*IG_loss) # alpha = 10.0
    total_loss.backward()
```


>## Reference

The code for backbone network and loss can be found in the official Insightface repository and CR-FIQA.
https://github.com/deepinsight/insightface
https://github.com/fdbtrs/CR-FIQA
