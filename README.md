<font face="Times New Roman" font color=orange size=8>YOLO Network Implementation Note</font>

# Target Calculations - Label Encoding

**Example Calculation - Targets**
![Example Calculation](./YOLOv1/example%20calculation1.png)
- Center point (x,y): Relative to anchor that (x,y) falls into

$$
(x,y,w,h) = (200,311,142,250)
$$
$$
\Delta x = (x-x_a)/64
$$
$$
\Delta y = (y-y_a)/64
$$
- Width/height (w,h): Relative to the whole image
$$
\Delta w = w/448
$$
$$
\Delta h = h/448
$$
![Example Calculation - Target](./YOLOv1/example%20calculation2.png)
- In this Example
$$
\Delta x = \frac{200 - 192}{64} \approx 0.13
$$
$$
\Delta y = \frac{311 - 256}{64} \approx 0.87
$$
$$
\Delta w = \frac{w}{448}
$$
$$
\Delta h = \frac{h}{448}
$$
$$
(200,311,142,250) \Rightarrow (0.13,0.87,0.31,0.56)
$$

### For YOLO output predictions (7 * 7 * 30 tensor)
- The model divides the image into an S*S grid
- For each grid cell predicts B bounding boxes (B = 2)
- For each Bounding box:
$$
(\Delta x_i, \Delta y_i, \Delta w_i, \Delta h_i, c_i)^B_{i=1}
$$
while c represents for the confidence of this bounding box
- Conditional class probabilities (C=20)
$$
(p_1, p_2, ..., p_{20})
$$
- We got (B * 5 + C) dimensionality for each grid cell
- And so these predictions are encoded as an S * S * (B * 5 + C) tensor

**For evaluating YOLO on PASCAL VOC, we use S = 7, B = 2. PASCAL VOC has 20 labelled classes so C = 20. So the final prediction is a 7 * 7 * 30 tensor.**
![YOLO prediction tensor](./YOLOv1/prediction%20tensor.png)
- **Example for prediction**

|grid cell|$(\Delta \hat{x}, \Delta \hat{y}, \Delta \hat{w}, \Delta \hat{h}, \hat c)_{B=1}$|$(\Delta \hat{x}, \Delta \hat{y}, \Delta \hat{w}, \Delta \hat{h}, \hat c)_{B=2}$|$(\hat p_1, \hat p_2, ..., \hat p_{20})$|
|:---:|:---:|:---:|:---:|
|$A_1$|(0 0 0 0 0)|(0 0 0 0 0)|(0 0 ... 0)|
|$A_2$|(0 0 0 0 0)|(0 0 0 0 0)|(0 0 ... 0)|
|...|...|...|...|
|$A_{32}$|(0.9 0.7 0.1 0.1 **1.0**)|(0.1 0.8 0.3 0.5 **1.0**)|(0 ... **1.0** ...)|
|...|...|...|**($\hat p_{14}$=person)**|
|$A_{49}$|(0 0 0 0 0)|(0 0 0 0 0)|(0 0 ... 0)|
# Model Architecture
- **Inspired by GoogleNet model**
- **Network:**
  - **24 convolution layers**
  - **2 fully connected layers**

  ![model architecture](./YOLOv1/model%20architecture.png)
- **Last 2 fully connected layers**
  - FLatten the last 7 * 7 * 1024 tensor to 50176 feature vector
  - 2 Linear (W1 & W2)
  - Reshape 1470 vector to 7 * 7 * 30 feature map

  ![output layer](./YOLOv1/output%20layer.png)
- Here we use a **Fully Convolutional** layer for implementation of the output head
  - network output (B, 7, 7, 30), no Linear
  - YOLOv2/YOLOv3 vanilla
  ```py
  class YOLO(nn.Module):
      def __init__(self, **kwargs):
          super().__init__()
          ### 24 Convolutional layers & Maxpooling 

          ### use Fully Convolution for regression head (no flatten or linear)
          self.head = nn.Conv2d(
              in_channels=1024,
              out_channels=30,
              kernel_size=1
              )
      def forward(self, x, **kwargs):
          x = self.conv(x)    # (B, 1024, 7, 7)
          x = self.head(x)    # (B, 30, 7, 7)

          ### adjust to YOLO format
          x = x.permute(0, 2, 3, 1)   # (B, 7, 7, 30)

          return x
  ```
# Training Process
- Pretrained on ImageNet-1k at 224 * 224
- Actual training on 448 * 448 on VOC
 ![training process](./YOLOv1/training%20process.png)
# Loss Function
$$
\lambda_{coord} \sum^{S^2}_{i=0}{\sum^B_{j=0}{\mathbf{1}^{obj}_{ij}[(x_i-\hat x_i)^2 + (y_i - \hat y_i)^2]}} + \lambda_{coord} \sum^{S^2}_{i=0}{\sum^B_{j=0}{\mathbf{1}^{obj}_{ij}[(\sqrt w_i - \sqrt{\hat w_i})^2 + (\sqrt h_i - \sqrt{\hat h_i})^2] + \sum^{S^2}_{i=0}{\sum^B_{j=0}{\mathbf{1}^{obj}_{ij}(C_i - \hat C_i)^2}}}} + \lambda_{noobj} \sum^{S^2}_{i=0}{\sum^B_{j=0}{\mathbf{1}^{noobj}_{ij}(C_i - \hat C_i)^2}} + \sum^{S^2}_{i=0}{\mathbf{1}^{obj}_i \sum_{c \in classes}{(p_i(c) - \hat p_i(c))^2}}
$$
where $\mathbf{1}^{obj}_i$ denotes if object appears in cell $i$ and $\mathbf{1}^{obj}_{ij}$ denotes that the $j$th bounding box preditor in cell $i$ is "responsible" for that prediction.

$\mathbf{1}^{obj}_i = 1$ if $i^{th}$ grid is **object anchor**, $\mathbf{1}^{noobj}_i = 1$ if $i^{th}$ grid is **no-object anchor**

In the original YOLO paper the parameters $\lambda_{coord}$ & $\lambda_{noobj}$ are set $\lambda_{coord} = 5$ and $\lambda_{noobj} = 0.5$
### Loss for object cells
  - Loss = Bounding Box Regression Loss + Objectness Confidence Loss + Classification Loss
$$
L_{i,obj} = \lambda_{coord} \times L^{box}_{i,obj} + L^{conf}_{i,obj} + L^{cls}_{i,obj}\ ,\ \lambda_{coord} = 5
$$

### Bounding Box Regression Loss
$$
L^{box}_{i,obj} = (\Delta x^*_i - \Delta{\hat x_i})^2 + (\Delta y^*_i - \Delta{\hat y_i})^2 + (\sqrt{\Delta w^*_i} - \sqrt{\Delta{\hat w_i}})^2 + (\sqrt{\Delta h^*_i} - \sqrt{\Delta{\hat h_i}})^2
$$
- $(\Delta{\hat x_i}, \Delta{\hat y_i},\Delta{\hat w_i},\Delta{\hat h_i}):$ ground truth box
- $(\Delta{x^*_i}, \Delta{y^*_i},\Delta{w^*_i},\Delta{h^*_i}):$ **responsible** predicted box that has the largest IoU with ground truth box
### Objectness Confidence Loss
$$
L^{conf}_{i,obj} = \sum^{S^2}_{i=1}{\mathbf{1}^{obj}_i \times L_{i,obj}} + \lambda_{noobj} \sum^{S^2}_{i=1}{\mathbf{1}^{noobj}_i \times L_{i,noobj}}
$$
while $L_{i,obj} = (c^*_i - \hat c_i)^2$
- Squared error between the predicted confidence and encoded label confidence

### Classification Loss
- Sum of squared errors over all class probabilities
$$
L^{cls}_{i,obj} = \sum^{20}_{c=1}{(p_{i,c} - \hat p_{i,c})^2}
$$
# Fast YOLO
![fast yolo](./YOLOv1/fast%20yolo.png)
# Performance
|Real-Time Detectors|Train|mAP|FPS
|:---|---:|:---:|:---:|
|100Hz DPM|2007|16.0|100|
|30Hz DPM|2007|26.1|30|
|Fast YOLO|2007+2012|52.7|**155**|
|YOLO|2007+2012|**63.4**|45|
|**Less Than Real-Time**| | | |
|Fastest DPM|2007|30.4|15|
|R-CNN Minus R|2007|53.5|6|
|Fast R-CNN|2007+2012|70.0|0.5|
|Faster R-CNN VGG-16|2007+2012|73.2|7|
|Faster R-CNN ZF|2007+2012|62.1|18|
|YOLO VGG-16|2007+2012|66.4|21|
![error analysis](./YOLOv1/error%20analysis.png)
# Generalization Ability
![generalization ability](./YOLOv1/generalization%20ability.png)
# Limitations
- Maximum of 49 objects can be detected
- Difficulty in detecting small objects that appear in groups
- Poor localization
# References
- [yolo original paper](https://arxiv.org/abs/1506.02640)
- [Video tutorials](https://youtu.be/zgbPj4lSc58?si=eVQiohUlznjTZj1Q) from Youtube uploader [ML For Nerds](https://www.youtube.com/@MLForNerds)
