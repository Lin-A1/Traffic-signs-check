# Traffic signs check

- 2024 iFLYTEK A.I.交通标识识别初赛100%精度,复赛忘了参加
- 链接：https://challenge.xfyun.cn/topic/info?type=traffic-signs

## 数据说明

所提供的数据大约包含62个标志类别和9000+对应的交通标识图片，图像分为训练数据集和测试数据集两个子数据集。训练数据集包含9808幅图像，测试集A包含968幅图像。本赛题测试集B将于8月15日发布。

其中交通标识名称对应的label如下表：
| Sign Name                          | Label |
|------------------------------------|-------|
| Speed limit (5km)                 | 0     |
| Speed limit (15km)                | 1     |
| Speed limit (20km)                | 2     |
| Speed limit (30km)                | 3     |
| Speed limit (40km)                | 4     |
| Speed limit (50km)                | 5     |
| Speed limit (60km)                | 6     |
| Speed limit (70km)                | 7     |
| Speed limit (80km)                | 8     |
| Speed limit (100km)               | 9     |
| Speed limit (120km)               | 10    |
| End of speed limit                 | 11    |
| End of speed limit (50km)         | 12    |
| End of speed limit (80km)         | 13    |
| Don't overtake from Left           | 14    |
| No stopping                        | 15    |
| No U-turn                          | 16    |
| No Car                             | 17    |
| No horn                            | 18    |
| No entry                           | 19    |
| No passage                         | 20    |
| Don't Go Right                     | 21    |
| Don't Go Left or Right             | 22    |
| Don't Go Left                      | 23    |
| Don't Go straight                  | 24    |
| Don't Go straight or Right         | 25    |
| Don't Go straight or Left          | 26    |
| Go right or straight               | 27    |
| Go left or straight                | 28    |
| Village                            | 29    |
| U-turn                             | 30    |
| ZigZag Curve                       | 31    |
| Bicycles crossing                  | 32    |
| Keep Right                         | 33    |
| Keep Left                          | 34    |
| Roundabout mandatory                | 35    |
| Watch out for cars                 | 36    |
| Slow down and give way             | 37    |
| Continuous detours                 | 38    |
| Slow walking                       | 39    |
| Horn                               | 40    |
| Uphill steep slope                 | 41    |
| Downhill steep slope               | 42    |
| Under Construction                  | 43    |
| Heavy Vehicle Accidents            | 44    |
| Parking inspection                  | 45    |
| Stop at intersection                | 46    |
| Train Crossing                     | 47    |
| Fences                             | 48    |
| Dangerous curve to the right       | 49    |
| Go Right                           | 50    |
| Go Left or right                   | 51    |
| Dangerous curve to the left        | 52    |
| Go Left                            | 53    |
| Go straight                        | 54    |
| Go straight or right               | 55    |
| Children crossing                  | 56    |
| Care bicycles crossing              | 57    |
| Danger Ahead                       | 58    |
| Traffic signals                    | 59    |
| Zebra Crossing                     | 60    |
| Road Divider                       | 61    |

## 数据预处理

1. **超分辨率重构**：
   - 使用 OpenCV 的 DNN 模块加载超分辨率模型（`ESPCN_x4.pb`），通过 `DnnSuperResImpl_create` 方法创建超分辨率实例。
   - 在 `__call__` 方法中，将输入图像转换为 OpenCV 的格式，然后进行超分辨率处理，最后再转换回 PIL 图像格式。

2. **数据预处理管道**：
   - 使用 `transforms.Compose` 创建了一个包含多个变换的管道，这些变换包括：
     - **颜色调整**：对亮度、对比度、饱和度和色相进行随机调整，以增强数据的多样性。
     - **灰度转换**：将图像转换为灰度图，但保留三个通道。
     - **尺寸调整**：将图像调整为 224x224 的尺寸，适合输入到神经网络。
     - **随机旋转**：随机旋转图像最多 15 度，以增加训练的多样性。
     - **张量转换**：将图像转换为 PyTorch 张量。
     - **归一化**：使用给定的均值和标准差对图像进行归一化，以适配预训练模型的输入要求。

## 模型
 ### `CA+RESNET`

1. **CA_Block**：
   - 这是一个通道注意力块，利用全局平均池化生成通道注意力图。
   - `avg_pool_x` 和 `avg_pool_y` 分别对输入特征图进行高度和宽度的全局平均池化。
   - 通过 1x1 卷积减少通道维度，并使用 ReLU 激活和批归一化进行处理。
   - 然后，分离通道注意力图，并使用 Sigmoid 激活生成最终的注意力权重。
   - 输入特征图通过这些权重进行缩放，以突出重要特征。
2. **ResnetCA**：
   - 继承自 `nn.Module`，整合了 ResNet18 模型和 CA_Block。
   - 使用 `self.resnet` 提取特征，去掉最后两层，以便于后续的处理。
   - 在特征图上应用 CA_Block，生成增强后的特征图。
   - 最后通过全局平均池化层和全连接层生成最终输出，输出的类别数为 62。

## 评估
详情见： [training_log.csv](logs/training_log.csv) 

