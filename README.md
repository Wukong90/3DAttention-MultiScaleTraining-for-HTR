-[English](#english)
-[中文](#中文)

---

### 中文

# 3维注意力多尺度训练网络(TDMTNet)

针对手写文本行识别的TDMTNet代码以及训练、测试代码已经开源(交叉熵损失辅助微调的代码暂未公布)。中文数据集的训练/测试代码为train_TDMSNet_Chinese.py，英文数据集的训练/测试代码为train_TDMSNet_eng.py。网络模型位于model/model.py中，configure中包含主要的配置文件、参数设置以及数据集构建与图像预处理代码，目录Datasets_list用于存放训练/测试图像数据以及文件名列表，目录weights中保存有我们在不同数据集中训练好的网络权重，名称中只含有CTC的目录表示该权重未经过CE损失微调，名称中包含CTC_CE表示网络权重经过CE微调。且它们包含有三个完整的分支。推理阶段，我们其实仅仅需要保留窗长长度为3所对应的分支。

# 实验数据集

所提出的网络在两个具有挑战的最新中文手写文本数据集（SCUT-HCCDoc和SCUT-EPT）和一个重要的英文手写文本数据集IAM中进行验证。 

在SCUT-HCCDoc数据集中，原始训练集包含文本行图像93,254幅，1,993张低质量（文字难以辨认、胡乱涂鸦、文字背景高度重叠以及字符不全）与竖直方向书写的图像被删除。因此实验中只使用了91,261张文本行图像用于网络训练。目录
Datasets_list/SCUT-HCCDoc/中train_list.txt与test_list.txt为我们使用的训练图像列表与测试图像列表，测试集包含了该数据集的原始全部测试图片。需要注意的是，该数据集的创建者只提供了篇章级文字图片以及它们中包含的文本行标注(json文件)，我们的列表中所列为文本行图像名，文本行的命名方式为 它所属于的原始篇章级图片名_它所在的行序号，行序号按照原始json文件中的文本行顺序。目录Datasets_list/SCUT-HCCDoc/abnormal_lists_and_images中是被排除的原始训练集中的低质量或竖直方向书写的手写文本图像。所有异常图像在abnormals目录中,all_abnormal_list.txt为所有异常图像列表，*_abnormal.txt为对应子集的异常文本图像名列表，*为原始数据子集的名称。

Datasets_list/SCUT-HCCDoc下的文件TrainDataRuChar2Int_HCCDoc.npy 与 TrainDataRuInt2Char_HCCDoc.npy 保存有我们在实验中使用的该数据集的字符与网络输出节点对应关系。

在SCUT-EPT数据集中，681张包含有字符交换或重叠的异常手写文本被删除，实际只使用39,319副文本行图像用于训练。Datasets_list/SCUT-EPT/中的train_list.txt与test_list.txt为我们使用的训练图像列表与测试图像列表，测试
集包含了该数据集的原始全部测试图像。abnormal.txt为训练集中被排除的681张异常图片列表，它们的具体分类可以另外参考我们的项目https://github.com/Wukong90/EHT-Dataset.SCUT-EPT-Abnormal 。

Datasets_list/SCUT-EPT/下的文件TrainDataRuChar2Int_EPT.npy 与 TrainDataRuInt2Char_EPT.npy 保存有我们在实验中使用的该数据集的字符与网络输出节点对应关系。

标准的IAM数据集提供了一个训练集、两个验证集和一个测试集。目录Datasets_list/IAM/split/中列出的trainset.txt、validationset1.txt、validationset2.txt、testset.txt为相应的训练列表、验证列表与测试列表。我们的实验中使用了原始的全部训练数据作为网络训练，两个验证集全部数据用于选择最好模型进行测试集上的评估，标准测试集中的所有图片都被用于评估模型的最终性能。

两个较新的具有挑战的手写中文数据集SCUT-HCCDoc和SCUT-EPT可以分别从https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release 与 https://github.com/HCIILAB/SCUT-EPT_Dataset_Release?tab=readme-ov-file 申请获得。手写英文数据集IAM可以从https://fki.tic.heia-fr.ch/databases/iam-handwriting-database 获得，需要注意的是自从2018年后，大多数相关研究者的工作采用了所谓的RWTH数据划分方式，它们与标准的IAM数据集划分并不相同。

---

### English

# Three-dimensional attention multi-scale training network (TDMTNet)

The TDMTNet code for handwritten text line recognition and the training and testing codes have been released (the cross entropy loss based fine-tuning code has not been released yet). The training/testing code for the Chinese dataset is train_TDMSNet_Chinese.py, and the training/testing code for the English dataset is train_TDMSNet_eng.py. The network model is located in model/model.py. The directory configure contains the main configuration files, parameter settings, and dataset construction and image preprocessing codes. The directory Datasets_list is used to store training/test image data and file name lists. The directory weights contains the trained network weights by using different datasets. A weight name only containing CTC indicates that the weight has not been fine-tuned by the CE loss while a weight name including CTC_CE indicates that the network weight has been fine-tuned by the CE loss. All weights contain three complete branches. In the inference stage, we actually only need to keep the branch corresponding to the window length of 3.

# Experimental datasets

The proposed network is validated on two latest Chinese handwritten text datasets (SCUT-HCCDoc and SCUT-EPT) and an important English handwritten text dataset IAM.

In the SCUT-HCCDoc dataset, the original training set contains 93,254 text line images, and 1,993 low-quality (illegible text, random graffiti, high overlap of text and background, and incomplete characters) and vertical writing text were deleted. Therefore, only 91,261 text line images were used for network training in the experiment. The train_list.txt and test_list.txt in the directory Datasets_list/SCUT-HCCDoc/ are the training image list and test image list we use. The test set contains all the original test images of the dataset. It should be noted that the creator of the dataset only provides page-level text images and the text line annotations (json files) contained in them. Our lists include the text line image names. The naming method of the text line name is the original page-level image name followed by the corresponding line number. The line number is the order of the text lines in the original annotation json file. The directory Datasets_list/SCUT-HCCDoc/abnormal_lists_and_images contains the low-quality and vertically written handwritten text images in the original training set that were excluded. All abnormal images are located in the abnormals directory, all_abnormal_list.txt is a list of all abnormal images, *_abnormal.txt is a list of abnormal text image names in the corresponding subset, and * is the name of the original data subset.

The files TrainDataRuChar2Int_HCCDoc.npy and TrainDataRuInt2Char_HCCDoc.npy under the directory Datasets_list/SCUT-HCCDoc store the corresponding relationship between the characters and network output nodes.

In the SCUT-EPT dataset, 681 abnormal handwritten texts including swapping and overlapped characters were deleted, and only 39,319 text line images were used for training. The train_list.txt and the test_list.txt in Datasets_list/SCUT-EPT/ are the training image list and the test image list we used in our experiments. The test set contains all original test images of the dataset. The file abnormal.txt is a list including 681 abnormal images that were excluded from the training set. About their detailed information, please refer to our project https://github.com/Wukong90/EHT-Dataset.SCUT-EPT-Abnormal.

The files TrainDataRuChar2Int_EPT.npy and TrainDataRuInt2Char_EPT.npy under the directory Datasets_list/SCUT-EPT/ store the corresponding relationship between the characters and network output nodes.

The standard IAM dataset provides a training set, two validation sets, and a test set. The trainset.txt, validationset1.txt, validationset2.txt, and testset.txt listed in the directory Datasets_list/IAM/split/ are the corresponding training list, validation lists, and test list. In our experiment, all the original training iamges were used for the network training, all images in two validation sets were used to select the best model and all original images in the standard test set were used to evaluate the final performance of the model.

Two latest challenging Chinese handwritten datasets SCUT-HCCDoc and SCUT-EPT can be obtained from https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release and https://github.com/HCIILAB/SCUT-EPT_Dataset_Release?tab=readme-ov-file respectively. The English handwritten dataset IAM can be obtained from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database. It should be noted that since 2018, most related researchers have adopted the so-called RWTH data partition, which is different from the standard IAM dataset partition.

 



