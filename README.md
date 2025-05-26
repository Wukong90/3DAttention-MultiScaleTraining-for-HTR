# 实验数据集

所提出的网络在两个具有挑战的最新中文手写文本数据集（SCUT-HCCDoc和SCUT-EPT）和一个重要的英文手写文本数据集IAM中进行验证。

在SCUT-HCCDoc数据集中，原始训练集包含文本行图像93,254幅，1,993张低质量（文字难以辨认、胡乱涂鸦、文字背景高度重叠以及字符不全）与竖直方向书写的图像被删除。实验中使用到的只使用了91,261张文本行图像用于网络训练。
目录Datasets_list/SCUT-HCCDoc/中train_list.txt与test_list.txt为我们使用的训练图像列表与测试图像列表，测试集包含了该数据集的原始全部测试图片。目录Datasets_list/SCUT-HCCDoc/abnormal_lists_and_images中是
我们排除的原始训练集中的低质量或竖直方向书写的手写文本图像。所有异常图像在abnormals目录中,all_abnormal_list.txt为所有异常图像列表，*_abnormal.txt为对应子集的异常图像列表，*为原始数据子集的名称。

在SCUT-EPT数据集中，681张包含有字符交换或重叠的异常手写文本被删除，实际只使用39,319副文本行图像用于训练。Datasets_list/SCUT-EPT/中的train_list.txt与test_list.txt为我们使用的训练图像列表与测试图像列表，测试集
包含了该数据集的原始全部测试图像。abnormal.txt为训练集中被排除的681张异常图片，它们的具体分类有兴趣的读者可以另外参考我们的项目https://github.com/Wukong90/EHT-Dataset.SCUT-EPT-Abnormal 。

标准的IAM数据集提供了一个训练集、两个验证集和一个测试集。目录Datasets_list/IAM/中的trainset.txt、validationset1.txt与validationset2.txt、
testset.txt为相应的训练集图像列表、验证集图像列表与测试集图像列表。我们的实验中使用了原始的全部训练数据作为网络训练，两个验证集全部数据用于选择最好模型进行测试集上的评估。

上述三个测试集中的所有图片都被用于评估模型的最终性能。
数据集SCUT-HCCDoc和SCUT-EPT可以分别从https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release 与 https://github.com/HCIILAB/SCUT-EPT_Dataset_Release?tab=readme-ov-file 申请获得。
IAM数据集可以从https://fki.tic.heia-fr.ch/databases/iam-handwriting-database 获得，需要注意的是自从2018年后，大多数相关研究者的工作采用了所谓的RWTH数据划分方式，它们与标准的IAM数据集划分并不相同。
