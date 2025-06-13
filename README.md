**Read this in other languages: [English](README.md),[中文](README_zh.md).**

# Three-dimensional attention multi-scale training network (TDMTNet)

The TDMTNet code for handwritten text line recognition and the training and testing codes have been released (the cross entropy loss based fine-tuning code has not been released yet). The training/testing code for the Chinese dataset is train_TDMSNet_Chinese.py, and the training/testing code for the English dataset is train_TDMSNet_eng.py. The network model is located in model/model.py. The directory configure contains the main configuration files, parameter settings, and dataset construction and image preprocessing codes. The directory Datasets_list is used to store training/test image data and file name lists. The directory weights contains the trained network weights by using different datasets. A weight name only containing CTC indicates that the weight has not been fine-tuned by the CE loss while a weight name including CTC_CE indicates that the network weight has been fine-tuned by the CE loss. All weights contain three complete branches. In the inference stage, we actually only need to keep the branch corresponding to the window length of 3.

# Experimental datasets

The proposed network is validated on two latest Chinese handwritten text datasets (SCUT-HCCDoc and SCUT-EPT) and an important English handwritten text dataset IAM.

In the SCUT-HCCDoc dataset, the original training set contains 93,254 text line images, and 1,993 low-quality (illegible text, random graffiti, high overlap of text and background, and incomplete characters) and vertical writing text were deleted. Therefore, only 91,261 text line images were used for network training in the experiment. The train_list.txt and test_list.txt in the directory Datasets_list/SCUT-HCCDoc/ are the training image list and test image list we used in our experiments. The test set contains all original test images of the dataset. It should be noted that the creator of the dataset only provides page-level text images and the text line annotations (json files) contained in them. Our lists include the text line image names. The naming method of the text line name is the original page-level image name followed by the corresponding line number. A line number is the order of the corresponding text line in the original annotation json file. The directory Datasets_list/SCUT-HCCDoc/abnormal_lists_and_images contains the low-quality and vertically written handwritten text images in the original training set that were excluded. All abnormal images are located in the abnormals directory, all_abnormal_list.txt is a list of all abnormal images, *_abnormal.txt is a list of abnormal text image names in the corresponding subset, and * is the name of the original data subset.

The files TrainDataRuChar2Int_HCCDoc.npy and TrainDataRuInt2Char_HCCDoc.npy under the directory Datasets_list/SCUT-HCCDoc store the corresponding relationship between the characters and network output nodes.

In the SCUT-EPT dataset, 681 abnormal handwritten texts including swapping and overlapped characters were deleted, and only 39,319 text line images were used for training. The train_list.txt and the test_list.txt in Datasets_list/SCUT-EPT/ are the training image list and the test image list we used in our experiments. The test set contains all original test images of the dataset. The file abnormal.txt is a list including 681 abnormal images that were excluded from the training set. About their detailed information, please refer to our project https://github.com/Wukong90/EHT-Dataset.SCUT-EPT-Abnormal.

The files TrainDataRuChar2Int_EPT.npy and TrainDataRuInt2Char_EPT.npy under the directory Datasets_list/SCUT-EPT/ store the corresponding relationship between the characters and network output nodes.

The standard IAM dataset provides a training set, two validation sets, and a test set. The trainset.txt, validationset1.txt, validationset2.txt, and testset.txt listed in the directory Datasets_list/IAM/split/ are the corresponding training list, validation lists, and test list. In our experiments, all original training iamges were used for training, all images in two validation sets were used to select the best model and all original images in the standard test set were used to evaluate the final performance of the model.

Two latest challenging Chinese handwritten datasets SCUT-HCCDoc and SCUT-EPT can be obtained from https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release and https://github.com/HCIILAB/SCUT-EPT_Dataset_Release?tab=readme-ov-file respectively. The English handwritten dataset IAM can be obtained from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database. It should be noted that since 2018, most related researchers have adopted the so-called RWTH data partition, which is different from the standard IAM dataset partition.

 



