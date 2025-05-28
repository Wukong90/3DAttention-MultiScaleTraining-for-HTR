'''
@author : Zi-Rui Wang
@time : 2024
@github : https://github.com/Wukong90
'''
common_config = {
    #For HCCDoc Dataset
    'train_data_dir': './Datasets_lists/SCUT-HCCDoc/train_jpg/',
    'train_list': './Datasets_lists/SCUT-HCCDoc/train_list.txt',
    'train_label_list': './Datasets_lists/SCUT-HCCDoc/train_lables.txt',
    'test_data_dir': './Datasets_lists/SCUT-HCCDoc/test_jpg/',
    'test_list': './Datasets_lists/SCUT-HCCDoc/test_list.txt',
    'test_label_list': './Datasets_lists/SCUT-HCCDoc/test_labels.txt',
    #For EPT Dataset
    #train_data_dir': './Datasets_lists/SCUT-EPT/train_jpg/',
    #train_list': './Datasets_lists/SCUT-EPT/train_list.txt',
    #'train_label_list': './Datasets_lists/SCUT-EPT/train_lables.txt',
    #'test_data_dir': './Datasets_lists/SCUT-EPT/test_jpg/',
    #'test_list': './Datasets_lists/SCUT-EPT/test_list.txt',
    #'test_label_list': './Datasets_lists/SCUT-EPT/test_labels.txt',
    'map_to_seq_hidden': 128,
    'rnn_hidden': 256,
}

train_config = {
    'epochs': 180, #HCCDoc,For EPT dataset you can use epochs = 300
    'train_batch_size': 20,
    'eval_batch_size': 64,
    'show_interval': 1000,
    'cpu_workers': 4,
    'decode_method': 'greedy',
    'beam_size': 10,
}
train_config.update(common_config)
