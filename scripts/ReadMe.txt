ʹ��CNN-static����
�������ã�
datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=25,k=400, filter_h=5)
U=numpy.asarray(W,dtype='float32')
 
perf = train_conv_net(datasets,U,
lr_decay=0.95,filter_hs=[3,4,5],conv_non_linear="relu",
hidden_units=[100,2064],shuffle_batch=True,n_epochs=35, sqr_norm_lim=9,
 non_static=non_static,
 batch_size=50,
 dropout_rate=[0.5])

ֻ���������title��������body��
ֻ����������40�������б�ѡΪ��ѻش��ߵ��û��ܹ�2064��
������10000�������н��в���