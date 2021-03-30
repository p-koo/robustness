import os
from six.moves import cPickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import argparse
import helper, tfomics
from model_zoo import cnn_dist as genome_model

#-----------------------------------------------------------------

base_name = 'cnn_dist'
batch_size = 32
num_trials = 5
results_path = helper.make_directory('../results2', 'synthetic')


# load data
data_path = '../data' 
filepath = os.path.join(data_path, 'synthetic_code_dataset.h5')
x_train, y_train, x_valid, y_valid, x_test, y_test, model_test = helper.load_data(filepath)
N, L, A = x_train.shape
num_labels = y_train.shape[1]


for reg in [False]:
  if reg:
    dropout=[0.1, 0.2, 0.3, 0.4, 0.5] 
    bn=[True, True, True, True, True]
  else:
    dropout = [0, 0, 0, 0, 0]
    bn = [False, False, False, False, False]


  for activation in ['exponential', 'relu']:
    for other_activation in ['relu']:

      for trial in range(num_trials):
        if reg:
          name = base_name+'_reg'
        else:
          name = base_name+ '_noreg'
        name += '_'+str(activation)
        name += '_'+str(trial)

        model = genome_model.model(input_shape=(L,A), num_labels=1, activation=activation, 
                                   other_activation=other_activation, dropout=dropout, bn=bn, l2=None)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
        optimizer = keras.optimizers.Adam(learning_rate=0.0003)


        if reg:
          history, trainer = tfomics.fit.fit_lr_decay(model, loss, optimizer, x_train, y_train, validation_data=(x_valid, y_valid), 
                                                 num_epochs=200, batch_size=batch_size, shuffle=True, metrics=['auroc','aupr'], 
                                                 es_patience=15, es_metric='auroc', es_criterion='max',
                                                 lr_decay=0.3, lr_patience=5, lr_metric='auroc', lr_criterion='max')
        else:
          history, trainer = tfomics.fit.fit(model, loss, optimizer, x_train, y_train, validation_data=(x_valid, y_valid), verbose=True,  
                                           metrics=['auroc', 'aupr'], num_epochs=100, batch_size=batch_size, shuffle=True, 
                                           es_patience=100, es_metric='auroc', es_criterion='max')

        
        #-----------------------------------------------------------------
        # save model
        model.save_weights(os.path.join(results_path, 'weights_'+name+'.h5'))

        #-----------------------------------------------------------------
        # Evaluate model on test set  (This uses Tensorflow running metrics -- not sklearn)
        testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        trainer.evaluate('test', testset, batch_size=128)

        # save history
        history = trainer.get_metrics('test', history)
        with open(os.path.join(results_path, 'history_'+name+'.pickle'), 'wb') as f:
          cPickle.dump(history, f)

        #-----------------------------------------------------------------
        # Interpretability analysis

        # get positive label sequences and sequence model
        pos_index = np.where(y_test[:,0] == 1)[0][:500]   
        X = x_test[pos_index]
        X_model = model_test[pos_index]

        # calculate attribution maps
        explainer = tfomics.explain.Explainer(model, class_index=0)
        mutagenesis_scores = explainer.mutagenesis(X, class_index=None)
        saliency_scores = explainer.saliency_maps(X)
        smoothgrad_scores = explainer.smoothgrad(X, num_samples=50, mean=0.0, stddev=0.1)
        intgrad_scores = explainer.integrated_grad(X, baseline_type='zeros')
        expintgrad_scores = explainer.expected_integrated_grad(X, num_baseline=50, baseline_type='random')

        # reduce attribution maps to 1D scores
        mut_scores = tfomics.explain.l2_norm(mutagenesis_scores)
        sal_scores = tfomics.explain.grad_times_input(X, saliency_scores)
        sg_scores = tfomics.explain.grad_times_input(X, smoothgrad_scores)
        int_scores = tfomics.explain.grad_times_input(X, intgrad_scores)
        expint_scores = tfomics.explain.grad_times_input(X, expintgrad_scores)

        # compare distribution of attribution scores at positions with and without motifs
        threshold = 0.1
        mutagenesis_roc, mutagenesis_pr = tfomics.evaluate.interpretability_performance(mut_scores, X_model, threshold)
        saliency_roc, saliency_pr = tfomics.evaluate.interpretability_performance(sal_scores, X_model, threshold)
        smoothgrad_roc, smoothgrad_pr = tfomics.evaluate.interpretability_performance(sg_scores, X_model, threshold)
        intgrad_roc, intgrad_pr = tfomics.evaluate.interpretability_performance(int_scores, X_model, threshold)
        expintgrad_roc, expintgrad_pr = tfomics.evaluate.interpretability_performance(expint_scores, X_model, threshold)

        # compare distribution of attribution scores at positions with and without motifs
        top_k = 10
        mut_signal, mut_noise_max, mut_noise_mean, mut_noise_topk = tfomics.evaluate.signal_noise_stats(mut_scores, X_model, top_k, threshold)
        sal_signal, sal_noise_max, sal_noise_mean, sal_noise_topk = tfomics.evaluate.signal_noise_stats(sal_scores, X_model, top_k, threshold)
        sg_signal, sg_noise_max, sg_noise_mean, sg_noise_topk = tfomics.evaluate.signal_noise_stats(sg_scores, X_model, top_k, threshold)
        int_signal, int_noise_max, int_noise_mean, int_noise_topk = tfomics.evaluate.signal_noise_stats(int_scores, X_model, top_k, threshold)
        expint_signal, expint_noise_max, expint_noise_mean, expint_noise_topk = tfomics.evaluate.signal_noise_stats(expint_scores, X_model, top_k, threshold)

        # compile and save results
        results = [mutagenesis_roc, mutagenesis_pr, 
                   saliency_roc, saliency_pr,
                   smoothgrad_roc, smoothgrad_pr,
                   intgrad_roc, intgrad_pr,
                   expintgrad_roc, expintgrad_pr,
                   mut_signal, mut_noise_max, mut_noise_mean, mut_noise_topk, 
                   sal_signal, sal_noise_max, sal_noise_mean, sal_noise_topk, 
                   sg_signal, sg_noise_max, sg_noise_mean, sg_noise_topk, 
                   int_signal, int_noise_max, int_noise_mean, int_noise_topk, 
                   expint_signal, expint_noise_max, expint_noise_mean, expint_noise_topk]
        headers = ['mutagenesis_roc', 'mutagenesis_pr', 
                   'saliency_roc', 'saliency_pr',
                   'smoothgrad_roc', 'smoothgrad_pr',
                   'intgrad_roc', 'intgrad_pr',
                   'expintgrad_roc', 'expintgrad_pr',
                   'mut_signal', 'mut_noise_max', 'mut_noise_mean', 'mut_noise_topk', 
                   'sal_signal', 'sal_noise_max', 'sal_noise_mean', 'sal_noise_topk', 
                   'sg_signal', 'sg_noise_max', 'sg_noise_mean', 'sg_noise_topk', 
                   'int_signal', 'int_noise_max', 'int_noise_mean', 'int_noise_topk', 
                   'expint_signal', 'expint_noise_max', 'expint_noise_mean', 'expint_noise_topk']
        df = pd.DataFrame(data=np.array(results).T, columns=headers)
        df.to_csv(os.path.join(results_path, 'results_'+name+'.tsv'), sep='\t', index=False, float_format="%.5f")

        #----------------------------------------------------------
        # plot results
        #----------------------------------------------------------

        # Plot performance as a box-violin plot
        score_names = ['saliency_scores', 'mut_scores', 'intgrad_scores', 'smoothgrad_scores', 'exp_intgrad_scores']
        names = ['Saliency', 'Mutagenesis', 'Integrated-Grad', 'SmoothGrad', 'Expected IntGrad']
        scores = [saliency_roc, mutagenesis_roc, intgrad_roc, smoothgrad_roc, expintgrad_roc]
        fig = plt.figure(figsize=(7,4)) 
        ax = tfomics.impress.box_violin_plot(scores, ylabel='AUROC', xlabel=names)
        ax.set_ybound([.55,1.0])
        scores = [saliency_pr, mutagenesis_pr, intgrad_pr, smoothgrad_pr, expintgrad_pr]
        fig = plt.figure(figsize=(7,4)) 
        ax = tfomics.impress.box_violin_plot(scores, ylabel='AUPR', xlabel=names)
        ax.set_ybound([.35,1.0])
        #ax.tick_params(labelbottom=False) 
        outfile = os.path.join(results_path, 'auc_'+name+'.pdf')
        fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
        plt.close()

        # plot signal to mean noise ratio
        scores = [sal_signal/sal_noise_mean, 
                  mut_signal/mut_noise_mean,
                  int_signal/int_noise_mean,
                  sg_signal/sg_noise_mean,
                  expint_signal/expint_noise_mean]
        names = ['Saliency', 'Mutagenesis', 'Integrated-Grad', 'SmoothGrad', 'Expected IntGrad']
        fig = plt.figure(figsize=(6,4))
        ax = tfomics.impress.box_violin_plot(scores, ylabel='SNR', xlabel=names)
        ax.set_ybound([0, 30])
        outfile = os.path.join(results_path, 'snr_'+name+'.pdf')
        fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
        plt.close()

        # plot signal to top-k noise ratio
        scores = [sal_signal/sal_noise_topk, 
                  mut_signal/mut_noise_topk,
                  int_signal/int_noise_topk,
                  sg_signal/sg_noise_topk,
                  expint_signal/expint_noise_topk]
        names = ['Saliency', 'Mutagenesis', 'Integrated-Grad', 'SmoothGrad', 'Expected IntGrad']
        fig = plt.figure(figsize=(6,4))
        ax = tfomics.impress.box_violin_plot(scores, ylabel='SNR', xlabel=names)
        ax.set_ybound([0, 10])
        outfile = os.path.join(results_path, 'signal_topk_'+name+'.pdf')
        fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
        plt.close()



