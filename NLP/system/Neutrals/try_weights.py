from NLP.system.Neutrals.Train import train_svc
import numpy as np


total = []
for wob in np.arange(1.0, 2.5, 1.0):
    for woh in np.arange(1.0, 3.5, 2.0):
        for wom in np.arange(1.0, 3.5, 2.0):
            for n_h in np.arange(1.0, 6.0, 2.0):
                for n_m in np.arange(1.0, 6.0, 2.0):
                    for n_p in np.arange(1.0, 6.0, 2.0):
                        for n_n in np.arange(1.0, 6.0, 2.0):
                            for sent in np.arange(1.0, 6.0, 2.0):
                                print(wob, woh, wom, n_h, n_m, n_p, n_n, sent)
                                weights = [wob, woh, wom, n_h, n_m, n_p, n_n, sent]
                                res = train_svc("F:/MultiStanceCat-IberEval-training-20180404/es.xml",
                                          "F:/MultiStanceCat-IberEval-training-20180404/truth-es.txt",
                                          None,
                                          "F:/Sentimientos", weights,
                                          None, None, None, None)
                                total.append(res)
