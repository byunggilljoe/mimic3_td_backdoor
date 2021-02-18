import sys
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#plt.rc('font',family='Times New Roman') 
if __name__ == "__main__":
    rootdir = sys.argv[1]
    metric = sys.argv[2]
    print("rootdir:", rootdir)

    metric_keyword_dict = {"acc": "accuracy", "pc0": "precision class 0", "pc1": "precision class 1", \
                           "rc0": "recall class 0", "rc1": "recall class 1", "roc": "AUC of ROC", \
                           "prc": "AUC of PRC", "mps": "min(+P, Se)", "trs": "Trigger", \
                            }
    title_dict = {"acc": "accuracy", "pc0": "precision class 0", "pc1": "precision class 1", \
                           "rc0": "recall class 0", "rc1": "recall class 1", "roc": "AUC of ROC", \
                           "prc": "AUC of PRC", "mps": "min(+P, Se)", "trs": "Trigger success ratio", \
                            }
    
    
    metric_keyword = metric_keyword_dict[metric]
    df_empty = pd.DataFrame()

    no_poisoning_result = None
    for file in glob.glob("{}/**/*.txt".format(rootdir)):
        proportion, strength = [float(a) for a  in file.split("/")[-2].split("_")]
        
        lines = open(file).readlines()
        for l in lines:
            if l.startswith(metric_keyword):
                if ":" in l:
                    val = l.split(":")[-1]
                else:
                    val = l.split("=")[-1]
                val = float(val.strip())

                if proportion == 0:
                    no_poisoning_result = val
                    break

                df_empty.loc[strength, proportion] = val
    
    index_sorted_df = df_empty.sort_index()
    column_sorted_df = index_sorted_df.reindex(sorted(index_sorted_df.columns), axis=1)
    
    print(column_sorted_df)
    fig = sns.heatmap(column_sorted_df, annot=True)
    
    plt.title(title_dict[metric]+" (No poisoning : {:.02f})".format(no_poisoning_result))
    plt.xlabel("Poisoning proportion")
    plt.ylabel("Trigger mahalanobis distance")
    plt.savefig("heatmap_{}.png".format(metric_keyword))

                
        