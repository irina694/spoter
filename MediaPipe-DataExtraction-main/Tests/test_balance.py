import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_pickle('../WLASL100_train_25fps.pkl')
df_test = pd.read_pickle('../WLASL100_test_25fps.pkl')
df_val = pd.read_pickle('../WLASL100_val_25fps.pkl')

#print(df_val[df_val['video_id'] == '58365'])
#print(df_train[df_train['video_id'] == '58366'])
#df_val['labels'].plot.hist(bins=100)
print(df_test[df_test['labels']==0]['video_id'])
#plt.show()
