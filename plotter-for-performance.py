import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

data = np.loadtxt('./reward_vs_iteration.dat')
df = pd.DataFrame(data, columns=['training', 'performance', 'steps'])
df.loc[:, 'inverse-performance'] = 1.0/df.loc[:, 'performance']
df.loc[:, 'mistaken-steps'] = df.loc[:, 'steps'] - df.loc[:, 'performance'] * df.loc[:, 'steps']

averaged_50 = df.rolling(128).mean().dropna(axis=0)

fig, ax = plt.subplots()

ax.set(yscale='log', xscale='linear')

ax.set(xlabel='training', ylabel='inverse-performance')
seaborn.lineplot(x='training', y='inverse-performance', data=df, ax=ax)
seaborn.lineplot(x='training', y='inverse-performance', data=averaged_50, ax=ax)
#plt.show()
plt.savefig('inverse-performance-vs-training')
plt.clf()

fig, ax = plt.subplots()
ax.set(yscale='linear', xscale='linear')
ax.set(xlabel='training', ylabel='performance')
seaborn.lineplot(x='training', y='performance', data=df)
seaborn.lineplot(x='training', y='performance', data=averaged_50)
#plt.show()
plt.savefig('performance-vs-training')
plt.clf()

fig, ax = plt.subplots()
ax.set(yscale='linear', xscale='linear')
ax.set(xlabel='training', ylabel='mistaken-steps')
seaborn.lineplot(x='training', y='mistaken-steps', data=df)
seaborn.lineplot(x='training', y='mistaken-steps', data=averaged_50)
#plt.show()
plt.savefig('mistaken-vs-training')
plt.clf()
