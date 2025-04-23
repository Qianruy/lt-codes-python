import pandas as pd
import matplotlib.pyplot as plt

fname = './experiments/fcfp_2025042121_0.01.csv'  
df    = pd.read_csv(fname)

iters = df['iter'].to_numpy()         
vals  = df.drop(columns='iter').to_numpy()   

# sample or avg the data points

x = range(vals.shape[1]-1)              
for i, row in enumerate(vals):
    if i > 400: break
    assert(row[1:].max() < 200)
    if i in [0, 1]:
        plt.plot(x, row[1:], label=f'it {iters[i]}')
plt.xlabel('code-word index (0 - 3w-1)')
plt.ylabel('remaining degree')
plt.title('Degree profile every 100 iterations')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.imshow(vals, aspect='auto', origin='lower')
plt.colorbar(label='degree')
plt.yticks(range(len(iters)), iters)   # label rows with iteration number
plt.xlabel('code-word index')
plt.ylabel('iteration')
plt.title('Remaining-degree heat-map')
plt.tight_layout()
plt.show()