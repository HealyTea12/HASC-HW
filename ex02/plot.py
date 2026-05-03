import math

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("results.txt", skipinitialspace=True)

    sns.lineplot(data=data, x="n", y="f1", label="f1")
    sns.lineplot(data=data, x="n", y="f2", label="f2")
    sns.lineplot(data=data, x="n", y="f1_v", label="f1_v")
    sns.lineplot(data=data, x="n", y="f2_v", label="f2_v")
    sns.despine()
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("value")
    xticks = data["n"].tolist()
    plt.xticks(xticks, [fr"$2^{{{int(math.log2(value))}}}$" for value in xticks])
    plt.legend()
    plt.tight_layout()
    # write to tfile
    plt.savefig("results.png")
