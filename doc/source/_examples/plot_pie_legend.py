"""
======================================
Plot Data as a Pie Chart with a Legend
======================================

"""
import matplotlib.pyplot as plt
import pyam_analysis as iam

fname = 'msg_input.csv'

df = iam.IamDataFrame(fname)

df = df.filter({'variable': 'Emissions|CO2|*',
                'level': 0,
                'year': 2050,
                'region': 'World'})

print(df.head())


fig, ax = plt.subplots(figsize=(10, 10))

df.pie_plot(ax=ax, labels=None, legend=True)

fig.subplots_adjust(right=0.55, left=-0.05)
plt.show()
