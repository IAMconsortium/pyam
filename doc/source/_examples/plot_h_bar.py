"""
==================================
Plot Data as a Horizontal Bar Plot
==================================

"""
import matplotlib.pyplot as plt
import pyam_analysis as iam

fname = 'msg_input.csv'

df = iam.IamDataFrame(fname)

df = df.filter({'variable': 'Emissions|CO2|*',
                'level': 0,
                'region': 'World'})

print(df.head())

df.bar_plot(stacked=True, orient='h')

plt.show()
