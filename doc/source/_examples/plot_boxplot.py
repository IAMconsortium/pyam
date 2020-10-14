"""
=========================
Plot Data as a boxplot
=========================

"""
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import pyam

###############################
# Read in the data from the first-steps tutorial and show a summary

df = pyam.IamDataFrame('tutorial_data.csv')
df

###############################
# We generate a simple boxplot of CO2 emissions
# across one scenario implemented by a range of models.

data = df.filter(scenario='CD-LINKS_NPi2020_1000',
                 variable='Emissions|CO2', region='World')

fig, ax = plt.subplots()
data.boxplot(x='year', ax=ax)
plt.show()

###############################
# We can add sub-groupings of the data by using the keyword argument `by`.

data = (
    df.filter(scenario='CD-LINKS_NPi2020_1000', variable='Emissions|CO2',
              year=[2010, 2020, 2030, 2050, 2100])
    .filter(region='World', keep=False)
)

fig, ax = plt.subplots()
data.boxplot(x='year', by='region', legend=True, ax=ax)

# We can use any matplotlib arguments to make the figure more appealing.
plt.legend(loc=1)
plt.show()
