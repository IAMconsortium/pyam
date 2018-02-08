import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


from testing_utils import test_ia


@image_comparison(baseline_images=['line_plot'],
                  extensions=['png'])
def test_line_plot(test_ia):
    fig, ax = plt.subplots()
    data = test_ia.filter(filters={'variable': 'Primary Energy'})
    data.line_plot(ax=ax)
