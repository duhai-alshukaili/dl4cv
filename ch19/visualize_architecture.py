import sys, pathlib

# Append the parent directory of ch07 to sys.path once
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from pyimagesearch.nn.conv import VGGNet5
from keras import utils

# initialize a VGGNet5 model then
# write the model architecture to a file
model = VGGNet5.build(32, 32, 3, 10)
utils.plot_model(model, to_file="vggnet5.png", show_shapes=True, show_layer_names=False)

# display the model architecture
from IPython.display import Image
Image(filename="vggnet5.png", width=800, height=600)
