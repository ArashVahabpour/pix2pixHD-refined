import os
from os.path import basename, splitext
import tempfile
import shutil
from string import ascii_uppercase
import random
import glob


### create small temporary test set
dataroot = "/home/shared/datasets/cars.merged.new"
test_size = 500

random.seed(42)
test_A_pattern = os.path.join(dataroot, 'test_A', '*')
test_A_paths = random.sample(glob.glob(test_A_pattern), test_size)

tmp_dataroot = tempfile.mkdtemp()
for letter in ascii_uppercase[:5]:
	os.mkdir(os.path.join())

try:
	start, end, step = 50, 70, 10#150, 10  # range of epochs to evaluate validation results of
	epochs = list(range(start, end + step, step))

	for epoch in epochs:
		print('Evaluating epoch {} validation results'.format(epoch))
		os.system("~/anaconda3/bin/python test.py --gpu_ids 0 --net_idx 0 --no_flip --num_nets 8 --name cars.merged.context.0 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot {} --phase val".format(dataroot))  


	html = \
	"""<!DOCTYPE html>
	<html>
	<body>


	<h2>Validation Results</h2>

	<table style="width:100%">"""

	# add table headers
	html += \
	"""  <tr>
	"""

	for epoch in epochs:
		html += \
		"""    <th>Epoch {}</th>
		""".format(epoch)

	html += \
	"""  </tr>
	"""

	# add images
	for test_image in test_images:
		html += \
		"""  <tr>
		"""

		for epoch in epochs:
			html += \
			"""    <td><img src="../val_{}/{}_synthesized_image.png"></td>
			""".format(epoch, splitext(basename(test_image))[0])

		html += \
		"""  </tr>
		"""


	html += \
	"""</table>

	</body>
	</html>"""
except Exception as e:
	print(e)
finally:
	shutil.rmtree(dirpath)

with open("./results/val_results.html", "w") as html_file:
    print(html, file=html_file)