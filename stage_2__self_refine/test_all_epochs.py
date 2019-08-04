import os
from os.path import basename, splitext
import tempfile
import shutil
from string import ascii_uppercase
import random
import glob

### create small temporary test set
dataroot = "/home/shared/datasets/cars.merged.new"
checkpoints_dir = "/home/arash/Desktop/checkpoints"
run_name = "cars.stage2.back2back"
test_size = 500

random.seed(42)
test_A_pattern = os.path.join(dataroot, 'test_A', '*')
test_A_paths = random.sample(glob.glob(test_A_pattern), test_size)

# creating a temporary dataset
tmp_dataroot = tempfile.mkdtemp()
for letter in ascii_uppercase[:5]:
    subfolder = 'test_{}'.format(letter)  # test_A, test_B, etc.

    dst_dir = os.path.join(tmp_dataroot, subfolder)
    os.mkdir(dst_dir)

    for test_A_path in test_A_paths:
        src = os.path.join(dataroot, subfolder, basename(test_A_path))
        dst = os.path.join(dst_dir, basename(src))
        shutil.copyfile(src, dst)

try:
    start, end, step = 10, 150 + 10, 10  # range of epochs to evaluate validation results of
    epochs = list(range(start, end, step))

    for epoch in epochs:
        print('Evaluating test results, epoch {}'.format(epoch))
        os.system(
            "~/anaconda3/bin/python test.py --gpu_ids 0 --no_flip --name {} --label_nc 0 --loadSize 256 --input_nc 3 --output_nc1 2 --output_nc2 2 --batchSize 1 --dataroot {} --checkpoints_dir {} --which_epoch {} --how_many -1".format(
                run_name, tmp_dataroot, checkpoints_dir, epoch))

    print("Creating HTML Output")
    html = \
        """<!DOCTYPE html>
        <html>
        <body>
    
    
        <h2>Test Results</h2>
    
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
        """    <th>Ground Truth</th>"""

    html += \
        """  </tr>
        """

    # add images
    for test_image in test_A_paths:
        html += \
            """  <tr>
            """

        for epoch in epochs:
            html += \
                """    <td><img src="./{}/test_{}/images/{}_synthesized_image2.jpg"></td>
                """.format(run_name, epoch, splitext(basename(test_image))[0])
        html += \
            """    <td><img src="./{}/test_{}/images/{}_real_image_and_edge.jpg"></td>
            """.format(run_name, start, splitext(basename(test_image))[0])

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
    print("Removing the temporary directory")
    shutil.rmtree(tmp_dataroot)

with open("./results/test_results.html", "w") as html_file:
    print(html, file=html_file)
