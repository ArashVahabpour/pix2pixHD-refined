### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
from util import html
import numpy as np

def run_test(opt, model, epoch):
    print('============')
    print('generating test results\n')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


    for i, data in enumerate(dataset):
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()

        generated1, generated2 = model.module.inference(
            data['label'], data['inst'], data['image'], data['context_all'], data['context_single'])

        input_ = np.vstack([util.tensor2label(data['label'][0], 0),
                            util.tensor2label(data['context_all'][0], 0),
                            util.tensor2label(data['context_single'][0], 0)
                            ])
        visuals = OrderedDict([('input_label', input_),
                               ('synthesized_image1', util.tensor2im(generated1.data[0])),
                               ('synthesized_image2', util.tensor2im(generated2.data[0]))])

        img_path = data['path']
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()
    print('\n============')

