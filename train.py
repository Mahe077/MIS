import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import os
import sys
import torch

try:
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    rows, cols = 4, 4  # You can adjust the dimensions as needed

    # Generate a 2D matrix containing coordinates
    coordinate_matrix = torch.tensor(
        [[(i, j) for j in range(cols)] for i in range(rows)])

    loss_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    model_start_time = time.time()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch)

            # if total_steps % opt.print_freq == 0:
            #     errors = model.get_current_errors()
            #     t = (time.time() - iter_start_time) / opt.batchSize
            #     visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            #     if opt.display_id > 0:
            #         visualizer.plot_current_errors(epoch, float(
            #             epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    with open(loss_log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Total Time : (%.3f) ================\n' % (
            time.time() - model_start_time))
        log_file.write(
            '================ Training Loss Ended(%s) ================\n' % now)  # TODO: log thes on something
    sys.exit(0)  # Exit with success code
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)  # Exit with error code
