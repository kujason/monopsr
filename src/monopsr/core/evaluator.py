"""Common functions for evaluating checkpoints.
"""
from multiprocessing import Process
import os
import sys
import time

import numpy as np
import tensorflow as tf

from monopsr.core import constants
from monopsr.core import evaluator_utils
from monopsr.core import checkpoint_utils
from monopsr.core import summary_utils
from monopsr.core.models.monopsr.monopsr_output_builder import MonoPSROutputBuilder


class Evaluator:

    def __init__(self,
                 model,
                 config,
                 eval_mode,
                 skip_evaluated_checkpoints=True,
                 eval_wait_interval=30,
                 do_kitti_native_eval=True):
        """Evaluator class for evaluating model's detection output.

        Args:
            model: An instance of DetectionModel
            config: Experiment configuration
            eval_mode: Evaluation mode ('val' or 'test')
            skip_evaluated_checkpoints: (optional) Enables checking evaluation
                results directory and if the folder names with the checkpoint
                index exists, it 'assumes' that checkpoint has already been
                evaluated and skips that checkpoint.
            eval_wait_interval: (optional) The number of seconds between looking
                for a new checkpoint.
            do_kitti_native_eval: (optional) flag to enable running kitti native
                eval code.
        """

        self.model = model

        self.config = config
        self.dataset_config = config.dataset_config
        self.model_config = model.model_config
        self.train_config = config.train_config

        # Parse model config
        self.model_name = self.model_config.model_type

        # Parse train config
        self.paths_config = config.train_config.paths_config
        self.checkpoint_dir = self.paths_config.checkpoint_dir

        self.eval_mode = eval_mode
        if eval_mode not in ['val', 'test']:
            raise ValueError('Evaluation mode can only be set to `val` or `test`')

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        self.skip_evaluated_checkpoints = skip_evaluated_checkpoints
        self.eval_wait_interval = eval_wait_interval

        self.do_kitti_native_eval = do_kitti_native_eval
        if self.do_kitti_native_eval:
            if self.eval_mode == 'test':
                raise ValueError('Cannot run native eval in test mode.')

            # Compile native evaluation if not compiled yet
            evaluator_utils.compile_kitti_native_code()

        # Whether to run the native eval in parallel processes
        self.kitti_native_eval_parallel = True

        # Set up output predictions folders
        self.predictions_base_dir = self.train_config.paths_config.pred_dir
        os.makedirs(self.predictions_base_dir, exist_ok=True)

        # Path to text file that keeps track of evaluated checkpoints
        self.already_evaluated_path = self.predictions_base_dir + '/evaluated_{}.txt'.format(
            self.dataset_config.data_split)

        self.output_types = MonoPSROutputBuilder.get_output_types_list(
            self.model_config.output_config)

        # Create a variable tensor to hold the global step
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        # Setup session
        allow_gpu_mem_growth = config.allow_gpu_mem_growth
        if allow_gpu_mem_growth:
            # GPU memory config
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = allow_gpu_mem_growth
            self._sess = tf.Session(config=sess_config)
        else:
            self._sess = tf.Session()

        # The model should return a dictionary of predictions
        self.output_dict, self.gt_dict, self.output_debug_dict = self.model.build()

        if eval_mode == 'val':
            # Setup loss and summary writer in val mode only
            self.losses_dict, self.total_loss_tensor = \
                self.model.loss(self.output_dict, self.gt_dict)

            # Setup graph nodes for evaluation
            self.metrics_dict, self.metrics_debug_dict = \
                self.model.evaluate_predictions(self.output_dict, self.gt_dict)

            self.summary_writer, self.summary_merged = \
                evaluator_utils.set_up_summary_writer(
                    self.train_config, self.dataset_config.data_split, self._sess)

        else:  # eval_mode == 'test'
            self.losses_dict = None
            self.total_loss_tensor = None
            self.summary_writer = None
            self.summary_merged = None

        self._saver = tf.train.Saver()

        # Add maximum memory usage summary op
        # This op can only be run on device with gpu so it's skipped on Travis
        if 'TRAVIS' not in os.environ:
            # tf 1.4
            tf.summary.scalar('bytes_in_use',
                              tf.contrib.memory_stats.BytesInUse())
            tf.summary.scalar('max_bytes',
                              tf.contrib.memory_stats.MaxBytesInUse())

    def run_checkpoint_once(self, checkpoint_to_restore):
        """Evaluates network metrics once over all the validation samples.

        Args:
            checkpoint_to_restore: The path of the checkpoint to restore,
                including the checkpoint step.
        """

        self._saver.restore(self._sess, checkpoint_to_restore)

        data_split = self.dataset_config.data_split
        num_samples = self.model.dataset.num_samples

        global_step = checkpoint_utils.get_global_step(self._sess, self.global_step_tensor)

        # Create or clear dict to track average losses and metrics
        if self.eval_mode == 'val':
            eval_losses_avg_dict = dict.fromkeys(self.losses_dict.keys(), 0.0)
            # Separate list per key
            metrics_list_dict = {key: [] for key in self.metrics_dict.keys()}

            # Metrics output dir
            metrics_dir = os.path.join(self.predictions_base_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)

        num_valid_samples = 0

        # Keep track of feed_dict and inference time
        total_feed_dict_time = []
        total_inference_time = []

        # Create output folders for this checkpoint step
        output_dirs = {}
        if constants.KEY_BOX_2D in self.output_types:
            proposal_dir = self.predictions_base_dir + '/proposals_{}/{}/{}'.format(
                self.output_types, data_split, global_step)
            os.makedirs(proposal_dir, exist_ok=True)
            output_dirs.update({constants.OUT_DIR_PROPS: proposal_dir})

            pred_box_2d_dir = self.predictions_base_dir + '/predictions_{}/{}/{}'.format(
                constants.REP_BOX_3D, data_split, global_step)
            os.makedirs(pred_box_2d_dir, exist_ok=True)
            output_dirs.update({constants.OUT_DIR_BOX_2D: pred_box_2d_dir})

            pred_masks_dir = self.predictions_base_dir + '/masks/{}/{}'.format(data_split,
                                                                               global_step)
            os.makedirs(pred_masks_dir, exist_ok=True)
            output_dirs.update({constants.OUT_DIR_MASKS: pred_masks_dir})

        # TODO: Put box_3d back into config
        if constants.KEY_CENTROIDS in self.output_types:
            pred_box_3d_dir = self.predictions_base_dir + '/predictions_{}/{}/{}'.format(
                constants.KEY_BOX_3D, data_split, global_step)
            os.makedirs(pred_box_3d_dir, exist_ok=True)
            output_dirs.update({constants.OUT_DIR_BOX_3D: pred_box_3d_dir})

            pred_box_2d_dir = self.predictions_base_dir + '/predictions_{}/{}/{}'.format(
                constants.KEY_BOX_2D, data_split, global_step)
            os.makedirs(pred_box_2d_dir, exist_ok=True)
            output_dirs.update({constants.OUT_DIR_BOX_2D: pred_box_2d_dir})

        if constants.KEY_INST_XYZ_MAP_LOCAL in self.output_types:
            pred_xyz_map_dir = self.predictions_base_dir + '/predictions_{}/{}/{}'.format(
                constants.KEY_INST_XYZ_MAP_LOCAL, data_split, global_step)
            os.makedirs(pred_xyz_map_dir, exist_ok=True)
            output_dirs.update({constants.OUT_DIR_XYZ_MAP_LOCAL: pred_xyz_map_dir})

        # Run through a single epoch
        current_epoch = self.model.dataset.epochs_completed
        while current_epoch == self.model.dataset.epochs_completed:

            # Keep track of feed_dict speed
            start_time = time.time()
            feed_dict, sample_dict = self.model.create_feed_dict()
            feed_dict_time = time.time() - start_time

            # Run inference
            if self.eval_mode == 'val':

                inference_start_time = time.time()
                # Do predictions, loss calculations, and summaries
                if self.summary_merged is not None:

                    # if sample_dict['sample_name'] == '000050':
                    #     output_debug = self._sess.run(self.output_debug_dict, feed_dict)
                    #
                    #     cam_p = sample_dict['sample_cam_p']
                    #     valid_mask_map_0 = output_debug['valid_mask_maps'][0]
                    #     inst_xyz_map_local_0 = output_debug['pred_inst_xyz_maps_local'][0]
                    #     inst_xyz_maps_global_0 = output_debug['pred_inst_xyz_maps_global'][0]
                    #     inst_pc_global_0 = output_debug['pred_inst_pc_global'][0]
                    #
                    #     valid_mask_flat = valid_mask_map_0.reshape(2304, 1)
                    #     valid_pc_mask = valid_mask_map_0.reshape(1, 2304)
                    #
                    #     proj_err_map_0 = output_debug['proj_err_map'][0]
                    #     gt_view_ang_0 = output_debug['gt_view_angs'][0][0]
                    #     proj_pred_cen_0 = output_debug['proj_pred_cen'][0]
                    #
                    #     proj_uv = calib_utils.project_pc_to_image(inst_pc_global_0, cam_p)
                    #     test_inst_points_global = instance_utils.inst_points_local_to_global(
                    #         inst_xyz_map_local_0.reshape(-1, 3), gt_view_ang_0, proj_pred_cen_0)
                    #     test_proj_uv = calib_utils.project_pc_to_image(test_inst_points_global.T, cam_p)
                    #
                    #
                    #     print('done')

                    # DEBUG
                    # output_debug = self._sess.run(self.output_debug_dict, feed_dict)
                    # metrics_debug = self._sess.run(self.metrics_debug_dict, feed_dict)
                    # gt_out = self._sess.run(self.gt_dict, feed_dict)

                    # DEBUG: Show projection error maps
                    # output_debug = self._sess.run(self.output_debug_dict, feed_dict)
                    # import cv2
                    # proj_err_map_u = output_debug['proj_err_map_norm'][0, :, :, 0]
                    # proj_err_map_v = output_debug['proj_err_map_norm'][0, :, :, 1]
                    # cv2.imshow('proj_err_u', abs(proj_err_map_u) / np.amax(abs(proj_err_map_u)))
                    # cv2.imshow('proj_err_v', abs(proj_err_map_v) / np.amax(abs(proj_err_map_v)))
                    # cv2.waitKey(0)

                    # Run network
                    predictions, eval_losses, metrics_dict_out = \
                        self._sess.run([self.output_dict,
                                        self.losses_dict,
                                        self.metrics_dict],
                                       feed_dict=feed_dict)

                    # Update averages for losses
                    for key in eval_losses.keys():
                        eval_losses_avg_dict.update({
                            key: eval_losses_avg_dict[key] + eval_losses[key]})

                    # Update averages for metrics
                    for key in metrics_dict_out.keys():

                        metric_val = metrics_dict_out[key]

                        # Check for NaN values
                        # TODO: Figure better way to ignore NaN
                        if np.isnan(metric_val).any():
                            continue

                        # Convert to list and extend list of values
                        metric_val_as_list = np.reshape(metric_val, (-1))
                        metrics_list_dict[key].extend(metric_val_as_list)

                else:  # eval_mode == 'test'
                    predictions, eval_losses, metrics_dict_out = \
                        self._sess.run([self.output_dict,
                                        self.losses_dict,
                                        self.metrics_dict],
                                       feed_dict=feed_dict)
                inference_time = time.time() - inference_start_time

                num_valid_samples += 1

                # Get sample name from model
                sample_name = sample_dict[constants.SAMPLE_NAME]

                # Use model to save predictions
                self.model.save_predictions(sample_name, predictions, sample_dict, output_dirs)

                sys.stdout.write(
                    '\r{}: Step {}: {} / {}, Sample: {}, '
                    'Feed: {:0.4f}, Inf: {:0.4f}, Total: {:0.4f}'.format(
                        self.config.config_name,
                        global_step, num_valid_samples, num_samples, sample_name,
                        feed_dict_time, inference_time, time.time() - start_time))
                sys.stdout.flush()

            else:  # self.eval_mode == 'test'
                # Run inference, don't calculate loss or run summaries for test
                inference_start_time = time.time()
                predictions = self._sess.run(self.output_dict, feed_dict=feed_dict)
                inference_time = time.time() - inference_start_time

                # Get sample name from model
                sample_name = sample_dict[constants.SAMPLE_NAME]

                # Use model to save predictions
                self.model.save_predictions(sample_name, predictions, sample_dict, output_dirs)

                num_valid_samples += 1

                sys.stdout.write(
                    '\r{}: Inference: {} / {}, Sample: {}, '
                    'Inf: {:0.4f}, Total: {:0.4f}'.format(
                        self.config.config_name, num_valid_samples, num_samples, sample_name,
                        inference_time, time.time() - start_time))
                sys.stdout.flush()

                # Add times to list
                total_feed_dict_time.append(feed_dict_time)
                total_inference_time.append(inference_time)

        # end while current_epoch == model.dataset.epochs_completed

        # After epoch is complete
        if self.eval_mode == 'val':

            # Average losses over number of samples
            for key in eval_losses_avg_dict.keys():
                average_loss = eval_losses_avg_dict[key] / num_valid_samples
                summary_utils.add_scalar_summary(
                    'losses/' + key, average_loss, self.summary_writer, global_step)

            # Save metrics
            if len(metrics_list_dict) > 0:
                # Save average metrics
                checkpoint_name = self.config.config_name
                evaluator_utils.save_metrics(
                    checkpoint_name, data_split, global_step,
                    metrics_list_dict, self.model_config, self.summary_writer)

            # Convert and evaluate predictions
            if constants.KEY_BOX_2D in self.output_types:
                # Convert predictions to KITTI format and run native evaluation
                evaluator_utils.save_predictions_box_2d_in_kitti_format(
                    self.train_config.kitti_score_threshold, self.model.dataset,
                    self.predictions_base_dir, pred_box_2d_dir, global_step)
                self.run_kitti_native_eval(global_step)

            if constants.KEY_CENTROIDS in self.output_types:
                # Convert predictions to KITTI format and run native evaluation
                evaluator_utils.save_predictions_box_3d_in_kitti_format(
                    self.train_config.kitti_score_threshold, self.model.dataset,
                    self.predictions_base_dir, pred_box_3d_dir, pred_box_2d_dir, global_step)
                self.run_kitti_native_eval(global_step)

            # Save list of already evaluated checkpoints
            with open(self.already_evaluated_path, 'ba') as f:
                np.savetxt(f, [global_step], fmt='%d')

            self.summary_writer.flush()

        else:  # self.eval_mode == 'test'

            # Run native KITTI eval if there are labels
            if self.model.dataset.has_kitti_labels:
                # Convert predictions to KITTI format and run native evaluation
                evaluator_utils.save_predictions_box_3d_in_kitti_format(
                    self.train_config.kitti_score_threshold, self.model.dataset,
                    self.predictions_base_dir, pred_box_3d_dir, pred_box_2d_dir, global_step)
                self.run_kitti_native_eval(global_step)

            evaluator_utils.print_inference_time_statistics(
                total_feed_dict_time, total_inference_time)

        print('\nStep {}: Finished evaluation'.format(global_step))

    def run_latest_checkpoints(self, ckpt_indices):
        """Evaluation function for evaluating all the existing checkpoints, or
        evaluates selected checkpoints.

        Args:
            ckpt_indices: checkpoint indices to evaluate

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        # Load the latest checkpoints available
        last_checkpoints = checkpoint_utils.load_checkpoints(self.checkpoint_dir, self._saver)

        # Create dictionary of the checkpoint paths. The key is the 8 digit checkpoint index.
        checkpoint_path_dict = {path[-8:]: path for path in last_checkpoints}

        num_checkpoints = len(self._saver.last_checkpoints)

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts()
        ckpt_indices = np.asarray([ckpt_indices])
        if ckpt_indices is not None:
            if ckpt_indices[0] == -1:
                # Restore the most recent checkpoint
                ckpt_idx = num_checkpoints - 1
                ckpt_indices = [ckpt_idx]
            for ckpt_idx in ckpt_indices:
                ckpt_idx_padded = str(ckpt_idx).rjust(8, '0')
                checkpoint_to_restore = checkpoint_path_dict[ckpt_idx_padded]
                self.run_checkpoint_once(checkpoint_to_restore)

        else:
            last_checkpoint_id = -1
            number_of_evaluations = 0
            # go through all existing checkpoints
            for ckpt_idx in range(num_checkpoints):
                checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                ckpt_id = checkpoint_utils.split_checkpoint_step(checkpoint_to_restore)

                # Check if checkpoint has been evaluated already
                already_evaluated = ckpt_id in already_evaluated_ckpts
                if already_evaluated or ckpt_id <= last_checkpoint_id:
                    number_of_evaluations = max((ckpt_idx + 1,
                                                 number_of_evaluations))
                    continue

                self.run_checkpoint_once(checkpoint_to_restore)
                number_of_evaluations += 1

                # Save the id of the latest evaluated checkpoint
                last_checkpoint_id = ckpt_id

    def repeated_checkpoint_run(self):
        """Periodically evaluates the checkpoints inside the `checkpoint_dir`.

        This function evaluates all the existing checkpoints as being generated.
        If there is none, it sleeps until new checkpoints become available.
        Since there is no synchronization guarantees for the trainer and
        evaluator, at each iteration it reloads all the checkpoints and searches
        for the last checkpoint to continue from. This is meant to be called in
        parallel to the trainer to evaluate the models regularly.

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts()

        print('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))

        last_checkpoint_step = -1
        number_of_evaluations = 0

        max_checkpoint_step = self.train_config.max_iterations
        while last_checkpoint_step < max_checkpoint_step:
            # Load current checkpoints available
            checkpoint_utils.load_checkpoints(self.checkpoint_dir, self._saver)
            num_checkpoints = len(self._saver.last_checkpoints)

            start = time.time()

            if number_of_evaluations >= num_checkpoints:
                print('No new checkpoints found in {}. Will try again in {} seconds'.format(
                      self.checkpoint_dir, self.eval_wait_interval))

                time_to_next_eval = start + self.eval_wait_interval - time.time()
                if time_to_next_eval > 0:
                    time.sleep(time_to_next_eval)

            else:
                for ckpt_idx in range(num_checkpoints):
                    checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                    ckpt_step = checkpoint_utils.split_checkpoint_step(checkpoint_to_restore)

                    # Check if checkpoint has been evaluated already
                    already_evaluated = ckpt_step in already_evaluated_ckpts
                    if already_evaluated or ckpt_step <= last_checkpoint_step:
                        number_of_evaluations = max((ckpt_idx + 1, number_of_evaluations))
                        continue

                    self.run_checkpoint_once(checkpoint_to_restore)
                    number_of_evaluations += 1

                    # Save the id of the latest evaluated checkpoint
                    last_checkpoint_step = ckpt_step

        print('All checkpoints evaluated, exiting.')

    def get_evaluated_ckpts(self):
        """Finds the evaluated checkpoints.

        Examines the evaluation average losses file to find the already
        evaluated checkpoints.

        Args:

        Returns:
            already_evaluated_ckpts: A list of checkpoint indices, or an
                empty list if no evaluated indices are found.
        """

        if os.path.exists(self.already_evaluated_path):
            already_evaluated_ckpts = np.loadtxt(self.already_evaluated_path, delimiter=',')
            already_evaluated_ckpts = already_evaluated_ckpts.reshape(-1).astype(np.int32)

            return already_evaluated_ckpts

        return []

    def run_kitti_native_eval(self, global_step):
        """Calls the kitti native C++ evaluation code.

        It first saves the predictions in kitti format. It then creates two
        child processes to run the evaluation code. The native evaluation
        hard-codes the IoU threshold inside the code, so hence its called
        twice for each IoU separately.

        Args:
            global_step: Global step of the current checkpoint to be evaluated.
        """

        # Kitti native evaluation, do this during validation
        checkpoint_name = self.config.config_name
        kitti_score_threshold = self.train_config.kitti_score_threshold
        data_split = self.dataset_config.data_split

        # Don't run last native evaluation in parallel
        if global_step == self.train_config.max_iterations or not self.kitti_native_eval_parallel:
            evaluator_utils.run_kitti_native_script(
                checkpoint_name, data_split, kitti_score_threshold, global_step)
            evaluator_utils.run_kitti_native_script_with_low_iou(
                checkpoint_name, data_split, kitti_score_threshold, global_step)

        else:
            # Create separate processes to run the native evaluation
            native_eval_proc = Process(
                target=evaluator_utils.run_kitti_native_script,
                args=(checkpoint_name, data_split, kitti_score_threshold, global_step))
            native_eval_proc_low_iou = Process(
                target=evaluator_utils.run_kitti_native_script_with_low_iou,
                args=(checkpoint_name, data_split, kitti_score_threshold, global_step))

            # Don't call join on this cuz we do not want to block
            # this will cause one zombie process - should be fixed later.
            native_eval_proc.start()
            native_eval_proc_low_iou.start()
