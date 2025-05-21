import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils

from skimage.filters import threshold_otsu
import cv2

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator:

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        print(f"Number of classes: {self.n_class}")
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device(
            "cuda:%s" % args.gpu_ids[0]
            if torch.cuda.is_available() and len(args.gpu_ids) > 0
            else "cpu"
        )
        print(self.device)
        self.model_str = args.net_G

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, "log_test.txt")
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, checkpoint_name="best_ckpt.pt"):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write("loading last checkpoint...\n")
            # load the entire checkpoint
            checkpoint = torch.load(
                os.path.join(self.checkpoint_dir, checkpoint_name),
                map_location=self.device,
            )

            self.net_G.load_state_dict(checkpoint["model_G_state_dict"])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint["best_val_acc"]
            self.best_epoch_id = checkpoint["best_epoch_id"]

            self.logger.write(
                "Eval Historical_best_acc = %.4f (at epoch %d)\n"
                % (self.best_val_acc, self.best_epoch_id)
            )
            self.logger.write("\n")

        else:
            raise FileNotFoundError("no such checkpoint %s" % checkpoint_name)

    def _visualize_pred(self):
        """
        Convert the prediction tensor into a grid visualization where each class is represented by a unique color.
        """
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)  # Shape: [8, 1, 512, 512]
        pred = pred.squeeze(
            1
        )  # Remove the channel dimension, resulting in shape [8, 512, 512]

        # Define a color palette (5 classes)
        palette = [
            [0, 0, 0],  # Black for class 0 (background)
            [0, 255, 0],  # Green for class 1
            [255, 255, 0],  # Yellow for class 2
            [255, 0, 0],  # Red for class 3
            [255, 165, 0],  # Orange for class 4
        ]

        # Map class indices to colors
        batch_size, height, width = pred.shape
        pred_vis = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8)

        for class_idx, color in enumerate(palette):
            mask = pred == class_idx  # Get mask for the current class
            pred_vis[mask] = torch.tensor(color, dtype=torch.uint8)

        # Combine the batch into a grid of 4 rows × 2 columns
        grid_vis = utils.make_numpy_grid(
            pred_vis.permute(0, 3, 1, 2)
        )  # Use permute to make it [B, C, H, W]
        return grid_vis  # Combined grid visualization

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch["L"].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(
            pr=G_pred.cpu().numpy(), gt=target.cpu().numpy()
        )
        return current_score

    def _visualize_gt(self, tensor_data):
        """
        Convert the ground truth tensor into an RGB visualization where each class is represented by a unique color.
        """
        tensor_data = tensor_data.squeeze(
            1
        )  # Remove the channel dimension, resulting in shape [B, H, W]

        # Custom color palette for ground truth classes
        colors = {
            0: [0, 0, 0],  # Black for class 0 (background)
            1: [0, 255, 0],  # Green for class 1
            2: [255, 255, 0],  # Yellow for class 2
            3: [255, 0, 0],  # Red for class 3
            4: [255, 165, 0],  # Orange for class 4
        }

        # Create an empty tensor for visualization
        batch_size, height, width = tensor_data.shape
        gt_vis = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8)

        # Map each class index to its corresponding color
        for class_idx, color in colors.items():
            mask = tensor_data == class_idx  # Get mask for the current class
            gt_vis[mask] = torch.tensor(color, dtype=torch.uint8)

        # Combine the batch into a grid of 4 rows × 2 columns
        grid_vis = utils.make_numpy_grid(
            gt_vis.permute(0, 3, 1, 2)
        )  # Permute to [B, C, H, W] for grid creation
        return grid_vis  # Return combined grid visualization

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = "Is_training: %s. [%d,%d],  running_mf1: %.5f\n" % (
                self.is_training,
                self.batch_id,
                m,
                running_acc,
            )
            self.logger.write(message)

        vis_input = utils.make_numpy_grid(de_norm(self.batch["A"]))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch["B"]))

        # Generate prediction grid visualization
        vis_pred = self._visualize_pred()  # Shape: [2048, 4096, 3]

        # Generate ground truth visualization (apply color mapping)
        vis_gt = self._visualize_gt(self.batch["L"])  # Shape: [2048, 4096, 3]

        # Combine all visualizations
        vis = np.concatenate(
            [vis_input, vis_input2, vis_pred, vis_gt], axis=0
        )  # Combine vertically
        vis = np.clip(vis, a_min=0.0, a_max=1.0)

        # Save the visualization
        file_name = os.path.join(self.vis_dir, "eval_" + str(self.batch_id) + ".jpg")
        plt.imsave(file_name, vis)

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        # np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict["mf1"]

        with open(
            os.path.join(self.checkpoint_dir, "%s.txt" % (self.epoch_acc)), mode="a"
        ) as file:
            pass

        message = ""
        for k, v in scores_dict.items():
            message += "%s: %.5f " % (k, v)
        self.logger.write("%s\n" % message)  # save the message

        self.logger.write("\n")

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch["A"].to(self.device)
        img_in2 = batch["B"].to(self.device)

        if self.model_str == "changeFormerV6":
            self.G_pred = self.net_G(img_in1, img_in2)[-1]
        else:
            self.G_pred = self.net_G(img_in1, img_in2)

    def eval_models(self, checkpoint_name="best_ckpt.pt"):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write("Begin evaluation...\n")
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
