import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        self.config = config
        self.save_dir = save_dir
        # Split the flat YAML config into the 3 sub-dicts used by 3DGS internals
        # (gaussian model I/O, optimizer hyperparams, rasterizer pipeline flags).
        # munchify() lets them be accessed as attributes (e.g. opt_params.iterations).
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        # live_mode: reading frames from a live RealSense camera instead of a dataset folder.
        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        # monocular: True for RGB-only input (no depth sensor available).
        # This flag changes both the tracking/mapping loss and the keyframe/init logic.
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            # live demo always needs the viewer to monitor tracking quality
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        # sh_degree: 0 = only the DC (flat color) term, 3 = full view-dependent color.
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        # The Gaussian map. Handed to the backend (which owns and optimizes it in
        # its own process) and to the GUI. The frontend never touches this
        # object: it tracks against detached snapshots that the backend sends
        # back through frontend_queue (see FrontEnd.sync_backend).
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)  # spatial_lr_scale, scales the xyz/scaling learning rates
        self.dataset = load_dataset(model_params, model_params.source_path, config=config)

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]  # black background used during rasterization
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Inter-process message queues: frontend <-> backend.
        # frontend_queue carries backend->frontend updates (new gaussians/poses).
        # backend_queue carries frontend->backend requests (init/keyframe/pause/stop).
        # Stored on self so run() (called separately, after __init__ returns) can
        # still reach them.
        frontend_queue = self.frontend_queue = mp.Queue()
        backend_queue = self.backend_queue = mp.Queue()

        # GUI queues; replaced by a no-op FakeQueue when the GUI is disabled so
        # frontend/backend code can push to them unconditionally without branching.
        q_main2vis = self.q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = self.q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        sync_mode = self.config["Training"].get("sync_mode")
        if sync_mode not in ("parallel", "hybrid", "sequential"):
            raise ValueError(
                f"Training.sync_mode must be 'parallel', 'hybrid' or 'sequential', got {sync_mode!r}"
            )

        # Frontend = tracking (runs in the main process via frontend.run() below).
        # Backend  = mapping/BA (runs in its own mp.Process, started further down).
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        # cameras_extent: rough scene radius, used to scale densify/prune distance
        # thresholds (init_gaussian_extent, gaussian_extent) in set_hyperparams().
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        # Bundle of everything the GUI process needs; passed once at process start
        # since GUI runs in a separate process and can't share Python objects directly.
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

    def run(self):
        # CUDA events used to time the whole run (start..end) for the FPS report below.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        frontend_queue = self.frontend_queue
        backend_queue = self.backend_queue
        q_main2vis = self.q_main2vis

        start.record()

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)  # give the GUI window/OpenGL context time to initialize

        backend_process.start()
        # Blocking call: drives the whole tracking loop over the dataset in this
        # (main) process until every frame has been consumed.
        self.frontend.run()
        # Dataset exhausted: tell the backend to stop its background mapping loop.
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            # Snapshot the map/trajectory as produced live by tracking, before any
            # extra offline refinement, to measure "online" SLAM quality.
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            # Ask backend to run offline "color refinement" (26k iters, standard 3DGS
            # photometric optimization with poses frozen) to get the "after" metrics.
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        # Clean shutdown: stop the backend process, then the GUI process (if any).
        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override Dataset.dataset_path from the config file",
    )

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    if args.dataset_path is not None:
        config["Dataset"]["dataset_path"] = args.dataset_path
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        dataset_path = os.path.normpath(config["Dataset"]["dataset_path"])
        path = dataset_path.split(os.sep)
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-2] + "_" + path[-1], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
