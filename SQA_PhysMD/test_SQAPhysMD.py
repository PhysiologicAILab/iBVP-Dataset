import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from copy import deepcopy

import torch
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

from model.SQAPhysMD import Model as sqPPG


class sqaPPGInference(object):
    """
        The class to infer signal quality for BVP signal
    """

    def __init__(self, config, preprocess=True) -> None:

        # Get cpu, gpu or mps device for inference.
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")
        self.device = torch.device(device)

        if os.path.exists(config):
            with open(config) as json_file:
                self.config = json.load(json_file)
            json_file.close()
        else:
            print("Config file does not exists", config)
            print("Exiting the code.")
            exit()

        self.target_fs = self.config["data"]["target_fs"]
        self.win_len_sec = self.config["data"]["window_len_sec"]
        self.win_samples = int((self.win_len_sec) * self.target_fs)

        self.sq_resolution = self.config["data"]["sq_resolution_sec"]
        self.total_secs = float(self.config["data"]["total_duration_sec"])
        self.target_samples = int(np.round(self.total_secs * self.target_fs))

        self.sqPPG_model = sqPPG(self.device, self.config).to(self.device)
        ckpt_path = os.path.join("SQA_PhysMD", "ckpt", self.config["ckpt_name"])
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.sqPPG_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No checkpoint found, existing...")
            # exit()
            return -1
        self.preprocess = preprocess
        if self.preprocess:
            order = 2
            freqs_bp_ppg = (0.5, 2.5)
            # freqs_bp_ppg = 2 * np.array(freqs_bp_ppg) / self.target_fs # Normalize frequency to Nyquist Frequency (Fs/2).
            self.sos_bp_ppg = signal.butter(N=order, Wn=freqs_bp_ppg, btype='bandpass', fs=self.target_fs, output='sos')
            # self.b_phys_ppg, self.a_phys_ppg = signal.butter(N=order, Wn=freqs_bp_ppg, btype="bandpass", fs=self.target_fs, analog=False)


    def run_inference(self, org_bvp_vec):
        
        self.sqPPG_model.eval()
        with torch.no_grad():
            bvp_vec = deepcopy(org_bvp_vec)
            if self.preprocess:
                org_total_samples = bvp_vec.shape[0]
                Told = np.linspace(1, self.target_samples, org_total_samples)
                F = interp1d(Told, bvp_vec[:, 0], kind='cubic')
                Tnew = np.linspace(1, self.target_samples, self.target_samples)
                bvp_vec = F(Tnew)
                bvp_vec = np.expand_dims(bvp_vec, 1)
                # bvp_vec = signal.resample(bvp_vec, self.target_samples)

            bvp_vec = bvp_vec.transpose(1, 0)
            sig_segs = np.arange(0, self.target_samples, self.win_samples)
            sqa_vec_array = np.empty((1, 0))
            bvp_vec_array = np.empty((1, 0))
            
            for st_indx in sig_segs:
                end_indx = st_indx + self.win_samples
                bvp_seg = deepcopy(bvp_vec)
                bvp_seg = bvp_seg[:, st_indx: end_indx]

                if self.preprocess:
                    bvp_seg = signal.sosfiltfilt(self.sos_bp_ppg, bvp_seg) # bvp_seg needs to be filtered in the same range
                    # bvp_seg = signal.filtfilt(self.b_phys_ppg, self.a_phys_ppg, bvp_seg)

                    min_ppg_val = np.min(bvp_seg)
                    max_ppg_val = np.max(bvp_seg)
                    bvp_seg = (bvp_seg - min_ppg_val) / (max_ppg_val - min_ppg_val)

                input_vec = torch.tensor(bvp_seg, dtype=torch.float)
                input_vec = input_vec.unsqueeze(1)
                input_vec = input_vec.to(self.device)

                sqa_vec = self.sqPPG_model(input_vec)
                sqa_vec = sqa_vec.cpu().numpy().squeeze(1)
                sqa_vec = 1 - sqa_vec
                # print("sqa_vec.shape", sqa_vec.shape)
                # exit()
                sqa_vec_array = np.append(sqa_vec_array, sqa_vec, axis=1)
                bvp_vec_array = np.append(bvp_vec_array, bvp_seg, axis=1)
                
            return bvp_vec_array, sqa_vec_array


def main(args_parser):
    preprocess = False if args_parser.prep == "0" else 1
    testObj = sqaPPGInference(args_parser.config, preprocess)

    if os.path.exists(args_parser.datadir):
        # files_list = os.listdir(args_parser.datadir)
        files_list = list(Path(args_parser.datadir).glob("*.csv"))
    else:
        print("Specified data directory not found:", args_parser.datadir)
        exit()

    if not os.path.exists(args_parser.savedir):
        try:
            os.makedirs(args_parser.savedir)
            print("Directory created for saving plots and inference:", args_parser.savedir)
        except:
            print("Directory creation failed - for saving outputs. Please check the permissions")
            exit()

    plot_dir = os.path.join(args_parser.savedir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    sq_inference_dir = os.path.join(args_parser.savedir, "sq_vec")
    if not os.path.exists(sq_inference_dir):
        os.makedirs(sq_inference_dir)

    for fn in files_list:
        print("Processing:", fn.name)
        plot_fn = os.path.join(plot_dir, fn.name.replace(".csv", ".jpg"))
        sq_vec_fn = os.path.join(sq_inference_dir, fn.name)
        org_bvp_vec = pd.read_csv(str(fn)).to_numpy()[:,0]
        org_bvp_vec = np.expand_dims(org_bvp_vec, 1)

        bvp_vec, sqa_vec = testObj.run_inference(org_bvp_vec)

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(bvp_vec.T)
        ax[0].set_ylim((-0.2, 1.2))
        ax[1].plot(sqa_vec.T)
        ax[1].set_ylim((-0.2, 1.2))
        plt.suptitle("BVP Signal and Signal Quality Metrics")
        # plt.show()
        plt.savefig(plot_fn)
        plt.close(fig)
        sq_df = pd.DataFrame(sqa_vec.T)
        sq_df.to_csv(sq_vec_fn)
        # break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, dest='config',
                        help='Config file for model')
    parser.add_argument('--datadir', type=str, dest='datadir',
                        help='Directory with PPG signals (.csv files)')
    parser.add_argument('--savedir', type=str, dest='savedir',
                        help='Directory to save signal quality inference and generated plots')
    parser.add_argument('--preprocess', type=str, dest='prep', default=1,
                        help='Whether to preprocess the signals; 0/[1]')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    main(args_parser)
