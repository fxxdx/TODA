# coding=utf-8
import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append('../ADATIME')
import pandas as pd

import collections
import argparse
import warnings
import sklearn.exceptions

from trainers.utils import fix_randomness, starting_logs, AverageMeter

from abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, '{}'.format(self.run_description))
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def train(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:    #首先获取源域和目标域，也就是它们的标号
            for run_id in range(self.num_runs):    #实验结果跑num_runs次，最终性能指标取平均值
                # fixing random seed
                fix_randomness(self.seed)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                # Average meters
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data  根据源域和目标域标号加载数据
                self.load_data(src_id, trg_id)

                # Train model
                non_adapted_model, last_adapted_model, best_adapted_model = self.train_model()

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model, last_adapted_model,
                                     best_adapted_model)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = '{}_to_{}'.format(src_id,trg_id)

                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)
                print('{}_to_{}:'.format(src_id,trg_id))
                print(table_results)
        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs', type=str,
                        help='Directory containing all experiments')
    parser.add_argument('--run_description', default='MAPU_new', type=str, help='Description of run, if none, DA method name will be used')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='TODA', type=str, help='TODA, SHOT, AaD, NRC, MAPU,')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'../data', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset of choice: (EEG - HAR - MFD)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--seed', default=6, type=int, help='SEED value')
    parser.add_argument('--splits', default=8, type=int, help='num_splits value')
    parser.add_argument('--TOV', default=0.5, type=float, help='TOV weight value')
    parser.add_argument('--CON', default=1.0, type=float, help='CON1 weight value')
    parser.add_argument('--CON2', default=1.0, type=float, help='CON2 weight value')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
