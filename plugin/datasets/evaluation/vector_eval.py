from functools import partial
import numpy as np
from multiprocessing import Pool
from mmdet3d.datasets import build_dataset, build_dataloader
import mmcv
from .AP import instance_match, average_precision
import prettytable
from time import time
from functools import cached_property
from shapely.geometry import LineString, box
from numpy.typing import NDArray
from typing import Dict, List, Optional
from logging import Logger
from mmcv import Config
from copy import deepcopy
import os
from IPython import embed

INTERP_NUM = 200 # number of points to interpolate during evaluation
THRESHOLDS = [0.5, 1.0, 1.5] # AP thresholds
N_WORKERS = 16 # num workers to parallel
SAMPLE_DIST = 0.15


def _limit_worker_threads():
    ''' One thread per eval worker: with N_WORKERS processes each running torch/BLAS
    on all cores, the machine is oversubscribed and evaluation slows down a lot. '''
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)

class VectorEvaluate(object):
    """Evaluator for vectorized map.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
        eval_regions (bool): also report A-mAP / R-mAP on the ahead / rear halves
    """

    def __init__(self, dataset_cfg: Config, n_workers: int=N_WORKERS, eval_regions: bool=False) -> None:
        self.dataset = build_dataset(dataset_cfg)
        self.eval_regions = eval_regions
        self.cat2id = self.dataset.cat2id
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        self.n_workers = n_workers
        self.new_split = 'newsplit' in self.dataset.ann_file
        self.roi_size = self.dataset.roi_size
        if self.roi_size == (60, 30):
            self.thresholds = [0.5, 1.0, 1.5]
        elif self.roi_size == (100, 50):
            self.thresholds = [1.0, 1.5, 2.0]
        
    @cached_property
    def gts(self) -> Dict[str, Dict[int, List[NDArray]]]:
        roi_size = self.dataset.roi_size
        if 'av2' in self.dataset.ann_file:
            dataset = 'av2'
        else:
            dataset = 'nusc'
        # cache gts next to the annotation file, so different annotation sets don't share a cache
        tmp_dir = os.path.dirname(self.dataset.ann_file)
        if self.new_split:
            tmp_file = os.path.join(tmp_dir, f'tmp_gts_{dataset}_{roi_size[0]}x{roi_size[1]}_newsplit.pkl')
        else:
            tmp_file = os.path.join(tmp_dir, f'tmp_gts_{dataset}_{roi_size[0]}x{roi_size[1]}.pkl')
        if os.path.exists(tmp_file):
            print(f'loading cached gts from {tmp_file}')
            gts = mmcv.load(tmp_file)
            return gts
        
        print('collecting gts...')
        gts = {}
        self.dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=self.n_workers, shuffle=False, dist=False)
        pbar = mmcv.ProgressBar(len(self.dataloader))
        for data in self.dataloader:
            token = deepcopy(data['img_metas'].data[0][0]['token'])
            gt = deepcopy(data['vectors'].data[0][0])
            gts[token] = gt
            pbar.update()
            del data # avoid dataloader memory crash
        
        if not os.path.exists(tmp_file):
            print(f"saving gt to {tmp_file}")
            mmcv.dump(gts, tmp_file)
        return gts
    
    def interp_fixed_num(self, 
                         vector: NDArray, 
                         num_pts: int) -> NDArray:
        ''' Interpolate a polyline.
        
        Args:
            vector (array): line coordinates, shape (M, 2)
            num_pts (int): 
        
        Returns:
            sampled_points (array): interpolated coordinates
        '''
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_pts)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()
        
        return sampled_points
    
    def interp_fixed_dist(self, 
                          vector: NDArray,
                          sample_dist: float) -> NDArray:
        ''' Interpolate a line at fixed interval.
        
        Args:
            vector (LineString): vector
            sample_dist (float): sample interval
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''
        line = LineString(vector)
        distances = list(np.arange(sample_dist, line.length, sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points

    def _evaluate_single(self, 
                         pred_vectors: List, 
                         scores: List, 
                         groundtruth: List, 
                         thresholds: List, 
                         metric: str='metric') -> Dict[int, NDArray]:
        ''' Do single-frame matching for one class.
        
        Args:
            pred_vectors (List): List[vector(ndarray) (different length)], 
            scores (List): List[score(float)]
            groundtruth (List): List of vectors
            thresholds (List): List of thresholds
        
        Returns:
            tp_fp_score_by_thr (Dict): matching results at different thresholds
                e.g. {0.5: (M, 2), 1.0: (M, 2), 1.5: (M, 2)}
        '''

        pred_lines = []

        # interpolate predictions
        for vector in pred_vectors:
            vector = np.array(vector)
            vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            pred_lines.append(vector_interp)
        if pred_lines:
            pred_lines = np.stack(pred_lines)
        else:
            pred_lines = np.zeros((0, INTERP_NUM, 2))

        # interpolate groundtruth
        gt_lines = []
        for vector in groundtruth:
            vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            gt_lines.append(vector_interp)
        if gt_lines:
            gt_lines = np.stack(gt_lines)
        else:
            gt_lines = np.zeros((0, INTERP_NUM, 2))
        
        scores = np.array(scores)
        tp_fp_list = instance_match(pred_lines, scores, gt_lines, thresholds, metric) # (M, 2)
        tp_fp_score_by_thr = {}
        for i, thr in enumerate(thresholds):
            tp, fp = tp_fp_list[i]
            tp_fp_score = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
            tp_fp_score_by_thr[thr] = tp_fp_score
        
        return tp_fp_score_by_thr # {0.5: (M, 2), 1.0: (M, 2), 1.5: (M, 2)}
        
    def evaluate(self, 
                 result_path: str, 
                 metric: str='chamfer', 
                 logger: Optional[Logger]=None) -> Dict[str, float]:
        ''' Do evaluation for a submission file and print evalution results to `logger` if specified.
        The submission will be aligned by tokens before evaluation. We use multi-worker to speed up.
        
        Args:
            result_path (str): path to submission file
            metric (str): distance metric. Default: 'chamfer'
            logger (Logger): logger to print evaluation result, Default: None
        
        Returns:
            new_result_dict (Dict): evaluation results. AP by categories.
        '''
        
        results = mmcv.load(result_path)
        results = results['results']
        
        # re-group samples and gt by label
        samples_by_cls = {label: [] for label in self.id2cat.keys()}

        # align by token
        for token, gt in self.gts.items():
            if token in results.keys():
                pred = results[token]
            else:
                pred = {'vectors': [], 'scores': [], 'labels': []}
            
            # for every sample
            vectors_by_cls = {label: [] for label in self.id2cat.keys()}
            scores_by_cls = {label: [] for label in self.id2cat.keys()}

            for i in range(len(pred['labels'])):
                # i-th pred line in sample
                label = pred['labels'][i]
                vector = pred['vectors'][i]
                score = pred['scores'][i]

                vectors_by_cls[label].append(vector)
                scores_by_cls[label].append(score)

            for label in self.id2cat.keys():
                new_sample = (vectors_by_cls[label], scores_by_cls[label], gt[label])
                samples_by_cls[label].append(new_sample)

        result_dict = self._evaluate_samples(samples_by_cls, metric, logger, title='all')

        new_result_dict = {}
        for name in self.cat2id:
            new_result_dict[name] = result_dict[name]['AP']
        new_result_dict['mAP'] = result_dict['mAP']

        if self.eval_regions:
            # ego heading is +x: ahead half is x in [0, xmax], rear half is x in [xmin, 0]
            half_x, half_y = self.roi_size[0] / 2, self.roi_size[1] / 2
            regions = {
                'ahead': box(0, -half_y, half_x, half_y),
                'rear': box(-half_x, -half_y, 0, half_y),
            }
            prefix = {'ahead': 'A', 'rear': 'R'}
            for region_name, region in regions.items():
                region_samples = {
                    label: [clip_sample_to_region(*sample, region) for sample in samples]
                    for label, samples in samples_by_cls.items()}
                region_result = self._evaluate_samples(
                    region_samples, metric, logger, title=region_name)
                for name in self.cat2id:
                    new_result_dict[f'{prefix[region_name]}-{name}'] = region_result[name]['AP']
                new_result_dict[f'{prefix[region_name]}-mAP'] = region_result['mAP']

        return new_result_dict

    def _evaluate_samples(self,
                          samples_by_cls: Dict[int, List],
                          metric: str,
                          logger: Optional[Logger],
                          title: str) -> Dict:
        ''' Compute AP per category for samples grouped by label.

        Args:
            samples_by_cls (Dict): {label: List[(pred_lines, scores, gts)]}
            title (str): name printed above the result table

        Returns:
            result_dict (Dict): {category: {num_gts, num_preds, AP@thr, AP}, 'mAP': float}
        '''
        num_gts = {label: sum(len(s[2]) for s in samples) for label, samples in samples_by_cls.items()}
        num_preds = {label: sum(len(s[1]) for s in samples) for label, samples in samples_by_cls.items()}

        result_dict = {}

        print(f'\nevaluating {len(self.id2cat)} categories ({title})...')
        start = time()
        if self.n_workers > 0:
            pool = Pool(self.n_workers, initializer=_limit_worker_threads)
        
        sum_mAP = 0
        pbar = mmcv.ProgressBar(len(self.id2cat))
        for label in self.id2cat.keys():
            samples = samples_by_cls[label] # List[(pred_lines, scores, gts)]
            result_dict[self.id2cat[label]] = {
                'num_gts': num_gts[label],
                'num_preds': num_preds[label]
            }
            sum_AP = 0

            fn = partial(self._evaluate_single, thresholds=self.thresholds, metric=metric)
            if self.n_workers > 0:
                tpfp_score_list = pool.starmap(fn, samples)
            else:
                tpfp_score_list = []
                for sample in samples:
                    tpfp_score_list.append(fn(*sample))
            
            for thr in self.thresholds:
                tp_fp_score = [i[thr] for i in tpfp_score_list]
                tp_fp_score = np.vstack(tp_fp_score) # (num_dets, 3)
                sort_inds = np.argsort(-tp_fp_score[:, -1])

                tp = tp_fp_score[sort_inds, 0] # (num_dets,)
                fp = tp_fp_score[sort_inds, 1] # (num_dets,)
                tp = np.cumsum(tp, axis=0)
                fp = np.cumsum(fp, axis=0)
                eps = np.finfo(np.float32).eps
                recalls = tp / np.maximum(num_gts[label], eps)
                precisions = tp / np.maximum((tp + fp), eps)

                AP = average_precision(recalls, precisions, 'area')
                sum_AP += AP
                result_dict[self.id2cat[label]].update({f'AP@{thr}': AP})

            pbar.update()
            
            AP = sum_AP / len(self.thresholds)
            sum_mAP += AP

            result_dict[self.id2cat[label]].update({f'AP': AP})
        
        if self.n_workers > 0:
            pool.close()
        
        mAP = sum_mAP / len(self.id2cat.keys())
        result_dict.update({'mAP': mAP})
        
        print(f"finished in {time() - start:.2f}s")

        # print results
        table = prettytable.PrettyTable(['category', 'num_preds', 'num_gts'] + 
                [f'AP@{thr}' for thr in self.thresholds] + ['AP'])
        for label in self.id2cat.keys():
            table.add_row([
                self.id2cat[label], 
                result_dict[self.id2cat[label]]['num_preds'],
                result_dict[self.id2cat[label]]['num_gts'],
                *[round(result_dict[self.id2cat[label]][f'AP@{thr}'], 4) for thr in self.thresholds],
                round(result_dict[self.id2cat[label]]['AP'], 4),
            ])
        
        from mmcv.utils import print_log
        print_log(f'\n[{title}]\n'+str(table), logger=logger)
        mAP_normal = 0
        for label in self.id2cat.keys():
            for thr in self.thresholds:
                mAP_normal += result_dict[self.id2cat[label]][f'AP@{thr}']
        mAP_normal = mAP_normal / 9

        print_log(f'[{title}] mAP_normal = {mAP_normal:.4f}\n', logger=logger)
        # print_log(f'mAP_hard = {mAP_easy:.4f}\n', logger=logger)

        return result_dict


def _clip_lines(lines: List, region) -> List[NDArray]:
    ''' Clip polylines to a region and return the pieces inside it.
    A line that crosses the region border twice becomes several pieces. '''
    pieces = []
    for line in lines:
        line = np.asarray(line)[:, :2]
        geom = LineString(line).intersection(region)
        if geom.is_empty:
            continue
        if geom.geom_type == 'LineString':
            parts = [geom]
        elif geom.geom_type in ('MultiLineString', 'GeometryCollection'):
            parts = [g for g in geom.geoms if g.geom_type == 'LineString']
        else: # Point, MultiPoint: touches the border only
            parts = []
        for part in parts:
            if part.length > 1e-3:
                pieces.append(np.array(part.coords))
    return pieces


def clip_sample_to_region(pred_vectors: List, scores: List, groundtruth: List, region):
    ''' Clip predictions and gts of one sample to a region (A-mAP / R-mAP).
    Each clipped piece of a prediction keeps the score of the original prediction. '''
    new_preds, new_scores = [], []
    for vector, score in zip(pred_vectors, scores):
        pieces = _clip_lines([vector], region)
        new_preds.extend(pieces)
        new_scores.extend([score] * len(pieces))
    new_gts = _clip_lines(groundtruth, region)
    return new_preds, new_scores, new_gts
