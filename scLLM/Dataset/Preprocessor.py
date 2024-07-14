from typing import Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np, scanpy as sc
from scipy.sparse import issparse
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData

from scLLM import logger
from scLLM.Dataset.paras import Dataset_para
class scPreprocessor:
    def __init__(self,para:Dataset_para):
        self.para=para

    def __call__(self,adata:AnnData, batch_key: Optional[str] = None):
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        # 
        logger.info("Preprocessing data with shape: {} ...".format(adata.shape))
        key_to_process = self.para.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # step 1: filter genes
        if self.para.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.para.filter_gene_by_counts
                if isinstance(self.para.filter_gene_by_counts, int)
                else None,
                inplace=True,
            )
            logger.info(f"Filtered genes: {adata.n_vars}")

        # step 2: filter cells
        if isinstance(self.para.filter_cell_by_counts, int):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.para.filter_cell_by_counts
                if isinstance(self.para.filter_cell_by_counts, int)
                else None,
                inplace=True,
            )
            logger.info(f"Filtered cells: {adata.n_obs}")
        # step 3: normalize total,this part will automatically 
        # added into the adata object(inplace=False)
        if self.para.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.para.normalize_total
                if isinstance(self.para.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.para.result_normed_key \
                if self.para.result_normed_key is not None else key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        # copy == False in default adata will be automatically changed
        if self.para.log1p:
            log1p_paras = {}
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.para.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.para.result_log1p_key,
                )
                key_to_process = self.para.result_log1p_key
                log1p_paras["layer"] = key_to_process

            if self.para.log1p_base:
                log1p_paras["base"] = self.para.log1p_base
            sc.pp.log1p(adata, **log1p_paras)

        # step 5: subset hvg
        if self.para.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.para.hvg_use_key,
                n_top_genes=self.para.subset_hvg
                if isinstance(self.para.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.para.hvg_flavor,
                subset=True,
                inplace=True, # this will automatically change the adata object
            )

        # step 6: binning
        if self.para.binning:
            logger.info("Binning data ...")
            if not isinstance(self.para.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.para.binning)
                )
            n_bins = self.para.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            for row in layer_data:
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = np.digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.para.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

        if self.para.preprocessed_loc is not None:
            logger.info(f"save preprocessed data to {self.para.preprocessed_loc}")
            adata.write(self.para.preprocessed_loc)
        logger.info("Preprocessing finished.")
        return adata
    
    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True
 