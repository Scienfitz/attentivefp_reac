import datetime
import torch
import copy
import logging
logger = logging.getLogger(__name__)

class EarlyStopping(object):
    """Early stop tracker

    Save model checkpoint when observing a performance improvement on
    the validation set and early stop if improvement has not been
    observed for a particular number of epochs.

    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.

    """
    def __init__(self, mode='higher', patience=10, metric=None):
        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score']:
                logger.info('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                logger.info('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.best_parameters = None
        self.best_score = None
        self.best_step = 1
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.

        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.

        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.

        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.

        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.best_step += self.counter + 1
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.

        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        self.best_parameters = copy.deepcopy(model.state_dict())

    def load_checkpoint(self, model):
        '''Load the latest checkpoint

        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(self.best_parameters)