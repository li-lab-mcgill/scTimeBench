from scTimeBench.shared.dataset.base import BaseDatasetFilter


class CopySplitFilter(BaseDatasetFilter):
    def __init__(self, config):
        super().__init__(config)
        self.splits = True

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {}

    def filter(self, ann_data, **kwargs):
        """
        Give two copies of the dataset as train and test sets.
        Useful for metrics that do not require train/test splits,
        and for datasets that are small.
        """
        train_data = ann_data.copy()
        test_data = ann_data.copy()

        return train_data, test_data
