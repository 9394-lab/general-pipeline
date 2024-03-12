import torch.nn as nn


class AbstractModel(nn.Module):

    def __init__(self, config, data_feature):
        nn.Module.__init__(self)

    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """

    def calculate_loss(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """


class AbstractTrafficStateModel(AbstractModel):

    def __init__(self, config, data_feature):
        self.data_feature = data_feature
        super().__init__(config, data_feature)

    def predict(self, batch):
        """

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """

    def calculate_loss(self, batch):
        """

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """

    def get_data_feature(self):
        return self.data_feature
