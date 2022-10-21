"""
This modules defines the adversaries proposed in [paper_url].
Authors: Ryan Sheatsley & Blaine Hoak
Fri Oct 21 2022
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
import sklearn.preprocessing  # Preprocessing and Normalization


class Adversary:
    """ """

    def minmaxscale(self, x, transform=True):
        """
        This method serves as a wrapper for sklearn.preprocessing.MinMaxScaler.
        Specifically, it maps inputs to [0, 1] (as some techniques assume this
        range, e.g., change of variables). Importantly, since these scalers
        always return numpy arrays, this method additionally casts these inputs
        back as PyTorch tensors with the original data type.

        :param x: the batch of inputs to scale
        :type x: PyTorch FloatTensor object (n, m)
        :param transform: performs the transformation if true; the inverse if false
        :return: a batch of scaled inputs
        :rtype: PyTorch FloatTensor (n, m)
        """
        if transform:
            self.scaler = sklearn.preprocessing.MinMaxScaler()
            self.scaler.dtype = x.dtype
            x = self.scaler.fit_transform(x)
        else:
            x = self.scaler.inverse_transform(x)
        return torch.from_numpy(x).to(self.scaler.dtype)


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
