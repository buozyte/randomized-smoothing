import torch
from scipy.stats import binom_test, norm
from statsmodels.stats.proportion import proportion_confint
from torch import nn


# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class RandSmoothedClassifier(nn.Module):
    """
    Define a randomly smoothed classifier based on a chosen base classifier.
    """

    def __init__(self, base_classifier, c, sigma, device, abstain=-1):
        """
        Initialize the randomly smoothed classifier

        :param base_classifier: a base classifier
        :param c: number of possible classes
        :param sigma: parameter used to define the variance in a normal distribution
        :param device: pytorch device handling
        :param abstain: value to be returned when smoothed classifier should abstain
        """

        super().__init__()

        self.base_classifier = base_classifier
        self.c = c
        self.sigma = sigma
        self.device = device
        self.abstain = abstain

    def sample_under_noise(self, x, n, batch_size):
        """
        Compute the class counts of the predictions of n perturbed inputs based on the base classifier.

        :param x: current input (shape: [1, channels, image_width, image_height])
        :param n: number of samples
        :param batch_size: size of the batches in which the evaluation should be performed
        :return: tensor containing the class counts
        """

        # create tensor containing all classes (from 0 to c-1)
        classes = torch.arange(self.c).to(self.device)

        class_counts = torch.zeros([self.c]).to(self.device)

        # sample and evaluate perturbed samples in batches
        remaining_samples = n
        while remaining_samples > 0:
            # define suitable current batch size
            current_batch = min(batch_size, remaining_samples)
            remaining_samples -= current_batch

            # create tensor containing n times the sample x
            repeat_x_n_times = x.repeat(current_batch, 1, 1, 1)

            # generate and evaluate (/classify) the perturbed samples
            perturbed_predictions = self.forward(repeat_x_n_times)

            # predicted class for one sample = index of highest value
            predicted_classes = (perturbed_predictions.argmax(dim=1, keepdim=True) == classes)

            # sum over each column -> "count number of samples predicted as (column) i"
            class_counts += predicted_classes.sum(dim=0)

        return class_counts

    def predict(self, x, n, alpha, batch_size):
        """
        Evaluate the smoothed classifier g (based on the base classifier) at x.

        :param x: current input (shape: [1, channels, image_width, image_height])
        :param n: number of samples
        :param alpha: probability that function will return a class other than g(x)
        :param batch_size: size of the batches in which the evaluation should be performed
        :return: predicted class of input
        """

        # set model to evaluation mode
        self.base_classifier.eval()

        counts = self.sample_under_noise(x, n, batch_size)

        # get indices and values of the top two values in counts
        top_2_counts = torch.topk(counts, k=2)
        # c_a, c_b = top_2_counts.indices
        n_a, n_b = top_2_counts.values

        if binom_test(n_a, n_a + n_b, 0.5) <= alpha:
            # return c_a
            return top_2_counts.indices[0].item()
        return self.abstain

    def certify(self, x, n_0, n, alpha, batch_size):
        """
        Certify the robustness of the smoothed classifier g (based on f) around x

        :param x: current input (shape: [1, channels, image_width, image_height])
        :param n_0: number of samples to estimate the class (c_a)
        :param n: number of samples to estimate a lower bound on the probability for the class (p_a)
        :param alpha: probability that function will return a class other than g(x)
        :param batch_size: size of the batches in which the evaluation should be performed
        :return: tuple of predicted class of input and according robust radius
        """

        # set model to evaluation mode
        self.base_classifier.eval()

        # generate samples to estimate/guess c_a
        counts_0 = self.sample_under_noise(x, n_0, batch_size)
        # determine most likely class
        c_a = counts_0.argmax().item()  # .item() = get value of tensor

        # generate samples to estimate/guess lower bound of p_a
        counts = self.sample_under_noise(x, n, batch_size)
        
        # note: used code from MLGS as I couldn't find a working implementation to get a lower bound
        # didn't work: p_a = proportion_confint(..., method='binom_test')
        p_a = proportion_confint(counts[c_a].cpu(), nobs=n, alpha=2 * alpha, method="beta")[0]

        if p_a > 0.5:
            return c_a, self.sigma * norm.ppf(p_a)  # norm.ppf = Phi^(-1)
        return self.abstain, 0.0

    def forward(self, x):
        """
        Generate and classify the perturbed samples x using the base classifier.

        :param x: current input (shape: [batch_size, channels, image_width, image_height])
        :return: evaluation of the perturbed input samples
        """

        # for each value in sample x create n normal distributed noise samples
        # (transform standard normal distributed variable to normal distributed variable with
        #  mean=mu and variance=sigma^2 by using: y = sigma*x+mu with standard normal distributed variable x)
        epsilon = torch.randn_like(x) * self.sigma  # + 0

        # generate and evaluate (/classify) the perturbed samples
        return self.base_classifier(x + epsilon)
