# stats_test_from_scratch
One issue with Python is that there is no unified source for statistical tests like there is for ML with scikit-learn or Deep Learning with Keras. 

You could have statistical tests in Scipy, Statsmodels, Numpy and Pandas; but it's not clear which libraries carry which tests. Additionally, there are some tests that you can find in R that aren't currently supported by these libraries.

My goal is two fold:
1) Implement the statistical tests from scratch, with an emphasis on making the code as presentable and easy-to-understand as possible. This is so anyone can understand exactly how what the test is measuring and how it is measuring it, even if it means we are sacrificing computational speed in the process.
2) Identify where in Python you can find these test(s), if at all. That way, for those who want the fastest implementation, they'll understand where to find the test.

# Statistical Tests currently supported and where to find them:
## Sample Tests
1) One and two sample Z Tests: Statsmodels through [ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html).
2) One and two sample T Tests: Scipy through [ttest_1samp](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_1samp.html) and [ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
3) Wilcoxon Rank-Sum Test: Scipy through [wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
4) Mann-Whitney-U Test: Scipy through [mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)

## Categorical Tests
1) Chi Square Test: Scipy through [chisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
2) Goodness of Fit Test: Scipy through [chi2_contingency](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html)
3) Fisher Test: Scipy through [fisher_exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)
4) McNemar Test: Statsmodels through [mcnemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
